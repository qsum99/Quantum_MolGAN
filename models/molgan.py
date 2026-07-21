"""
models/molgan.py
Classical MolGAN — Generator, Discriminator, Reward Network.

Architecture follows:
  De Cao & Kipf (2018) "MolGAN: An implicit generative model
  for small molecular graphs"  https://arxiv.org/abs/1805.11973

Generator  : noise z → (adjacency logits, node logits)
Discriminator: graph → scalar (WGAN)
Reward      : graph → drug-likeness scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.dataset import NUM_ATOMS, NUM_ATOM_TYPES, NUM_BOND_TYPES


# ── Graph Convolution ─────────────────────────────────────────────────────────

class GraphConv(nn.Module):
    """
    Relational GCN layer.
    Aggregates neighbour features weighted by bond type.
    """

    def __init__(self, in_dim, out_dim, num_relations=NUM_BOND_TYPES):
        super().__init__()
        self.num_relations = num_relations
        # One linear per bond type
        self.linears = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(num_relations)
        ])
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj_one_hot):
        """
        x           : (B, N, in_dim)
        adj_one_hot : (B, R, N, N)  — one-hot bond type adjacency
        """
        out = self.self_lin(x)

        for r in range(self.num_relations):
            # (B, N, N) × (B, N, in_dim) → (B, N, in_dim)
            agg = torch.bmm(adj_one_hot[:, r], x)
            out = out + self.linears[r](agg)

        # BatchNorm over node dimension
        B, N, D = out.shape
        out = self.bn(out.reshape(B * N, D)).reshape(B, N, D)
        return F.relu(out)


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    MLP that maps noise z → (adjacency logits, node logits).

    adj_logits  : (B, N, N, NUM_BOND_TYPES) — bond type per pair
    node_logits : (B, N, NUM_ATOM_TYPES)    — atom type per position
    """

    def __init__(self, z_dim=8, hidden=128):
        super().__init__()
        self.z_dim = z_dim

        n_adj  = NUM_ATOMS * NUM_ATOMS * NUM_BOND_TYPES
        n_node = NUM_ATOMS * NUM_ATOM_TYPES

        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
        )
        self.adj_head  = nn.Linear(hidden * 2, n_adj)
        self.node_head = nn.Linear(hidden * 2, n_node)

    def forward(self, z):
        B = z.size(0)
        h = self.net(z)

        adj_logits  = self.adj_head(h).reshape(B, NUM_ATOMS, NUM_ATOMS, NUM_BOND_TYPES)
        node_logits = self.node_head(h).reshape(B, NUM_ATOMS, NUM_ATOM_TYPES)

        # Symmetrise adjacency (undirected graph)
        adj_logits = (adj_logits + adj_logits.transpose(1, 2)) / 2

        return adj_logits, node_logits

    def sample(self, z, tau=1.0, hard=True):
        """
        Returns (adj, nodes) via Gumbel-Softmax.
        hard=True  → discrete argmax (for inference / discriminator input)
        hard=False → soft one-hot (for gradient flow through G)

        adj   : (B, N, N)  long  (hard) or (B, N, N, R) float (soft)
        nodes : (B, N)     long  (hard) or (B, N, A)    float (soft)
        """
        adj_logits, node_logits = self.forward(z)

        if hard:
            adj   = adj_logits.argmax(dim=-1)
            nodes = node_logits.argmax(dim=-1)
        else:
            # Gumbel-Softmax for differentiable sampling
            adj   = F.gumbel_softmax(adj_logits,   tau=tau, hard=False, dim=-1)
            nodes = F.gumbel_softmax(node_logits,  tau=tau, hard=False, dim=-1)

        return adj, nodes


# ── Discriminator ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    WGAN discriminator.
    Graph → scalar (higher = more real).
    Uses 3 relational GCN layers then global mean pool.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.embed_atom = nn.Embedding(NUM_ATOM_TYPES + 1, hidden)

        self.gcn1 = GraphConv(hidden, hidden)
        self.gcn2 = GraphConv(hidden, hidden)
        self.gcn3 = GraphConv(hidden, hidden)

        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, adj, nodes):
        """
        adj   : (B, N, N)  long — bond type indices
        nodes : (B, N)     long — atom type indices
        """
        B, N = nodes.shape

        # One-hot adjacency per bond type: (B, R, N, N)
        adj_oh = F.one_hot(adj, num_classes=NUM_BOND_TYPES).float()
        adj_oh = adj_oh.permute(0, 3, 1, 2)

        x = self.embed_atom(nodes)          # (B, N, hidden)
        x = self.gcn1(x, adj_oh)
        x = self.gcn2(x, adj_oh)
        x = self.gcn3(x, adj_oh)

        x = x.mean(dim=1)                  # global mean pool (B, hidden)
        return self.fc(x).squeeze(-1)      # (B,)


# ── Reward Network ────────────────────────────────────────────────────────────

class RewardNetwork(nn.Module):
    """
    Predicts drug-likeness score from graph.
    Same GCN backbone as discriminator but outputs [0, 1].
    Trained with supervision from RDKit QED scores.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.embed_atom = nn.Embedding(NUM_ATOM_TYPES + 1, hidden)

        self.gcn1 = GraphConv(hidden, hidden)
        self.gcn2 = GraphConv(hidden, hidden)

        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, adj, nodes):
        B, N = nodes.shape
        adj_oh = F.one_hot(adj, num_classes=NUM_BOND_TYPES).float()
        adj_oh = adj_oh.permute(0, 3, 1, 2)

        x = self.embed_atom(nodes)
        x = self.gcn1(x, adj_oh)
        x = self.gcn2(x, adj_oh)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)      # (B,) in [0, 1]
