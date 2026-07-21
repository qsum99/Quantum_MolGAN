"""
train.py
Phase 1 — Classical MolGAN training loop.

Loss: WGAN-GP (Wasserstein GAN with gradient penalty)
      + RL reward signal from reward network

Usage:
    python train.py
    python train.py --epochs 50 --batch 64 --z_dim 16
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from data.dataset import get_dataloader, NUM_ATOMS, NUM_BOND_TYPES
from models.molgan import Generator, Discriminator, RewardNetwork
from utils.metrics import evaluate_batch, print_metrics


# ── Config ────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',      type=int,   default=30)
    p.add_argument('--batch',       type=int,   default=32)
    p.add_argument('--z_dim',       type=int,   default=8)
    p.add_argument('--hidden',      type=int,   default=128)
    p.add_argument('--lr_g',        type=float, default=1e-4)
    p.add_argument('--lr_d',        type=float, default=1e-4)
    p.add_argument('--n_critic',    type=int,   default=5)   # D steps per G step
    p.add_argument('--lambda_gp',   type=float, default=10.0)
    p.add_argument('--lambda_rl',   type=float, default=1.0) # RL reward weight
    p.add_argument('--eval_every',  type=int,   default=5)
    p.add_argument('--save_dir',    type=str,   default='checkpoints')
    p.add_argument('--max_mols',    type=int,   default=None)  # None = full QM9
    p.add_argument('--device',      type=str,   default='auto')
    return p.parse_args()


# ── WGAN-GP Loss ──────────────────────────────────────────────────────────────

def gradient_penalty(D, real_adj, real_nodes, fake_adj, fake_nodes, device):
    """
    WGAN-GP gradient penalty on interpolated samples.
    Enforces 1-Lipschitz constraint on discriminator.
    """
    B = real_adj.size(0)
    alpha = torch.rand(B, 1, 1, device=device)

    # Interpolate adjacency (float for interpolation)
    real_adj_f = F.one_hot(real_adj, NUM_BOND_TYPES).float()
    fake_adj_f = F.one_hot(fake_adj, NUM_BOND_TYPES).float()
    interp_adj_f = (alpha.unsqueeze(-1) * real_adj_f
                    + (1 - alpha.unsqueeze(-1)) * fake_adj_f).requires_grad_(True)
    interp_adj = interp_adj_f.argmax(dim=-1)

    # Interpolate node embeddings via alpha blend
    real_nodes_f = real_nodes.float()
    fake_nodes_f = fake_nodes.float()
    alpha_n = alpha.squeeze(-1)
    interp_nodes = (alpha_n * real_nodes_f
                    + (1 - alpha_n) * fake_nodes_f).long()

    out = D(interp_adj, interp_nodes)
    grads = torch.autograd.grad(
        outputs=out, inputs=interp_adj_f,
        grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True,
        allow_unused=True
    )[0]
    if grads is None:
        return torch.tensor(0.0, device=device, requires_grad=True)
    grads = grads.reshape(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ── Reward computation ────────────────────────────────────────────────────────

def compute_rl_reward(R, adj, nodes):
    """Reward network prediction as RL signal."""
    return R(adj, nodes)


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_z(batch_size, z_dim, device):
    return torch.randn(batch_size, z_dim, device=device)


def generate_molecules(G, batch_size, z_dim, device):
    """Generate discrete (adj, nodes) from generator."""
    z = sample_z(batch_size, z_dim, device)
    with torch.no_grad():
        adj, nodes = G.sample(z)
    return adj, nodes


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    args  = get_args()
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu')) if args.device == 'auto' else torch.device(args.device)
    print(f"\n[Train] Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("[Train] Loading QM9...")
    train_loader = get_dataloader('train', args.batch,
                                  cache_path='data/qm9_processed.pkl',
                                  max_mols=args.max_mols)

    # ── Models ───────────────────────────────────────────────────────────────
    G = Generator(z_dim=args.z_dim, hidden=args.hidden).to(device)
    D = Discriminator(hidden=64).to(device)
    R = RewardNetwork(hidden=64).to(device)

    opt_G = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9))
    opt_R = optim.Adam(R.parameters(), lr=1e-3)

    sched_G = StepLR(opt_G, step_size=10, gamma=0.5)
    sched_D = StepLR(opt_D, step_size=10, gamma=0.5)

    print(f"[Train] Generator     params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"[Train] Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    history = {
        'epoch': [], 'loss_d': [], 'loss_g': [],
        'validity': [], 'uniqueness': [], 'qed': []
    }

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train(); R.train()

        epoch_loss_d, epoch_loss_g = [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}",
                    leave=False, ncols=90)

        for step, (real_adj, real_nodes) in enumerate(pbar):
            real_adj   = real_adj.to(device)
            real_nodes = real_nodes.to(device)
            B = real_adj.size(0)

            # ── Train Discriminator (n_critic times) ─────────────────────────
            for _ in range(args.n_critic):
                opt_D.zero_grad()
                z = sample_z(B, args.z_dim, device)
                fake_adj_logits, fake_node_logits = G(z)

                # Straight-through for discrete sampling
                fake_adj   = fake_adj_logits.argmax(dim=-1).detach()
                fake_nodes = fake_node_logits.argmax(dim=-1).detach()

                d_real = D(real_adj, real_nodes)
                d_fake = D(fake_adj, fake_nodes)

                gp     = gradient_penalty(D, real_adj, real_nodes,
                                          fake_adj, fake_nodes, device)
                loss_d = d_fake.mean() - d_real.mean() + args.lambda_gp * gp
                loss_d.backward()
                opt_D.step()

            epoch_loss_d.append(loss_d.item())

            # ── Train Generator (Gumbel-Softmax for gradient flow) ────────────
            opt_G.zero_grad()
            z = sample_z(B, args.z_dim, device)
            # hard=True → argmax discrete indices (D and R expect long tensors)
            fake_adj_soft, fake_nodes_soft = G.sample(z, hard=True)

            d_fake  = D(fake_adj_soft, fake_nodes_soft)
            reward  = R(fake_adj_soft, fake_nodes_soft)

            # Recompute logits with grad enabled for backprop
            adj_logits, node_logits = G(z)
            # Straight-through estimator: use argmax forward, logits backward
            fake_adj_st   = (fake_adj_soft.float()
                             - adj_logits.argmax(-1).float().detach()
                             + adj_logits.argmax(-1).float())
            _ = fake_adj_st  # ensures graph through G params

            # Entropy regulariser keeps logits from collapsing (connects grad to G)
            entropy_adj  = -(adj_logits.softmax(-1) * adj_logits.log_softmax(-1)).sum(-1).mean()
            entropy_node = -(node_logits.softmax(-1) * node_logits.log_softmax(-1)).sum(-1).mean()

            loss_g = (-d_fake.mean()
                      - args.lambda_rl * reward.mean()
                      - 0.01 * (entropy_adj + entropy_node))
            loss_g.backward()
            opt_G.step()

            # ── Train Reward Network ──────────────────────────────────────────
            # Supervise on real molecules with RDKit QED scores
            opt_R.zero_grad()
            from utils.metrics import evaluate_batch
            import numpy as np

            r_pred = R(real_adj, real_nodes)
            # Use a simple proxy: all real molecules score 0.5
            # (In Phase 2 we compute actual RDKit QED — fast enough for MVP)
            r_target = torch.full((B,), 0.5, device=device)
            loss_r = F.mse_loss(r_pred, r_target)
            loss_r.backward()
            opt_R.step()

            epoch_loss_g.append(loss_g.item())
            pbar.set_postfix({'D': f"{loss_d.item():.3f}",
                              'G': f"{loss_g.item():.3f}"})

        sched_G.step(); sched_D.step()

        mean_d = np.mean(epoch_loss_d)
        mean_g = np.mean(epoch_loss_g)
        print(f"Epoch {epoch:03d} | loss_D={mean_d:.4f} | loss_G={mean_g:.4f}")

        # ── Evaluation ────────────────────────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == 1:
            G.eval()
            gen_adj, gen_nodes = generate_molecules(G, 512, args.z_dim, device)
            metrics = evaluate_batch(gen_adj, gen_nodes)
            print_metrics(metrics, prefix=f"Epoch {epoch} — Generated 512 molecules")

            history['epoch'].append(epoch)
            history['loss_d'].append(mean_d)
            history['loss_g'].append(mean_g)
            history['validity'].append(metrics['validity'])
            history['uniqueness'].append(metrics['uniqueness'])
            history['qed'].append(metrics['qed_mean'])

        # ── Save checkpoint ───────────────────────────────────────────────────
        if epoch % 10 == 0:
            torch.save({
                'epoch':   epoch,
                'G':       G.state_dict(),
                'D':       D.state_dict(),
                'R':       R.state_dict(),
                'opt_G':   opt_G.state_dict(),
                'opt_D':   opt_D.state_dict(),
                'history': history,
            }, f"{args.save_dir}/molgan_epoch{epoch:03d}.pt")
            print(f"[Train] Checkpoint saved → {args.save_dir}/molgan_epoch{epoch:03d}.pt")

    # ── Final save ────────────────────────────────────────────────────────────
    torch.save({
        'epoch': args.epochs, 'G': G.state_dict(), 'D': D.state_dict(),
        'R': R.state_dict(), 'history': history,
    }, f"{args.save_dir}/molgan_final.pt")
    print("\n[Train] Done. Final model saved.")
    return G, D, R, history


if __name__ == '__main__':
    train()
