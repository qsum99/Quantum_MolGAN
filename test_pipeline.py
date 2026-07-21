"""
test_pipeline.py
Smoke test — validates the full pipeline without training.
Run this first to confirm everything is wired correctly.

  python test_pipeline.py
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F

print("=" * 60)
print("  QGAN Drug Discovery — Phase 1 Pipeline Test")
print("=" * 60)


# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n[1/6] Checking imports...")
try:
    from rdkit import Chem
    from rdkit.Chem import QED
    print("  ✓ RDKit")
except ImportError as e:
    print(f"  ✗ RDKit: {e}"); sys.exit(1)

try:
    from data.dataset import (mol_to_graph, graph_to_mol,
                               NUM_ATOMS, NUM_ATOM_TYPES, NUM_BOND_TYPES,
                               ATOM_TYPES)
    print("  ✓ data.dataset")
except Exception as e:
    print(f"  ✗ data.dataset: {e}"); sys.exit(1)

try:
    from models.molgan import Generator, Discriminator, RewardNetwork
    print("  ✓ models.molgan")
except Exception as e:
    print(f"  ✗ models.molgan: {e}"); sys.exit(1)

try:
    from utils.metrics import evaluate_batch, score_molecule
    print("  ✓ utils.metrics")
except Exception as e:
    print(f"  ✗ utils.metrics: {e}"); sys.exit(1)


# ── 2. Molecule → Graph → Molecule roundtrip ──────────────────────────────────
print("\n[2/6] Molecule ↔ graph roundtrip...")
test_smiles = ['CC(=O)O', 'c1ccccc1', 'CCO', 'CN', 'C=O', 'CC#N', 'C1CCCCC1']

ok = 0
for smi in test_smiles:
    mol  = Chem.MolFromSmiles(smi)
    adj, nodes = mol_to_graph(mol)
    if adj is None:
        print(f"  ⚠  Skipped (too large): {smi}")
        continue
    mol2 = graph_to_mol(adj, nodes)
    if mol2 is not None:
        smi2 = Chem.MolToSmiles(mol2)
        print(f"  ✓ {smi:15s} → adj{adj.shape} → {smi2}")
        ok += 1
    else:
        print(f"  ✗ {smi} — failed to reconstruct")

print(f"  Roundtrip: {ok}/{len(test_smiles)} succeeded")


# ── 3. Model forward pass ─────────────────────────────────────────────────────
print("\n[3/6] Model forward pass (B=4)...")
B, Z = 4, 8

G = Generator(z_dim=Z, hidden=64)
D = Discriminator(hidden=32)
R = RewardNetwork(hidden=32)

z = torch.randn(B, Z)
adj_logits, node_logits = G(z)
print(f"  G output | adj_logits: {tuple(adj_logits.shape)} | node_logits: {tuple(node_logits.shape)}")

fake_adj   = adj_logits.argmax(dim=-1)
fake_nodes = node_logits.argmax(dim=-1)
print(f"  Sampled  | adj: {tuple(fake_adj.shape)} | nodes: {tuple(fake_nodes.shape)}")

d_out = D(fake_adj, fake_nodes)
print(f"  D output | scores: {d_out.detach().numpy().round(3)}")

r_out = R(fake_adj, fake_nodes)
print(f"  R output | rewards: {r_out.detach().numpy().round(3)}")


# ── 4. Gradient flow ──────────────────────────────────────────────────────────
print("\n[4/6] Gradient flow check...")
loss = -d_out.mean() - r_out.mean()
loss.backward()

g_params_with_grad = sum(1 for p in G.parameters() if p.grad is not None)
print(f"  G params with gradients: {g_params_with_grad}/{len(list(G.parameters()))}")
if g_params_with_grad > 0:
    print("  ✓ Gradients flowing through generator")
else:
    print("  ✗ No gradients! Check training loop")


# ── 5. Batch evaluation ───────────────────────────────────────────────────────
print("\n[5/6] Metrics evaluation on 64 random molecules...")
G.eval()
with torch.no_grad():
    z = torch.randn(64, Z)
    adj_logits, node_logits = G(z)
    batch_adj   = adj_logits.argmax(dim=-1)
    batch_nodes = node_logits.argmax(dim=-1)

metrics = evaluate_batch(batch_adj, batch_nodes)
print(f"  Validity   : {metrics['validity']:.1%}")
print(f"  Uniqueness : {metrics['uniqueness']:.1%}")
print(f"  Valid mols : {metrics['n_valid']}/{metrics['n_total']}")
if metrics['qed_mean']:
    print(f"  QED mean   : {metrics['qed_mean']:.3f}")

print("  (Low validity on untrained model is expected!)")


# ── 6. Device check ───────────────────────────────────────────────────────────
print("\n[6/6] Device availability...")
print(f"  CUDA  : {'✓ available' if torch.cuda.is_available() else '✗ not available'}")
print(f"  MPS   : {'✓ available' if torch.backends.mps.is_available() else '✗ not available'}")
print(f"  CPU   : ✓ always available")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✓ Pipeline test complete — ready to train!")
print("  Run: python train.py --epochs 30 --batch 32")
print("=" * 60 + "\n")
