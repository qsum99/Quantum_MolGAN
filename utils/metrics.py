"""
utils/metrics.py
Molecule evaluation metrics — the standard MolGAN benchmark set.

Metrics:
  validity   — % of generated graphs that are valid RDKit molecules
  uniqueness — % of valid molecules that are unique SMILES
  novelty    — % of unique molecules not in the training set
  QED        — drug-likeness score (0–1, higher is better)
  SA score   — synthetic accessibility (1–10, lower is easier to make)
  logP       — lipophilicity (Lipinski rule: -0.4 to 5.6)
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA

from data.dataset import graph_to_mol, NUM_ATOM_TYPES

try:
    from rdkit.Contrib.SA_Score import sascorer
    HAS_SA = True
except ImportError:
    HAS_SA = False


# ── SA Score ─────────────────────────────────────────────────────────────────

def sa_score(mol):
    """Synthetic accessibility score 1 (easy) – 10 (hard)."""
    if not HAS_SA:
        return None
    try:
        return sascorer.calculateScore(mol)
    except Exception:
        return None


# ── Per-molecule scores ───────────────────────────────────────────────────────

def score_molecule(mol):
    """
    Returns dict of property scores for a single RDKit mol.
    Returns None values on failure.
    """
    if mol is None:
        return {'qed': None, 'sa': None, 'logp': None, 'valid': False}
    try:
        Chem.SanitizeMol(mol)
        return {
            'valid': True,
            'smiles': Chem.MolToSmiles(mol),
            'qed':   QED.qed(mol),
            'sa':    sa_score(mol),
            'logp':  Descriptors.MolLogP(mol),
        }
    except Exception:
        return {'qed': None, 'sa': None, 'logp': None, 'valid': False}


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_batch(adj_batch, node_batch, train_smiles_set=None):
    """
    Evaluate a batch of generated (adj, node) graphs.

    Args:
        adj_batch   : (B, N, N) tensor or ndarray — bond type indices
        node_batch  : (B, N)    tensor or ndarray — atom type indices
        train_smiles_set: set of training SMILES for novelty calculation

    Returns:
        dict with all metrics
    """
    if torch.is_tensor(adj_batch):
        adj_batch  = adj_batch.cpu().numpy()
        node_batch = node_batch.cpu().numpy()

    results = []
    for adj, nodes in zip(adj_batch, node_batch):
        mol = graph_to_mol(adj, nodes)
        results.append(score_molecule(mol))

    # Validity
    valid_results = [r for r in results if r['valid']]
    validity = len(valid_results) / len(results) if results else 0.0

    # Uniqueness
    valid_smiles = [r['smiles'] for r in valid_results]
    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0

    # Novelty (vs training set)
    novelty = None
    if train_smiles_set and unique_smiles:
        novel = unique_smiles - train_smiles_set
        novelty = len(novel) / len(unique_smiles)

    # Property distributions (over valid unique molecules)
    qed_scores  = [r['qed']  for r in valid_results if r['qed']  is not None]
    sa_scores   = [r['sa']   for r in valid_results if r['sa']   is not None]
    logp_scores = [r['logp'] for r in valid_results if r['logp'] is not None]

    return {
        'validity':   validity,
        'uniqueness': uniqueness,
        'novelty':    novelty,
        'qed_mean':   np.mean(qed_scores)  if qed_scores  else None,
        'qed_std':    np.std(qed_scores)   if qed_scores  else None,
        'sa_mean':    np.mean(sa_scores)   if sa_scores   else None,
        'logp_mean':  np.mean(logp_scores) if logp_scores else None,
        'n_valid':    len(valid_results),
        'n_unique':   len(unique_smiles),
        'n_total':    len(results),
    }


def print_metrics(metrics, prefix=''):
    """Pretty-print evaluation metrics."""
    print(f"\n{'─' * 50}")
    if prefix:
        print(f"  {prefix}")
    print(f"  Validity   : {metrics['validity']:.1%}  ({metrics['n_valid']}/{metrics['n_total']})")
    print(f"  Uniqueness : {metrics['uniqueness']:.1%}  ({metrics['n_unique']} unique)")
    if metrics['novelty'] is not None:
        print(f"  Novelty    : {metrics['novelty']:.1%}")
    if metrics['qed_mean'] is not None:
        print(f"  QED        : {metrics['qed_mean']:.3f} ± {metrics['qed_std']:.3f}")
    if metrics['sa_mean'] is not None:
        print(f"  SA score   : {metrics['sa_mean']:.2f}")
    if metrics['logp_mean'] is not None:
        print(f"  logP       : {metrics['logp_mean']:.2f}")
    print(f"{'─' * 50}\n")
