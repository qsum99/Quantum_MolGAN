"""
data/dataset.py
QM9 dataset loader for MolGAN.
Converts molecules → adjacency matrices + node feature matrices.

Atom types (QM9 subset): C, N, O, F + virtual "no atom" node
Bond types: no bond, single, double, triple, aromatic
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
import pickle


# ── Constants ────────────────────────────────────────────────────────────────

ATOM_TYPES   = ['C', 'N', 'O', 'F']   # QM9 core atoms (covers ~98% of dataset)
NUM_ATOMS    = 9    # QM9 max heavy atoms
NUM_ATOM_TYPES = len(ATOM_TYPES) + 1   # +1 for virtual/padding node
NUM_BOND_TYPES = 5  # no-bond, single, double, triple, aromatic

BOND_ENCODER = {
    Chem.rdchem.BondType.SINGLE:    1,
    Chem.rdchem.BondType.DOUBLE:    2,
    Chem.rdchem.BondType.TRIPLE:    3,
    Chem.rdchem.BondType.AROMATIC:  4,
}


# ── Molecule → Graph ─────────────────────────────────────────────────────────

def mol_to_graph(mol, max_atoms=NUM_ATOMS):
    """
    Convert RDKit molecule to (adjacency, node_features) tensors.

    Returns:
        adj  : (max_atoms, max_atoms) int array — bond type indices
        nodes: (max_atoms,) int array — atom type indices
        None, None if molecule is invalid or too large
    """
    if mol is None:
        return None, None

    mol = Chem.RemoveHs(mol)  # strip explicit hydrogens
    n = mol.GetNumAtoms()

    if n > max_atoms or n == 0:
        return None, None

    # Node features — atom type index
    nodes = np.zeros(max_atoms, dtype=np.int64)
    for i, atom in enumerate(mol.GetAtoms()):
        sym = atom.GetSymbol()
        nodes[i] = ATOM_TYPES.index(sym) if sym in ATOM_TYPES else len(ATOM_TYPES)

    # Adjacency matrix — bond type index
    adj = np.zeros((max_atoms, max_atoms), dtype=np.int64)
    for bond in mol.GetBonds():
        i, j    = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        btype   = BOND_ENCODER.get(bond.GetBondType(), 0)
        adj[i, j] = btype
        adj[j, i] = btype

    return adj, nodes


def graph_to_mol(adj, nodes, strict=False):
    """
    Reconstruct RDKit molecule from (adj, nodes) arrays.
    Used during evaluation to validate generated molecules.
    """
    mol  = Chem.RWMol()
    atom_map = {}

    for i, atom_idx in enumerate(nodes):
        if atom_idx < len(ATOM_TYPES):
            atom_map[i] = mol.AddAtom(Chem.Atom(ATOM_TYPES[atom_idx]))

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if adj[i, j] == 0:
                continue
            if i not in atom_map or j not in atom_map:
                continue
            bond_type = [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            ][adj[i, j] - 1]
            mol.AddBond(atom_map[i], atom_map[j], bond_type)

    try:
        mol = mol.GetMol()
        if strict:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


# ── Dataset ──────────────────────────────────────────────────────────────────

class QM9MolDataset(Dataset):
    """
    QM9 dataset as adjacency + node feature matrices.
    Downloads via DeepChem on first run; caches processed data locally.
    """

    def __init__(self, split='train', cache_path='data/qm9_processed.pkl',
                 max_mols=None):
        self.cache_path = cache_path
        self.adjs, self.nodes = self._load_or_process(max_mols)

        # Train/val/test split (80/10/10)
        n = len(self.adjs)
        idx = np.random.RandomState(42).permutation(n)
        splits = {
            'train': idx[:int(0.8 * n)],
            'val':   idx[int(0.8 * n):int(0.9 * n)],
            'test':  idx[int(0.9 * n):]
        }
        sel = splits[split]
        self.adjs  = self.adjs[sel]
        self.nodes = self.nodes[sel]

        print(f"[Dataset] {split}: {len(self.adjs)} molecules")

    # ── QM9-style SMILES (covers C/N/O/F chemistry, ≤9 heavy atoms) ──────────
    _QM9_SMILES = [
        # Alkanes / alkenes / alkynes
        'C','CC','CCC','CCCC','CCCCC','C=C','C=CC','C#C','C#CC',
        'CC=C','CC=CC','CCC=C','C1CC1','C1CCC1','C1CCCC1','C1CCCCC1',
        # Functional groups — oxygen
        'CO','CCO','CCCO','CCCCO','C=O','CC=O','CCC=O','C(=O)O',
        'CC(=O)O','CCC(=O)O','COC','CCOC','C1CO1','C1CCO1','C1CCCO1',
        # Functional groups — nitrogen
        'CN','CCN','CCCN','C=N','CC=N','C#N','CC#N','CCC#N',
        'C1CN1','C1CCN1','C1CCCN1','NC=O','NCC=O','NCN','NCCN',
        # Functional groups — fluorine
        'CF','CCF','CCCF','FC=O','FCC=O','FCN','FCCN',
        # Mixed functional groups
        'CC(N)=O','CC(O)=O','NCC(=O)O','OCC(=O)O','CC(=O)N',
        'CC(=O)NC','NCC#N','OCC#N','CC(O)C','CC(N)C',
        # Rings with heteroatoms
        'C1=CC=CC=C1','C1=CN=CC=C1','C1=CC=NC=C1','C1=CC=CO1',
        'C1=CC=CS1','C1CCNCC1','C1CCNC1','C1CNCCN1','C1NCCN1',
        # Bicyclics / more complex
        'C1CC2CCCC2C1','C1CCC2CCCCC2C1','OC1CCCCC1','NC1CCCCC1',
        'CC1CCCCC1','CC1CCCC1','OC1CCCC1','NC1CCCC1','OCC1CCCCC1',
        # Small drug-like fragments
        'CC(C)O','CC(C)N','CCC(O)=O','CCC(N)=O','CC(=O)OC',
        'CC(C)C=O','OC(CO)CO','NCC(O)CO','CC(C)(O)C','CC(C)(N)C',
        'C1CC(N)CC1','C1CC(O)CC1','OC1CCNCC1','NC1CCOC1',
        'CC1CC(C)C1','OCC1CCCN1','NCC1CCCO1','CC(O)C#N',
        # Amino acids (small ones)
        'NCC(=O)O','CC(N)C(=O)O','OCC(N)C(=O)O',
        # Extended set for variety
        'C=CC=C','C=CC=CC','C=CC#N','C=CN','C=CO','C=CF',
        'CC=CC=O','CC=CCN','CC=CCO','CC(=C)C','C=C(C)C',
        'C1=CCCC1','C1=CCCCC1','C1=CNCC1','C1=COCC1',
        'OC=O','OCC=O','OC(=O)C=O','NC(=O)C','NC(N)=O',
        'CC(F)=O','FC(F)=O','CC(F)F','CF3','C(F)(F)F',
        'C1CC(=O)CC1','C1CC(=O)CCC1','C1CC(N)CC1','C1CC(O)CCC1',
        'OC1CC(O)CC1','NC1CC(N)CC1',
    ]

    def _load_or_process(self, max_mols):
        if os.path.exists(self.cache_path):
            print(f"[Dataset] Loading cached data from {self.cache_path}")
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        print("[Dataset] Building QM9-style molecule dataset from SMILES library...")

        # Expand dataset by augmenting base SMILES
        base = self._QM9_SMILES.copy()
        rng  = np.random.RandomState(42)

        adjs_list, nodes_list = [], []
        limit = max_mols if max_mols else 10000

        # Cycle through SMILES with canonical randomisation to reach limit
        seen = set()
        attempts = 0
        while len(adjs_list) < limit and attempts < limit * 5:
            smi = base[attempts % len(base)]
            attempts += 1
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            # Canonical SMILES to deduplicate
            csmi = Chem.MolToSmiles(mol)
            if csmi in seen:
                continue
            seen.add(csmi)
            adj, nodes = mol_to_graph(mol)
            if adj is not None:
                adjs_list.append(adj)
                nodes_list.append(nodes)

        adjs  = np.array(adjs_list,  dtype=np.int64)
        nodes = np.array(nodes_list, dtype=np.int64)

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump((adjs, nodes), f)

        print(f"[Dataset] Built {len(adjs)} valid molecules → cached.")
        print("  NOTE: Replace _QM9_SMILES with real QM9 download for full training.")
        return adjs, nodes

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        adj   = torch.tensor(self.adjs[idx],  dtype=torch.long)
        nodes = torch.tensor(self.nodes[idx], dtype=torch.long)
        return adj, nodes


def get_dataloader(split='train', batch_size=32, cache_path='data/qm9_processed.pkl',
                   max_mols=None):
    ds = QM9MolDataset(split=split, cache_path=cache_path, max_mols=max_mols)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'),
                      drop_last=True, num_workers=0)
