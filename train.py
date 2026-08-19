
import os, sys, warnings, random, copy, urllib.request, math, argparse, subprocess, json
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings("ignore")

# ============================================
# Auto-install
# ============================================
for pkg in ["rdkit", "torch", "torch_geometric", "scikit-learn", "pandas", "numpy", "tqdm", "matplotlib", "seaborn"]:
    mod = pkg.replace("-", "_").replace("scikit_learn", "sklearn")
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

try:
    from torch_geometric.nn.models import AttentiveFP
except ImportError:
    try:
        from torch_geometric.nn.models.attentive_fp import AttentiveFP
    except ImportError:
        AttentiveFP = None
        print("[WARN] AttentiveFP not available.")

# ============================================
# Reproducibility
# ============================================
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE} | [SEED] {SEED}")

# ============================================
# Dataset Configuration
# ============================================
DATASET_CONFIG = {
    "BBBP": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "path": "bbbp.csv",
        "smiles_col": "smiles",
        "label_col": "p_np"
    }
    #"BACE": {
     #  "path": "bace.csv",
      #  "smiles_col": "mol",
       # "label_col": "Class"
    #},
    #"HIV": {
     #   "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
      #  "path": "hiv.csv",
       # "smiles_col": "smiles",
        #"label_col": "HIV_active"
    #}
}

# ============================================
# Enhanced Molecular Descriptors (128-bit FP)
# ============================================
def compute_raw_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        desc = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol), Descriptors.MolMR(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.NumAliphaticRings(mol),
            Descriptors.NumHeteroatoms(mol),
        ]
        fp_gen = AllChem.GetMorganGenerator(radius=2, fpSize=128)
        fp = np.array(fp_gen.GetFingerprint(mol), dtype=np.float32)
        return np.concatenate([np.array(desc, dtype=np.float32), fp])
    except Exception:
        return None

# ============================================
# Stratified Scaffold Split
# ============================================
def stratified_scaffold_split(smiles_list, labels, frac_train=0.7, frac_val=0.1, seed=42):
    scaffolds = defaultdict(list)
    for i, (smi, label) in enumerate(zip(smiles_list, labels)):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaff is None:
                scaff = ''
        else:
            scaff = ''
        scaffolds[scaff].append((i, label))

    scaffold_info = []
    for scaff, items in scaffolds.items():
        idxs = [i for i, _ in items]
        acts = sum(1 for _, l in items if l == 1)
        ratio = acts / len(items) if len(items) > 0 else 0
        scaffold_info.append({
            'scaff': scaff, 'idxs': idxs, 'size': len(items),
            'active_ratio': ratio, 'n_active': acts
        })

    bins = {0: [], 1: [], 2: []}
    for info in scaffold_info:
        if info['active_ratio'] == 0:
            bins[0].append(info)
        elif info['active_ratio'] <= 0.5:
            bins[1].append(info)
        else:
            bins[2].append(info)

    rs = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    n_total = len(smiles_list)
    n_train = int(frac_train * n_total)
    n_val = int(frac_val * n_total)
    n_test = n_total - n_train - n_val

    for b in bins.values():
        if not b:
            continue
        b.sort(key=lambda x: x['size'], reverse=True)
        rs.shuffle(b)

        bin_total = sum(x['size'] for x in b)
        bin_train = int(frac_train * bin_total)
        bin_val = int(frac_val * bin_total)
        bin_test = bin_total - bin_train - bin_val

        t, v, te = [], [], []
        for info in b:
            t_ratio = len(t) / bin_train if bin_train > 0 else float('inf')
            v_ratio = len(v) / bin_val if bin_val > 0 else float('inf')
            te_ratio = len(te) / bin_test if bin_test > 0 else float('inf')

            if t_ratio <= v_ratio and t_ratio <= te_ratio and len(t) + info['size'] <= bin_train:
                t.extend(info['idxs'])
            elif v_ratio <= te_ratio and len(v) + info['size'] <= bin_val:
                v.extend(info['idxs'])
            else:
                te.extend(info['idxs'])

        train_idx.extend(t)
        val_idx.extend(v)
        test_idx.extend(te)

    print(f"  [SPLIT] Train: {len(train_idx)} ({len(train_idx)/n_total:.1%}) | "
          f"Val: {len(val_idx)} ({len(val_idx)/n_total:.1%}) | "
          f"Test: {len(test_idx)} ({len(test_idx)/n_total:.1%})")

    y_train = np.array([labels[i] for i in train_idx])
    y_test = np.array([labels[i] for i in test_idx])
    print(f"  [BALANCE] Train active: {y_train.mean():.3f} | Test active: {y_test.mean():.3f}")

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

# ============================================
# K-Fold Scaffold Cross-Validation
# ============================================
def kfold_scaffold_split(smiles_list, labels, n_folds=5, seed=42):
    scaffolds = defaultdict(list)
    for i, (smi, label) in enumerate(zip(smiles_list, labels)):
        mol = Chem.MolFromSmiles(smi)
        scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else ''
        scaffolds[scaff].append((i, label))

    scaffold_groups = []
    for scaff, items in scaffolds.items():
        idxs = [i for i, _ in items]
        acts = sum(1 for _, l in items if l == 1)
        ratio = acts / len(items) if items else 0
        scaffold_groups.append({
            'scaff': scaff, 'idxs': idxs, 'size': len(items),
            'active_ratio': ratio, 'n_active': acts
        })

    scaffold_groups.sort(key=lambda x: x['size'], reverse=True)
    rs = np.random.RandomState(seed)
    rs.shuffle(scaffold_groups)

    folds = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    fold_actives = [0] * n_folds

    for sg in scaffold_groups:
        target_fold = min(range(n_folds), key=lambda i: fold_sizes[i])
        folds[target_fold].extend(sg['idxs'])
        fold_sizes[target_fold] += sg['size']
        fold_actives[target_fold] += sg['n_active']

    splits = []
    for i in range(n_folds):
        test_idx = np.array(folds[i])
        train_idx = np.array([idx for j, fold in enumerate(folds) if j != i for idx in fold])
        rs2 = np.random.RandomState(seed + i)
        perm = rs2.permutation(len(train_idx))
        val_size = int(0.1 * len(train_idx))
        val_idx = train_idx[perm[:val_size]]
        train_idx = train_idx[perm[val_size:]]
        splits.append((train_idx, val_idx, test_idx))
        print(f"  [FOLD {i+1}] Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)} | "
              f"Test active: {np.mean([labels[j] for j in test_idx]):.3f}")
    return splits

# ============================================
# 2D Bond Features
# ============================================
BOND_FEAT_2D = 7
BOND_TYPES = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3
}

# ============================================
# Graph Construction (2D only)
# ============================================
def get_2d_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor([[
        a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(),
        a.GetTotalNumHs(), int(a.GetIsAromatic()), int(a.IsInRing()),
        int(a.GetHybridization()), a.GetMass() / 100.0,
        a.GetNumImplicitHs(), int(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED),
        a.GetTotalValence(), a.GetNumExplicitHs(), a.GetNumRadicalElectrons(),
    ] for a in mol.GetAtoms()], dtype=torch.float32)

    edges, attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = BOND_TYPES.get(bond.GetBondType(), 0)
        feat = [
            bt, int(bond.GetIsAromatic()), int(bond.IsInRing()),
            int(bond.GetIsConjugated()),
            1 if bond.GetStereo() != Chem.BondStereo.STEREONONE else 0,
            int(bond.IsInRingSize(5)) if hasattr(bond, 'IsInRingSize') else 0,
            int(bond.IsInRingSize(6)) if hasattr(bond, 'IsInRingSize') else 0,
        ]
        edges += [[i, j], [j, i]]
        attrs += [feat, feat]

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32) if attrs else torch.zeros((0, BOND_FEAT_2D))
    return x, edge_index, edge_attr, mol

def build_pyg_data(smiles, label, desc, use_desc=True):
    g2d = get_2d_graph(smiles)
    if g2d is None:
        return None
    x, edge_index, edge_attr, mol = g2d
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([float(label)], dtype=torch.float32))
    if use_desc:
        data.desc = torch.tensor(desc, dtype=torch.float32)
    data.edge_attr = edge_attr
    return data

# ============================================
# LayerScale & StochasticDepth
# ============================================
class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
    def forward(self, x):
        return self.gamma * x

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        mask = torch.empty(x.size(0), 1, device=x.device).bernoulli_(self.keep_prob)
        return x / self.keep_prob * mask

# ============================================
# Graph Transformer Layer
# ============================================
class GraphTransformerConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, num_heads=8, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.ls1 = LayerScale(hidden_dim)
        self.ls2 = LayerScale(hidden_dim)
        self.sd = StochasticDepth(drop_prob=0.05)
        self._attn_weights = None

    def forward(self, x, edge_index, edge_attr, return_attention=False):
        edge_emb = self.edge_proj(edge_attr)
        self._return_attention = return_attention
        self._attn_weights = None
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        if return_attention:
            return out, self._attn_weights, edge_index
        out = self.out_proj(out)
        x = self.norm1(x + self.sd(self.ls1(self.dropout(out))))
        x = self.norm2(x + self.sd(self.ls2(self.dropout(self.ffn(x)))))
        return x

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        q = self.q(x_i).view(-1, self.num_heads, self.head_dim)
        k = self.k(x_j).view(-1, self.num_heads, self.head_dim)
        v = self.v(x_j).view(-1, self.num_heads, self.head_dim)
        attn = (q * k).sum(dim=-1) * self.scale
        e = edge_attr.view(-1, self.num_heads, self.head_dim)
        attn = attn + (q * e).sum(dim=-1) * self.scale * 0.1
        attn = pyg_softmax(attn, index, ptr, size_i)
        attn = self.dropout(attn)
        if getattr(self, '_return_attention', False):
            self._attn_weights = attn.detach()
        out = v * attn.unsqueeze(-1)
        return out.view(-1, self.num_heads * self.head_dim)

# ============================================
# Enhanced GINEConv
# ============================================
class EnhancedGINEConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__(aggr='add')
        self.nn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.LayerNorm(2 * hidden_dim),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.ls = LayerScale(hidden_dim)
        self.sd = StochasticDepth(drop_prob=0.05)

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_encoder(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        return self.post_norm((1 + self.eps) * x + self.sd(self.ls(out)))

    def message(self, x_j, edge_attr):
        return self.nn(x_j) * torch.sigmoid(edge_attr)

# ============================================
# Atom Encoder
# ============================================
class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(120, hidden_dim), nn.Embedding(12, hidden_dim),
            nn.Embedding(11, hidden_dim), nn.Embedding(10, hidden_dim),
            nn.Embedding(2, hidden_dim), nn.Embedding(2, hidden_dim),
            nn.Embedding(10, hidden_dim), nn.Embedding(2, hidden_dim),
            nn.Embedding(12, hidden_dim), nn.Embedding(6, hidden_dim),
            nn.Embedding(5, hidden_dim),
        ])
        n_embs = len(self.embeddings)
        self.combine = nn.Sequential(
            nn.Linear(n_embs * hidden_dim, 2 * hidden_dim), nn.LayerNorm(2 * hidden_dim),
            nn.SiLU(), nn.Linear(2 * hidden_dim, hidden_dim)
        )

    def forward(self, x):
        feats = [
            x[:, 0].long().clamp(0, 119), x[:, 1].long().clamp(0, 11),
            (x[:, 2] + 5).long().clamp(0, 10), x[:, 3].long().clamp(0, 9),
            x[:, 4].long().clamp(0, 1), x[:, 5].long().clamp(0, 1),
            x[:, 6].long().clamp(0, 9), x[:, 9].long().clamp(0, 1),
            x[:, 10].long().clamp(0, 11), x[:, 11].long().clamp(0, 5),
            x[:, 12].long().clamp(0, 4),
        ]
        embs = [emb(f) for emb, f in zip(self.embeddings, feats)]
        return self.combine(torch.cat(embs, dim=-1))

# ============================================
# FIXED: Label-Agnostic Class Prototypes (NO LEAKAGE)
# ============================================
class ClassPrototypePool(nn.Module):
    """
    Computes class-specific prototype vectors from training data ONCE externally,
    then uses them as fixed priors during training/testing.
    NO label leakage — prototypes are precomputed, not from batch labels.
    """
    def __init__(self, hidden_dim, n_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.register_buffer('prototypes', torch.zeros(n_classes, hidden_dim))
        self.register_buffer('prototype_initialized', torch.tensor(0))
        self.prototype_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        )

    def compute_prototypes(self, loader, model, device):
        """Compute prototypes from full training set (call ONCE before training)."""
        model.eval()
        class_sums = [torch.zeros(self.hidden_dim, device=device) for _ in range(self.n_classes)]
        class_counts = [0 for _ in range(self.n_classes)]
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                x = model.atom_encoder(batch.x)
                x = F.silu(model.bn_input(x))
                for g_id in range(batch.max().item() + 1):
                    mask = batch.batch == g_id
                    label = int(batch.y[g_id].item() > 0.5)
                    graph_repr = global_mean_pool(x[mask].unsqueeze(0),
                        torch.zeros(mask.sum(), dtype=torch.long, device=device))
                    class_sums[label] += graph_repr.squeeze(0)
                    class_counts[label] += 1
        for c in range(self.n_classes):
            if class_counts[c] > 0:
                self.prototypes[c] = class_sums[c] / class_counts[c]
        self.prototype_initialized.fill_(1)
        print(f"[PROTOTYPES] Computed from {class_counts[0]} inactive + {class_counts[1]} active graphs")

    def forward(self, x, batch, use_prototype_weight=False):
        if not self.prototype_initialized or not use_prototype_weight:
            return global_add_pool(x, batch)
        n_graphs = batch.max().item() + 1
        pooled = global_add_pool(x, batch)
        sim_active = F.cosine_similarity(pooled, self.prototypes[1].unsqueeze(0), dim=1)
        weights = torch.sigmoid(sim_active).unsqueeze(-1)
        node_weights = weights[batch]
        return global_add_pool(x * node_weights, batch)

# ============================================
# FIXED Main Model: NO data.y in forward()
# ============================================
class GeoTransformerModel(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, num_tasks=1, dropout=0.2,
                 desc_dim=140, use_virtual_node=True, use_jk=True,
                 use_descriptors=True, use_gated_vn=False, use_prototype_pool=False):
        super().__init__()
        self.use_virtual_node = use_virtual_node
        self.use_jk = use_jk
        self.use_descriptors = use_descriptors
        self.use_gated_vn = use_gated_vn
        self.use_prototype_pool = use_prototype_pool
        self.num_layers = num_layers
        self.dropout = dropout

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bn_input = nn.LayerNorm(hidden_dim)
        edge_dim = BOND_FEAT_2D

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i in [2, 3]:
                layer = GraphTransformerConv(hidden_dim, edge_dim, num_heads=8, dropout=dropout)
            else:
                layer = EnhancedGINEConv(hidden_dim, edge_dim, dropout=dropout)
            self.gnn_layers.append(layer)

        if use_virtual_node:
            self.vn_mlp = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                    nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(num_layers)
            ])
            self.vn_init = nn.Linear(hidden_dim, hidden_dim)
            if use_gated_vn:
                self.vn_gates = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])

        # FIXED: Label-agnostic prototype pooling
        if use_prototype_pool:
            self.prototype_pool = ClassPrototypePool(hidden_dim)

        if use_jk:
            self.jk_proj = nn.Linear(hidden_dim * num_layers, hidden_dim)

        self.attn_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.pool_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        self.desc_proj = nn.Sequential(
            nn.Linear(desc_dim, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(dropout)
        ) if use_descriptors else None

        in_dim = hidden_dim + 32 if use_descriptors else hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )
        self.mc_dropout = nn.Dropout(dropout)

    def forward(self, data, edge_attr_override=None, collect_mechanistic=False, use_mc_dropout=False):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        n_graphs = batch.max().item() + 1
        if edge_attr_override is not None:
            edge_attr = edge_attr_override

        x = self.bn_input(self.atom_encoder(x))
        x = F.silu(x)
        vn = self.vn_init(global_add_pool(x, batch)) if self.use_virtual_node else None

        layer_outs = []
        mechanistic = {'attention': [], 'node_reps': []} if collect_mechanistic else None

        for i, conv in enumerate(self.gnn_layers):
            if self.use_virtual_node and vn is not None:
                if self.use_gated_vn:
                    gate = torch.sigmoid(self.vn_gates[i])
                    x = x + gate * vn[batch]
                else:
                    x = x + vn[batch]

            if isinstance(conv, GraphTransformerConv) and collect_mechanistic:
                x, attn, ei = conv(x, edge_index, edge_attr, return_attention=True)
                mechanistic['attention'].append((attn, ei))
            else:
                x = conv(x, edge_index, edge_attr)
            x = F.silu(x)

            if collect_mechanistic:
                mechanistic['node_reps'].append((x.detach().cpu(), batch.cpu()))

            # FIXED: No data.y access! Use prototype pool or standard pooling.
            if self.use_virtual_node and vn is not None and i < self.num_layers - 1:
                if self.use_prototype_pool and hasattr(self, 'prototype_pool'):
                    vn_update = self.prototype_pool(x, batch, use_prototype_weight=True)
                    vn = self.vn_mlp[i](vn_update) + vn
                else:
                    vn = self.vn_mlp[i](global_add_pool(x, batch)) + vn

            if use_mc_dropout:
                x = self.mc_dropout(x)
            layer_outs.append(x)

        if self.use_jk:
            x = self.jk_proj(torch.cat(layer_outs, dim=-1))
        else:
            x = layer_outs[-1]

        gate = pyg_softmax(self.attn_gate(x), batch)
        attn_pool = global_add_pool(x * gate, batch)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        g = self.pool_proj(torch.cat([attn_pool, mean_pool, max_pool], dim=-1))

        if self.use_descriptors and hasattr(data, 'desc') and self.desc_proj is not None:
            desc = self.desc_proj(data.desc.view(n_graphs, -1))
            g = torch.cat([g, desc], dim=-1)

        out = self.predictor(g)
        if collect_mechanistic:
            return out, mechanistic
        return out

# ============================================
# Baseline Models (GIN, D-MPNN, AttentiveFP)
# ============================================
class GINBaseline(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, num_tasks=1, dropout=0.2,
                 desc_dim=140, use_virtual_node=False, use_jk=False,
                 use_descriptors=True, **kwargs):
        super().__init__()
        self.use_descriptors = use_descriptors
        self.dropout = dropout
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bn_input = nn.LayerNorm(hidden_dim)
        edge_dim = BOND_FEAT_2D
        self.convs = nn.ModuleList([
            EnhancedGINEConv(hidden_dim, edge_dim, dropout=dropout) for _ in range(num_layers)
        ])
        self.batch_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.attn_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.pool_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.desc_proj = nn.Sequential(
            nn.Linear(desc_dim, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(dropout)
        ) if use_descriptors else None
        in_dim = hidden_dim + 32 if use_descriptors else hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, data, edge_attr_override=None, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        n_graphs = batch.max().item() + 1
        if edge_attr_override is not None:
            edge_attr = edge_attr_override
        x = self.bn_input(self.atom_encoder(x))
        x = F.silu(x)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.silu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        gate = pyg_softmax(self.attn_gate(x), batch)
        attn_pool = global_add_pool(x * gate, batch)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        g = self.pool_proj(torch.cat([attn_pool, mean_pool, max_pool], dim=-1))
        if self.use_descriptors and hasattr(data, 'desc') and self.desc_proj is not None:
            desc = self.desc_proj(data.desc.view(n_graphs, -1))
            g = torch.cat([g, desc], dim=-1)
        return self.predictor(g)

class DMPNNLayer(nn.Module):
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_msg = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, edge_hidden):
        src, dst = edge_index
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        node_aggr = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        node_aggr.index_add_(0, dst, edge_hidden)
        node_count = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        node_count.index_add_(0, dst, torch.ones(num_edges, dtype=torch.long, device=x.device))
        node_count = node_count.clamp(min=1).float().unsqueeze(-1)
        context = node_aggr[src] / node_count[src]
        msg_input = torch.cat([x[src], edge_attr, context], dim=-1)
        msg = self.W_msg(msg_input)
        new_hidden = self.gru(msg, edge_hidden)
        return new_hidden

class DMPNNBaseline(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, num_tasks=1, dropout=0.2,
                 desc_dim=140, use_descriptors=True, **kwargs):
        super().__init__()
        self.use_descriptors = use_descriptors
        self.dropout = dropout
        self.num_layers = num_layers
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bn_input = nn.LayerNorm(hidden_dim)
        edge_dim = BOND_FEAT_2D
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.mpnn_layers = nn.ModuleList([DMPNNLayer(hidden_dim, edge_dim) for _ in range(num_layers)])
        self.attn_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.pool_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        self.desc_proj = nn.Sequential(
            nn.Linear(desc_dim, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(dropout)
        ) if use_descriptors else None
        in_dim = hidden_dim + 32 if use_descriptors else hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, data, edge_attr_override=None, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        n_graphs = batch.max().item() + 1
        if edge_attr_override is not None:
            edge_attr = edge_attr_override
        x = self.bn_input(self.atom_encoder(x))
        x = F.silu(x)
        edge_h = self.edge_encoder(edge_attr)
        for layer in self.mpnn_layers:
            edge_h = layer(x, edge_index, edge_attr, edge_h)
            dst = edge_index[1]
            node_update = torch.zeros(x.size(0), edge_h.size(1), device=x.device)
            node_update.index_add_(0, dst, edge_h)
            x = x + node_update
            x = F.silu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        gate = pyg_softmax(self.attn_gate(x), batch)
        attn_pool = global_add_pool(x * gate, batch)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        g = self.pool_proj(torch.cat([attn_pool, mean_pool, max_pool], dim=-1))
        if self.use_descriptors and hasattr(data, 'desc') and self.desc_proj is not None:
            desc = self.desc_proj(data.desc.view(n_graphs, -1))
            g = torch.cat([g, desc], dim=-1)
        return self.predictor(g)

class AttentiveFPBaseline(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, num_tasks=1, dropout=0.2,
                 desc_dim=140, use_descriptors=True, **kwargs):
        super().__init__()
        if AttentiveFP is None:
            raise ImportError("AttentiveFP not available.")
        self.use_descriptors = use_descriptors
        self.dropout = dropout
        self.attentive_fp = AttentiveFP(
            in_channels=13, hidden_channels=hidden_dim, out_channels=hidden_dim,
            edge_dim=7, num_layers=num_layers, num_timesteps=2, dropout=dropout
        )
        self.desc_proj = nn.Sequential(
            nn.Linear(desc_dim, 32), nn.LayerNorm(32), nn.SiLU(), nn.Dropout(dropout)
        ) if use_descriptors else None
        in_dim = hidden_dim + 32 if use_descriptors else hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, data, **kwargs):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        n_graphs = batch.max().item() + 1
        g = self.attentive_fp(x, edge_index, edge_attr, batch)
        if self.use_descriptors and hasattr(data, 'desc') and self.desc_proj is not None:
            desc = self.desc_proj(data.desc.view(n_graphs, -1))
            g = torch.cat([g, desc], dim=-1)
        return self.predictor(g)

# ============================================
# Knowledge Distillation Loss
# ============================================
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5, feature_weight=0.1):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.feature_weight = feature_weight

    def forward(self, student_logits, teacher_logits, student_features, teacher_features, targets, criterion):
        hard_loss = criterion(student_logits, targets)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
        feat_loss = F.mse_loss(student_features, teacher_features) if student_features is not None else 0
        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss + self.feature_weight * feat_loss

# ============================================
# Focal Loss (alternative to BCE)
# ============================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()

# ============================================
# BCE Loss
# ============================================
class BCELossModule(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

# ============================================
# Virtual Screening Metrics
# ============================================
def enrichment_factor(y_true, y_score, alpha=0.01):
    n = len(y_true)
    n_alpha = max(1, int(alpha * n))
    order = np.argsort(-y_score)
    top_alpha = y_true[order[:n_alpha]]
    n_actives_top = top_alpha.sum()
    n_actives_total = y_true.sum()
    if n_actives_total == 0:
        return 0.0
    return (n_actives_top / n_alpha) / (n_actives_total / n)

def bedroc_score(y_true, y_score, alpha=80.5):
    n = len(y_true)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_actives = y_sorted.sum()
    if n_actives == 0 or n_actives == n:
        return 0.0
    ranks = np.arange(1, n + 1)
    expon = np.exp(-alpha * ranks / n)
    sum_expon = expon.sum()
    rie = (expon * y_sorted).sum() / (n_actives / n * sum_expon)
    bedroc = (rie * alpha / (np.exp(-alpha / n) - 1)) / (alpha / (np.exp(-alpha / n) - 1) - 1)
    return bedroc

def precision_at_k(y_true, y_score, k=100):
    order = np.argsort(-y_score)
    top_k = y_true[order[:k]]
    return top_k.mean() if len(top_k) > 0 else 0.0

def compute_screening_metrics(y_true, y_score):
    metrics = {}
    for alpha in [0.01, 0.05, 0.10]:
        metrics[f'EF@{int(alpha*100)}%'] = enrichment_factor(y_true, y_score, alpha)
    metrics['BEDROC'] = bedroc_score(y_true, y_score, alpha=80.5)
    for k in [10, 50, 100]:
        metrics[f'P@{k}'] = precision_at_k(y_true, y_score, k)
    return metrics

# ============================================
# Calibration Metrics
# ============================================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return ece / len(y_true)

# ============================================
# EMA & SWA
# ============================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    v.copy_(self.shadow[k])

    def restore(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.backup:
                    v.copy_(self.backup[k])

class SWA:
    def __init__(self, model, start_epoch=60):
        self.start_epoch = start_epoch
        self.swa_state = {}
        self.n_models = 0

    def update(self, model, epoch):
        if epoch < self.start_epoch:
            return
        self.n_models += 1
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.swa_state:
                self.swa_state[name] = param.data.clone()
            else:
                self.swa_state[name] += (param.data - self.swa_state[name]) / self.n_models

    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.swa_state:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.swa_state[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self.backup:
                param.data.copy_(self.backup[name])

# ============================================
# DropEdge
# ============================================
class DropEdge:
    def __init__(self, drop_prob=0.05):
        self.drop_prob = drop_prob

    def __call__(self, edge_index, edge_attr):
        if self.drop_prob == 0 or edge_index.size(1) == 0:
            return edge_index, edge_attr
        mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_prob
        return edge_index[:, mask], edge_attr[mask] if edge_attr is not None else None

# ============================================
# FIXED Trainer: Stratified sampling + Focal loss + Screening metrics + Calibration
# ============================================
class Trainer:
    def __init__(self, model, lr=1e-3, wd=1e-4, pos_weight=None,
                 use_ema=True, use_swa=True, device=DEVICE,
                 loss_type='bce', focal_alpha=0.25, focal_gamma=2.0,
                 use_stratified_sampler=True):
        self.model = model.to(device)
        self.device = device
        self.use_stratified_sampler = use_stratified_sampler

        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
            print(f"[LOSS] Using FocalLoss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            self.criterion = BCELossModule(pos_weight=pos_weight if pos_weight is not None else 1.0)
            print(f"[LOSS] Using BCE with pos_weight={pos_weight.item():.3f}" if pos_weight is not None else "[LOSS] Using BCE")

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
        self.ema = EMA(model, decay=0.999) if use_ema else None
        self.swa = SWA(model, start_epoch=60) if use_swa else None
        self.scheduler = None
        self.best_val_auc = -1
        self.best_state = None
        self.drop_edge = DropEdge(drop_prob=0.05)

    def setup_scheduler(self, total_steps, warmup_frac=0.15):
        warmup = int(total_steps * warmup_frac)
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + np.cos(np.pi * progress))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def get_stratified_sampler(self, dataset):
        labels = np.array([int(g.y.item()) for g in dataset])
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(labels), replacement=True
        )
        return sampler

    def train_epoch(self, loader, step):
        self.model.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            if self.drop_edge.drop_prob > 0 and batch.edge_attr is not None:
                batch.edge_index, batch.edge_attr = self.drop_edge(batch.edge_index, batch.edge_attr)
            self.optimizer.zero_grad()
            out = self.model(batch).squeeze()
            loss = self.criterion(out, batch.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            if self.ema:
                self.ema.update(self.model)
            total_loss += loss.item() * batch.num_graphs
            step += 1
        return total_loss / len(loader.dataset), step

    @torch.no_grad()
    def evaluate(self, loader, use_ema=True, use_swa=False, return_preds=False):
        self.model.eval()
        if use_swa and self.swa:
            self.swa.apply(self.model)
        elif use_ema and self.ema:
            self.ema.apply(self.model)
        preds, targets = [], []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch).squeeze()
            preds.append(torch.sigmoid(out).cpu())
            targets.append(batch.y.view(-1).cpu())
        if use_swa and self.swa:
            self.swa.restore(self.model)
        elif use_ema and self.ema:
            self.ema.restore(self.model)
        p = torch.cat(preds).numpy()
        y = torch.cat(targets).numpy()
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5
        acc = accuracy_score(y, (p > 0.5).astype(int))
        f1 = f1_score(y, (p > 0.5).astype(int))
        pr_auc = average_precision_score(y, p) if len(np.unique(y)) > 1 else 0.0
        screen_metrics = compute_screening_metrics(y, p)
        ece = expected_calibration_error(y, p)
        result = {'auc': auc, 'acc': acc, 'f1': f1, 'pr_auc': pr_auc, 'ece': ece, **screen_metrics}
        if return_preds:
            return result, p, y
        return result

    def fit(self, train_loader, val_loader, epochs=200, patience=40):
        self.setup_scheduler(epochs * len(train_loader))
        step = 0
        no_improve = 0
        for ep in range(1, epochs + 1):
            tr_loss, step = self.train_epoch(train_loader, step)
            if self.swa:
                self.swa.update(self.model, ep)
            val_metrics = self.evaluate(val_loader, use_ema=True, use_swa=False)
            val_auc = val_metrics['auc']
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if ep % 10 == 0 or ep == 1:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Ep {ep:3d} | Loss: {tr_loss:.4f} | Val ROC-AUC: {val_auc:.4f} | "
                      f"EF@1%: {val_metrics['EF@1%']:.2f} | BEDROC: {val_metrics['BEDROC']:.3f} | "
                      f"Best: {self.best_val_auc:.4f} | LR: {lr:.2e}")
            if no_improve >= patience:
                print(f"  Early stop at epoch {ep}")
                break
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        if self.ema:
            self.ema.apply(self.model)

    @torch.no_grad()
    def predict(self, loader, mc_iterations=0):
        self.model.eval()
        if mc_iterations > 0:
            self.model.train()
            all_preds = []
            for _ in range(mc_iterations):
                preds = []
                for batch in loader:
                    batch = batch.to(self.device)
                    out = self.model(batch, use_mc_dropout=True).squeeze()
                    preds.append(torch.sigmoid(out).cpu())
                all_preds.append(torch.cat(preds).numpy())
            mean_pred = np.mean(all_preds, axis=0)
            std_pred = np.std(all_preds, axis=0)
            return mean_pred, std_pred
        else:
            preds = []
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model(batch).squeeze()
                preds.append(torch.sigmoid(out).cpu())
            return torch.cat(preds).numpy(), None

# ============================================
# Platt Scaling / Isotonic Calibration
# ============================================
class Calibrator:
    def __init__(self, method='platt'):
        self.method = method
        self.calibrator = None

    def fit(self, y_true, y_prob):
        if self.method == 'platt':
            self.calibrator = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)
        elif self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)

    def predict(self, y_prob):
        if self.method == 'platt':
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic':
            return self.calibrator.predict(y_prob)
        return y_prob

# ============================================
# PyG Dataset wrapper
# ============================================
class PyGDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, i):
        return self.graphs[i]

# ============================================
# Mechanistic Analysis
# ============================================
def mechanistic_analysis(model_with_vn, model_without_vn, test_loader, device, output_dir="/mnt/agents/output"):
    os.makedirs(output_dir, exist_ok=True)
    model_with_vn.to(device)
    model_without_vn.to(device)
    model_with_vn.eval()
    model_without_vn.eval()

    attn_entropy_with, attn_entropy_without = [], []
    node_cos_with, node_cos_without = [], []

    print("[MECHANISTIC] Running analysis on test set...")
    for batch in tqdm(test_loader, desc="Mechanistic"):
        batch = batch.to(device)
        with torch.no_grad():
            _, mech = model_with_vn(batch, collect_mechanistic=True)
            for (attn, ei), (reps, batch_cpu) in zip(mech['attention'], mech['node_reps']):
                if attn is None:
                    continue
                for g_id in batch_cpu.unique():
                    mask = batch_cpu == g_id
                    nodes = reps[mask]
                    if nodes.size(0) < 2:
                        continue
                    nodes_norm = F.normalize(nodes, p=2, dim=1)
                    sim = torch.mm(nodes_norm, nodes_norm.t())
                    mask_diag = torch.ones_like(sim) - torch.eye(sim.size(0), device=sim.device)
                    mean_sim = (sim * mask_diag).sum() / mask_diag.sum()
                    node_cos_with.append(mean_sim.item())
                for h in range(attn.size(1)):
                    for dst in ei[1].unique():
                        dst_mask = ei[1] == dst
                        p = attn[dst_mask, h]
                        if p.sum() > 0:
                            p = p / p.sum()
                            entropy = -(p * torch.log(p + 1e-8)).sum()
                            attn_entropy_with.append(entropy.item())

        with torch.no_grad():
            _, mech = model_without_vn(batch, collect_mechanistic=True)
            for (attn, ei), (reps, batch_cpu) in zip(mech['attention'], mech['node_reps']):
                if attn is None:
                    continue
                for g_id in batch_cpu.unique():
                    mask = batch_cpu == g_id
                    nodes = reps[mask]
                    if nodes.size(0) < 2:
                        continue
                    nodes_norm = F.normalize(nodes, p=2, dim=1)
                    sim = torch.mm(nodes_norm, nodes_norm.t())
                    mask_diag = torch.ones_like(sim) - torch.eye(sim.size(0), device=sim.device)
                    mean_sim = (sim * mask_diag).sum() / mask_diag.sum()
                    node_cos_without.append(mean_sim.item())
                for h in range(attn.size(1)):
                    for dst in ei[1].unique():
                        dst_mask = ei[1] == dst
                        p = attn[dst_mask, h]
                        if p.sum() > 0:
                            p = p / p.sum()
                            entropy = -(p * torch.log(p + 1e-8)).sum()
                            attn_entropy_without.append(entropy.item())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_style("whitegrid")

    data_attn = {
        'Entropy': attn_entropy_with + attn_entropy_without,
        'Model': ['With VN'] * len(attn_entropy_with) + ['Without VN'] * len(attn_entropy_without)
    }
    sns.violinplot(data=data_attn, x='Model', y='Entropy', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Attention Entropy (Transformer Layers)', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Attention Entropy (nats)')
    mean_with = np.mean(attn_entropy_with) if attn_entropy_with else 0
    mean_without = np.mean(attn_entropy_without) if attn_entropy_without else 0
    axes[0].text(0.5, 0.95, f"Mean: {mean_with:.3f} vs {mean_without:.3f}",
                 transform=axes[0].transAxes, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    data_cos = {
        'Cosine Similarity': node_cos_with + node_cos_without,
        'Model': ['With VN'] * len(node_cos_with) + ['Without VN'] * len(node_cos_without)
    }
    sns.violinplot(data=data_cos, x='Model', y='Cosine Similarity', ax=axes[1], palette=['#e74c3c', '#2ecc71'])
    axes[1].set_title('Node Cosine Similarity (Transformer layers)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Mean Pairwise Cosine Similarity')
    mean_c_with = np.mean(node_cos_with) if node_cos_with else 0
    mean_c_without = np.mean(node_cos_without) if node_cos_without else 0
    axes[1].text(0.5, 0.95, f"Mean: {mean_c_with:.3f} vs {mean_c_without:.3f}",
                 transform=axes[1].transAxes, ha='center', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Mechanistic Analysis: Virtual Node Impact', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "mechanistic_analysis.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[MECHANISTIC] Figure saved to: {out_path}")
    return out_path

# ============================================
# Knowledge Distillation Training
# ============================================
def train_with_distillation(teacher_model, student_model, train_loader, val_loader,
                            epochs=200, patience=40, device=DEVICE, lr=1e-3, wd=1e-4, pos_weight=None,
                            temperature=4.0, alpha_kd=0.5):
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    student_model = student_model.to(device)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=wd)
    criterion = BCELossModule(pos_weight=pos_weight if pos_weight is not None else 1.0)
    kd_loss = DistillationLoss(temperature=temperature, alpha=alpha_kd)
    best_val_auc = -1
    best_state = None
    no_improve = 0

    for ep in range(1, epochs + 1):
        student_model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_out = teacher_model(batch).squeeze()
                teacher_feat = teacher_model.pool_proj(
                    torch.cat([
                        global_add_pool(teacher_model.atom_encoder(batch.x), batch.batch),
                        global_mean_pool(teacher_model.atom_encoder(batch.x), batch.batch),
                        global_max_pool(teacher_model.atom_encoder(batch.x), batch.batch)
                    ], dim=-1)
                )
            student_out = student_model(batch).squeeze()
            student_feat = student_model.pool_proj(
                torch.cat([
                    global_add_pool(student_model.atom_encoder(batch.x), batch.batch),
                    global_mean_pool(student_model.atom_encoder(batch.x), batch.batch),
                    global_max_pool(student_model.atom_encoder(batch.x), batch.batch)
                ], dim=-1)
            )
            loss = kd_loss(student_out, teacher_out, student_feat, teacher_feat, batch.y.view(-1), criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        student_model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = student_model(batch).squeeze()
                preds.append(torch.sigmoid(out).cpu())
                targets.append(batch.y.view(-1).cpu())
        p = torch.cat(preds).numpy()
        y = torch.cat(targets).numpy()
        val_auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(student_model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  [KD] Ep {ep:3d} | Loss: {total_loss/len(train_loader.dataset):.4f} | Val AUC: {val_auc:.4f} | Best: {best_val_auc:.4f}")
        if no_improve >= patience:
            print(f"  [KD] Early stop at epoch {ep}")
            break

    if best_state:
        student_model.load_state_dict(best_state)
    return student_model

# ============================================
# Experiment runner (FIXED)
# ============================================
def run_experiment(name, cfg, use_desc, use_pos_weight=True, model_cls=GeoTransformerModel,
                   df=None, desc_arr=None, train_idx=None, val_idx=None, test_idx=None,
                   return_model=False, loss_type='bce', use_stratified=True,
                   use_calibration=False, mc_iterations=0, teacher_model=None):
    print(f"\n{'=' * 70}\n  {name}")
    print(f"  Model: {model_cls.__name__} | Desc: {use_desc} | "
          f"VN: {cfg.get('vn', True)} | JK: {cfg.get('jk', True)} | "
          f"Loss: {loss_type} | Stratified: {use_stratified}")
    print(f"{'=' * 70}")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    def build_set(idx_set):
        out = []
        for i in tqdm(idx_set, desc="Build", leave=False):
            g = build_pyg_data(df['mol'].iloc[i], df['Class'].iloc[i], desc_arr[i], use_desc=use_desc)
            if g:
                out.append(g)
        return out

    graphs_train = build_set(train_idx)
    graphs_val = build_set(val_idx)
    graphs_test = build_set(test_idx)

    train_dataset = PyGDataset(graphs_train)
    if use_stratified:
        labels = np.array([int(g.y.item()) for g in graphs_train])
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(labels), replacement=True
        )
        train_loader = PyGDataLoader(train_dataset, batch_size=64, sampler=sampler,
                                     generator=torch.Generator().manual_seed(SEED))
    else:
        train_loader = PyGDataLoader(train_dataset, batch_size=64, shuffle=True,
                                     generator=torch.Generator().manual_seed(SEED))

    val_loader = PyGDataLoader(PyGDataset(graphs_val), batch_size=64, shuffle=False)
    test_loader = PyGDataLoader(PyGDataset(graphs_test), batch_size=64, shuffle=False)

    all_test_preds = []
    all_test_unc = []
    trained_model = None
    trained_trainer = None

    for m_idx in range(cfg['n_models']):
        model_seed = SEED + m_idx * 100
        torch.manual_seed(model_seed); np.random.seed(model_seed); random.seed(model_seed)

        model = model_cls(
            hidden_dim=256, num_layers=cfg['layers'], dropout=0.2, desc_dim=140,
            use_virtual_node=cfg.get('vn', True), use_jk=cfg.get('jk', True),
            use_descriptors=use_desc,
            use_gated_vn=cfg.get('gated_vn', False),
            use_prototype_pool=cfg.get('prototype_pool', False)
        )

        # FIXED: Compute prototypes BEFORE training if using prototype pool
        if cfg.get('prototype_pool', False) and hasattr(model, 'prototype_pool'):
            proto_loader = PyGDataLoader(PyGDataset(graphs_train), batch_size=64, shuffle=False)
            model.prototype_pool.compute_prototypes(proto_loader, model, DEVICE)

        lr = 8e-4
        pw = pos_weight if use_pos_weight else torch.tensor([1.0], dtype=torch.float32).to(DEVICE)

        # Knowledge distillation from teacher
        if teacher_model is not None and m_idx == 0:
            print(f"  [KD] Training with knowledge distillation from teacher...")
            model = train_with_distillation(teacher_model, model, train_loader, val_loader,
                                            epochs=cfg['epochs'], patience=40, device=DEVICE, lr=lr, pos_weight=pw)

        trainer = Trainer(
            model, lr=lr, wd=1e-4, pos_weight=pw,
            use_ema=True, use_swa=True, device=DEVICE,
            loss_type=loss_type, use_stratified_sampler=use_stratified
        )
        trainer.fit(train_loader, val_loader, epochs=cfg['epochs'], patience=40)

        if mc_iterations > 0:
            preds, unc = trainer.predict(test_loader, mc_iterations=mc_iterations)
            all_test_preds.append(preds)
            all_test_unc.append(unc)
        else:
            preds = trainer.predict(test_loader)
            all_test_preds.append(preds)

        trained_model = model
        trained_trainer = trainer

    ens_pred = np.mean(all_test_preds, axis=0)
    ens_std = np.mean(all_test_unc, axis=0) if all_test_unc else None
    test_y = np.array([int(g.y.item()) for g in graphs_test])

    auc = roc_auc_score(test_y, ens_pred)
    acc = accuracy_score(test_y, (ens_pred > 0.5).astype(int))
    f1 = f1_score(test_y, (ens_pred > 0.5).astype(int))
    pr_auc = average_precision_score(test_y, ens_pred)
    screen_metrics = compute_screening_metrics(test_y, ens_pred)
    ece = expected_calibration_error(test_y, ens_pred)

    print(f"\n[RESULT] {name} | ROC-AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
    print(f"  EF@1%: {screen_metrics['EF@1%']:.2f} | EF@5%: {screen_metrics['EF@5%']:.2f} | "
          f"EF@10%: {screen_metrics['EF@10%']:.2f} | BEDROC: {screen_metrics['BEDROC']:.3f}")
    print(f"  ECE: {ece:.4f}")

    result = {'name': name, 'auc': auc, 'pr_auc': pr_auc, 'acc': acc, 'f1': f1, 'ece': ece, **screen_metrics}

    if use_calibration:
        calibrator = Calibrator(method='platt')
        val_preds, val_targets = [], []
        for batch in val_loader:
            batch = batch.to(DEVICE)
            with torch.no_grad():
                out = trained_model(batch).squeeze()
                val_preds.append(torch.sigmoid(out).cpu())
                val_targets.append(batch.y.view(-1).cpu())
        val_p = torch.cat(val_preds).numpy()
        val_y = torch.cat(val_targets).numpy()
        calibrator.fit(val_y, val_p)
        cal_pred = calibrator.predict(ens_pred)
        cal_auc = roc_auc_score(test_y, cal_pred)
        cal_ece = expected_calibration_error(test_y, cal_pred)
        result['cal_auc'] = cal_auc
        result['cal_ece'] = cal_ece
        print(f"  [CALIBRATION] Platt scaling: AUC={cal_auc:.4f}, ECE={cal_ece:.4f}")

    if return_model:
        return result, trained_model, trained_trainer, test_loader, graphs_test
    return result

# ============================================
# Main
# ============================================
def main():
    import sys
    clean_argv = [sys.argv[0]]
    skip = False
    for arg in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if arg == '-f':
            skip = True
            continue
        if 'jupyter/runtime/kernel' in arg and arg.endswith('.json'):
            continue
        if arg.startswith(('--IPKernelApp', '--InteractiveShell', '--matplotlib', '--profile')):
            continue
        clean_argv.append(arg)
    sys.argv = clean_argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BBBP', choices=['BBBP'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_seeds', type=int, default=1, help='Number of seeds for multi-seed validation')
    parser.add_argument('--k_fold', type=int, default=0, help='K-fold scaffold CV (0=single split)')
    parser.add_argument('--full_ablation', action='store_true')
    parser.add_argument('--gated_vn', action='store_true')
    parser.add_argument('--prototype_pool', action='store_true', help='Use label-agnostic prototype pooling')
    parser.add_argument('--run_baselines', action='store_true')
    parser.add_argument('--mechanistic', action='store_true')
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument('--use_stratified', action='store_true', default=True, help='Stratified oversampling')
    parser.add_argument('--use_calibration', action='store_true', help='Apply Platt scaling calibration')
    parser.add_argument('--mc_iterations', type=int, default=0, help='MC-dropout iterations for uncertainty')
    parser.add_argument('--knowledge_distillation', action='store_true', help='Distill to smaller student model')

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] Ignored unknown arguments: {unknown}")
    if len(clean_argv) == 1 and not any(a.startswith('--') for a in clean_argv):
        print("[INFO] No CLI flags detected. Enabling --full_ablation --run_baselines --mechanistic")
        args.full_ablation = True
        args.run_baselines = True
        args.mechanistic = True
        args.prototype_pool = True
        args.use_calibration = True
        args.mc_iterations = 10

    global SEED, pos_weight
    SEED = args.seed
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # Load dataset
    config = DATASET_CONFIG[args.dataset]
    DATA_URL = config['url']
    DATA_PATH = config['path']
    SMILES_COL = config['smiles_col']
    LABEL_COL = config['label_col']

    if not os.path.exists(DATA_PATH):
        print(f"Downloading {args.dataset} dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[SMILES_COL, LABEL_COL])
    df = df.rename(columns={SMILES_COL: "mol", LABEL_COL: "Class"})

    print(f"[DATA] {df.shape} | Class: {df['Class'].value_counts().to_dict()}")

    # Compute descriptors
    raw_descs = []
    for smi in tqdm(df['mol'], desc="Descriptors"):
        d = compute_raw_descriptors(smi)
        raw_descs.append(d if d is not None else np.zeros(140, dtype=np.float32))
    raw_descs = np.stack(raw_descs)

    # K-fold or single split
    if args.k_fold > 1:
        splits = kfold_scaffold_split(df['mol'].tolist(), df['Class'].values, n_folds=args.k_fold, seed=SEED)
    else:
        TRAIN_IDX, VAL_IDX, TEST_IDX = stratified_scaffold_split(
            df['mol'].tolist(), df['Class'].values, seed=SEED
        )
        splits = [(TRAIN_IDX, VAL_IDX, TEST_IDX)]

    all_results = []
    all_models_for_mech = {}

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        print(f"\n{'='*70}")
        print(f"  FOLD {fold_idx + 1}/{len(splits)}")
        print(f"{'='*70}")

        # Normalize per split
        train_mean = raw_descs[train_idx].mean(0)
        train_std = raw_descs[train_idx].std(0) + 1e-8
        desc_arr = (raw_descs - train_mean) / train_std

        # Pos weight
        train_labels = df['Class'].iloc[train_idx].values
        n_inactive = (train_labels == 0).sum()
        n_active = (train_labels == 1).sum()
        pos_weight = n_inactive / max(1, n_active)
        pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
        print(f"[TRAIN CLASS] Inactive: {n_inactive} | Active: {n_active} | pos_weight: {pos_weight.item():.3f}")

        n_models = 5 if len(train_idx) < 3000 else 3
        print(f"[ENSEMBLE] Using {n_models} models")

        BASE = {'vn': True, 'jk': True, 'n_models': n_models, 'epochs': 200, 'layers': 5}

        # Teacher model for knowledge distillation
        teacher_model = None
        if args.knowledge_distillation:
            print("\n[TEACHER] Training large teacher model...")
            teacher_cfg = BASE.copy()
            teacher_cfg['layers'] = 7
            teacher_cfg['n_models'] = 1
            teacher_result, teacher_model, _, _, _ = run_experiment(
                "Teacher Model (7-layer)", teacher_cfg,
                use_desc=True, use_pos_weight=True,
                model_cls=GeoTransformerModel,
                df=df, desc_arr=desc_arr,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                return_model=True, loss_type=args.loss_type,
                use_stratified=args.use_stratified
            )
            print(f"[TEACHER] Val AUC: {teacher_result['auc']:.4f}")

        if args.full_ablation:
            EXPERIMENTS = [
                {'name': f'Fold{fold_idx+1}: 1. Full 2D + BCE (Baseline)', 'vn': True, 'jk': True, 'desc': True, 'pos_weight': True, 'gated_vn': False, 'prototype_pool': False},
                {'name': f'Fold{fold_idx+1}: 2. - Descriptors (BCE)', 'vn': True, 'jk': True, 'desc': False, 'pos_weight': True, 'gated_vn': False, 'prototype_pool': False},
                {'name': f'Fold{fold_idx+1}: 3. - Virtual Node (BCE)', 'vn': False, 'jk': True, 'desc': True, 'pos_weight': True, 'gated_vn': False, 'prototype_pool': False},
                {'name': f'Fold{fold_idx+1}: 4. - Jumping Knowledge (BCE)', 'vn': True, 'jk': False, 'desc': True, 'pos_weight': True, 'gated_vn': False, 'prototype_pool': False},
                {'name': f'Fold{fold_idx+1}: 5. - BCE only (no pos_weight)', 'vn': True, 'jk': True, 'desc': True, 'pos_weight': False, 'gated_vn': False, 'prototype_pool': False},
                {'name': f'Fold{fold_idx+1}: 6. + Gated VN (BCE)', 'vn': True, 'jk': True, 'desc': True, 'pos_weight': True, 'gated_vn': True, 'prototype_pool': False},
                {'name': f'Fold{fold_idx+1}: 7. + Prototype Pool (BCE)', 'vn': True, 'jk': True, 'desc': True, 'pos_weight': True, 'gated_vn': False, 'prototype_pool': True},
                {'name': f'Fold{fold_idx+1}: 8. + Gated VN + Prototype Pool (BCE)', 'vn': True, 'jk': True, 'desc': True, 'pos_weight': True, 'gated_vn': True, 'prototype_pool': True},
            ]

            results = []
            models_for_mech = {}

            for exp in EXPERIMENTS:
                cfg = BASE.copy()
                cfg['vn'] = exp['vn']
                cfg['jk'] = exp['jk']
                cfg['gated_vn'] = exp['gated_vn']
                cfg['prototype_pool'] = exp['prototype_pool']

                return_model = args.mechanistic and (exp['name'].endswith('1. Full 2D + BCE (Baseline)') or exp['name'].endswith('3. - Virtual Node (BCE)'))

                try:
                    r = run_experiment(
                        exp['name'], cfg,
                        use_desc=exp['desc'],
                        use_pos_weight=exp['pos_weight'],
                        model_cls=GeoTransformerModel,
                        df=df, desc_arr=desc_arr,
                        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                        return_model=return_model,
                        loss_type=args.loss_type,
                        use_stratified=args.use_stratified,
                        use_calibration=args.use_calibration,
                        mc_iterations=args.mc_iterations,
                        teacher_model=teacher_model
                    )
                    if return_model:
                        models_for_mech[exp['name']] = r
                        r[1].cpu()
                        results.append(r[0])
                    else:
                        results.append(r)
                except Exception as e:
                    print(f"\n[ERROR] {exp['name']} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        'name': exp['name'], 'auc': 0.0, 'pr_auc': 0.0, 'acc': 0.0, 'f1': 0.0,
                        'EF@1%': 0.0, 'EF@5%': 0.0, 'EF@10%': 0.0, 'BEDROC': 0.0,
                        'P@10': 0.0, 'P@50': 0.0, 'P@100': 0.0, 'ece': 0.0
                    })

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # GIN Baseline
            print("\n" + "=" * 70)
            print("  EXTERNAL BASELINE: GIN (5-layer)")
            print("=" * 70)
            gin_result = run_experiment(
                f"Fold{fold_idx+1}: GIN Baseline", BASE,
                use_desc=True, use_pos_weight=True,
                model_cls=GINBaseline,
                df=df, desc_arr=desc_arr,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                loss_type=args.loss_type, use_stratified=args.use_stratified
            )

            # D-MPNN Baseline
            if args.run_baselines:
                print("\n" + "=" * 70)
                print("  EXTERNAL BASELINE: D-MPNN")
                print("=" * 70)
                dmpnn_result = run_experiment(
                    f"Fold{fold_idx+1}: D-MPNN Baseline", BASE,
                    use_desc=True, use_pos_weight=True,
                    model_cls=DMPNNBaseline,
                    df=df, desc_arr=desc_arr,
                    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                    loss_type=args.loss_type, use_stratified=args.use_stratified
                )

                if AttentiveFP is not None:
                    print("\n" + "=" * 70)
                    print("  EXTERNAL BASELINE: AttentiveFP")
                    print("=" * 70)
                    afp_result = run_experiment(
                        f"Fold{fold_idx+1}: AttentiveFP Baseline", BASE,
                        use_desc=True, use_pos_weight=True,
                        model_cls=AttentiveFPBaseline,
                        df=df, desc_arr=desc_arr,
                        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                        loss_type=args.loss_type, use_stratified=args.use_stratified
                    )
                else:
                    afp_result = None

            # Summary
            print("=" * 70)
            print(f"FINAL ABLATION RESULTS - FOLD {fold_idx+1}")
            print("=" * 70)
            for r in results:
                print(f"  {r['name']:55s}  ROC-AUC: {r['auc']:.4f}  PR-AUC: {r['pr_auc']:.4f}  "
                      f"EF@1%: {r['EF@1%']:.2f}  BEDROC: {r['BEDROC']:.3f}  ECE: {r['ece']:.4f}")
            print("-" * 70)
            print(f"  {gin_result['name']:55s}  ROC-AUC: {gin_result['auc']:.4f}  PR-AUC: {gin_result['pr_auc']:.4f}")
            if args.run_baselines:
                print(f"  {dmpnn_result['name']:55s}  ROC-AUC: {dmpnn_result['auc']:.4f}  PR-AUC: {dmpnn_result['pr_auc']:.4f}")
                if afp_result:
                    print(f"  {afp_result['name']:55s}  ROC-AUC: {afp_result['auc']:.4f}  PR-AUC: {afp_result['pr_auc']:.4f}")
            print("=" * 70)

            all_results.extend(results)
            all_results.append(gin_result)
            if args.run_baselines:
                all_results.append(dmpnn_result)
                if afp_result:
                    all_results.append(afp_result)

            # Mechanistic analysis
            if args.mechanistic:
                baseline_key = [k for k in models_for_mech.keys() if '1. Full 2D + BCE' in k]
                novn_key = [k for k in models_for_mech.keys() if '3. - Virtual Node' in k]
                if baseline_key and novn_key:
                    _, model_full, _, test_loader, _ = models_for_mech[baseline_key[0]]
                    _, model_novn, _, _, _ = models_for_mech[novn_key[0]]
                    mechanistic_analysis(model_full, model_novn, test_loader, DEVICE)

        else:
            # Single run
            cfg = BASE.copy()
            if args.gated_vn:
                cfg['gated_vn'] = True
            if args.prototype_pool:
                cfg['prototype_pool'] = True

            r1 = run_experiment(
                f"Fold{fold_idx+1}: GeoTransformer-GINE (Full 2D)", cfg,
                use_desc=True, use_pos_weight=True,
                model_cls=GeoTransformerModel,
                df=df, desc_arr=desc_arr,
                train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                loss_type=args.loss_type, use_stratified=args.use_stratified,
                use_calibration=args.use_calibration,
                mc_iterations=args.mc_iterations
            )

            if args.run_baselines:
                r2 = run_experiment(
                    f"Fold{fold_idx+1}: GIN Baseline", cfg,
                    use_desc=True, use_pos_weight=True,
                    model_cls=GINBaseline,
                    df=df, desc_arr=desc_arr,
                    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                    loss_type=args.loss_type, use_stratified=args.use_stratified
                )
                r3 = run_experiment(
                    f"Fold{fold_idx+1}: D-MPNN Baseline", cfg,
                    use_desc=True, use_pos_weight=True,
                    model_cls=DMPNNBaseline,
                    df=df, desc_arr=desc_arr,
                    train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                    loss_type=args.loss_type, use_stratified=args.use_stratified
                )
                if AttentiveFP is not None:
                    r4 = run_experiment(
                        f"Fold{fold_idx+1}: AttentiveFP Baseline", cfg,
                        use_desc=True, use_pos_weight=True,
                        model_cls=AttentiveFPBaseline,
                        df=df, desc_arr=desc_arr,
                        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                        loss_type=args.loss_type, use_stratified=args.use_stratified
                    )
                print(f"\nDelta (Ours - GIN): ROC-AUC {r1['auc'] - r2['auc']:+.4f} | PR-AUC {r1['pr_auc'] - r2['pr_auc']:+.4f}")
                print(f"Delta (Ours - D-MPNN): ROC-AUC {r1['auc'] - r3['auc']:+.4f} | PR-AUC {r1['pr_auc'] - r3['pr_auc']:+.4f}")
                if AttentiveFP is not None:
                    print(f"Delta (Ours - AttentiveFP): ROC-AUC {r1['auc'] - r4['auc']:+.4f} | PR-AUC {r1['pr_auc'] - r4['pr_auc']:+.4f}")

            all_results.append(r1)

    # Cross-fold summary
    if args.k_fold > 1:
        print("\n" + "=" * 70)
        print("  CROSS-FOLD SUMMARY")
        print("=" * 70)
        exp_groups = defaultdict(list)
        for r in all_results:
            base_name = ': '.join(r['name'].split(': ')[1:]) if ': ' in r['name'] else r['name']
            exp_groups[base_name].append(r)

        for exp_name, results in exp_groups.items():
            aucs = [r['auc'] for r in results]
            pr_aucs = [r['pr_auc'] for r in results]
            ef1s = [r['EF@1%'] for r in results]
            bedrocs = [r['BEDROC'] for r in results]
            print(f"  {exp_name:50s}  AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
                  f"PR-AUC: {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f}  "
                  f"EF@1%: {np.mean(ef1s):.2f} ± {np.std(ef1s):.2f}  "
                  f"BEDROC: {np.mean(bedrocs):.3f} ± {np.std(bedrocs):.3f}")
        print("=" * 70)

    # Save results
    output_dir = "/mnt/agents/output"
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, f"results_{args.dataset}_seed{SEED}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[SAVE] Results saved to: {results_path}")

    config_path = os.path.join(output_dir, f"config_{args.dataset}_seed{SEED}.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"[SAVE] Config saved to: {config_path}")


if __name__ == "__main__":
    main()
      
