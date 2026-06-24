"""
GeoTransformer-GINE : 2D + GIN Baseline
Scaffold-split molecular property prediction (BACE/BBBP/HIV).
GIN baseline added for fair comparison.
"""

import os, sys, warnings, random, copy, urllib.request, math, argparse
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm.auto import tqdm

# ============================================
# Reproducibility
# ============================================
SEED = 42
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
    "BACE": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "path": "bace.csv",
        "smiles_col": "mol",
        "label_col": "Class"
    },
    "BBBP": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "path": "bbbp.csv",
        "smiles_col": "smiles",
        "label_col": "p_np"
    },
    "HIV": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "path": "hiv.csv",
        "smiles_col": "smiles",
        "label_col": "HIV_active"
    }
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
# Scaffold Split
# ============================================
def scaffold_split_indices(smiles_list, frac_train=0.7, frac_val=0.1, seed=42):
    scaffolds = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaff is None:
                scaff = ''
        else:
            scaff = ''
        scaffolds[scaff].append(i)

    scaffold_sets = list(scaffolds.values())
    rs = np.random.RandomState(seed)
    rs.shuffle(scaffold_sets)

    n_total = len(smiles_list)
    n_train = int(frac_train * n_total)
    n_val = int(frac_val * n_total)
    n_test = n_total - n_train - n_val

    train_idx, val_idx, test_idx = [], [], []

    for s_set in scaffold_sets:
        train_ratio = len(train_idx) / n_train if n_train > 0 else float('inf')
        val_ratio = len(val_idx) / n_val if n_val > 0 else float('inf')
        test_ratio = len(test_idx) / n_test if n_test > 0 else float('inf')

        if train_ratio <= val_ratio and train_ratio <= test_ratio and len(train_idx) + len(s_set) <= n_train:
            train_idx.extend(s_set)
        elif val_ratio <= test_ratio and len(val_idx) + len(s_set) <= n_val:
            val_idx.extend(s_set)
        else:
            test_idx.extend(s_set)

    print(f"  [SPLIT] Train: {len(train_idx)} ({len(train_idx)/n_total:.1%}) | "
          f"Val: {len(val_idx)} ({len(val_idx)/n_total:.1%}) | "
          f"Test: {len(test_idx)} ({len(test_idx)/n_total:.1%})")

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

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

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([float(label)], dtype=torch.float32)
    )
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
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.ls1 = LayerScale(hidden_dim)
        self.ls2 = LayerScale(hidden_dim)
        self.sd = StochasticDepth(drop_prob=0.05)

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_proj(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
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

        out = v * attn.unsqueeze(-1)
        return out.view(-1, self.num_heads * self.head_dim)

# ============================================
# Enhanced GINEConv
# ============================================
class EnhancedGINEConv(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, dropout=0.1):
        super().__init__(aggr='add')
        self.nn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
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
            nn.Embedding(120, hidden_dim),
            nn.Embedding(12, hidden_dim),
            nn.Embedding(11, hidden_dim),
            nn.Embedding(10, hidden_dim),
            nn.Embedding(2, hidden_dim),
            nn.Embedding(2, hidden_dim),
            nn.Embedding(10, hidden_dim),
            nn.Embedding(2, hidden_dim),
            nn.Embedding(12, hidden_dim),
            nn.Embedding(6, hidden_dim),
            nn.Embedding(5, hidden_dim),
        ])
        n_embs = len(self.embeddings)
        self.combine = nn.Sequential(
            nn.Linear(n_embs * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, hidden_dim)
        )

    def forward(self, x):
        feats = [
            x[:, 0].long().clamp(0, 119),
            x[:, 1].long().clamp(0, 11),
            (x[:, 2] + 5).long().clamp(0, 10),
            x[:, 3].long().clamp(0, 9),
            x[:, 4].long().clamp(0, 1),
            x[:, 5].long().clamp(0, 1),
            x[:, 6].long().clamp(0, 9),
            x[:, 9].long().clamp(0, 1),
            x[:, 10].long().clamp(0, 11),
            x[:, 11].long().clamp(0, 5),
            x[:, 12].long().clamp(0, 4),
        ]
        embs = [emb(f) for emb, f in zip(self.embeddings, feats)]
        return self.combine(torch.cat(embs, dim=-1))

# ============================================
# Main Hybrid Model (2D only)
# ============================================
class GeoTransformerModel(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, num_tasks=1, dropout=0.2,
                 desc_dim=140, use_virtual_node=True, use_jk=True,
                 use_descriptors=True):
        super().__init__()
        self.use_virtual_node = use_virtual_node
        self.use_jk = use_jk
        self.use_descriptors = use_descriptors
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
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim)
                ) for _ in range(num_layers)
            ])
            self.vn_init = nn.Linear(hidden_dim, hidden_dim)

        if use_jk:
            self.jk_proj = nn.Linear(hidden_dim * num_layers, hidden_dim)

        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.pool_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        self.desc_proj = nn.Sequential(
            nn.Linear(desc_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout)
        ) if use_descriptors else None

        in_dim = hidden_dim + 32 if use_descriptors else hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, data, edge_attr_override=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        n_graphs = batch.max().item() + 1
        if edge_attr_override is not None:
            edge_attr = edge_attr_override

        x = self.bn_input(self.atom_encoder(x))
        x = F.silu(x)

        vn = self.vn_init(global_add_pool(x, batch)) if self.use_virtual_node else None

        layer_outs = []
        for i, conv in enumerate(self.gnn_layers):
            if self.use_virtual_node and vn is not None:
                x = x + vn[batch]
            x = conv(x, edge_index, edge_attr)
            x = F.silu(x)
            if self.use_virtual_node and vn is not None and i < self.num_layers - 1:
                vn = self.vn_mlp[i](global_add_pool(x, batch)) + vn
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
        return self.predictor(g)

# ============================================
# External Baseline: GIN (5-layer, standard)
# ============================================
class GINBaseline(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, num_tasks=1, dropout=0.2,
                 desc_dim=140, use_virtual_node=False, use_jk=False,
                 use_descriptors=True):
        super().__init__()
        self.use_descriptors = use_descriptors
        self.dropout = dropout

        self.atom_encoder = AtomEncoder(hidden_dim)
        self.bn_input = nn.LayerNorm(hidden_dim)

        edge_dim = BOND_FEAT_2D

        self.convs = nn.ModuleList([
            EnhancedGINEConv(hidden_dim, edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.batch_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.pool_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        self.desc_proj = nn.Sequential(
            nn.Linear(desc_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout)
        ) if use_descriptors else None

        in_dim = hidden_dim + 32 if use_descriptors else hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, data, edge_attr_override=None):
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

# ============================================
# BCE Loss
# ============================================
class BCELossModule(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
       self.pos_weight = pos_weight

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

# ============================================
# EMA
# ============================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items() if k in self.shadow
        }
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    v.copy_(self.shadow[k])
                  def restore(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.backup:
                    v.copy_(self.backup[k])

# ============================================
# SWA
# ============================================
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
# Trainer
# ============================================
class Trainer:
    def __init__(self, model, lr=1e-3, wd=1e-4, pos_weight=None,
                 use_ema=True, use_swa=True, device=DEVICE):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999)
        )
        self.criterion = BCELossModule(pos_weight=pos_weight if pos_weight is not None else 1.0)
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
    def evaluate(self, loader, use_ema=True, use_swa=False):
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
      return auc, acc, f1

    def fit(self, train_loader, val_loader, epochs=200, patience=40):
        self.setup_scheduler(epochs * len(train_loader))
        step = 0
        no_improve = 0
        for ep in range(1, epochs + 1):
            tr_loss, step = self.train_epoch(train_loader, step)
            if self.swa:
                self.swa.update(self.model, ep)
            val_auc, val_acc, val_f1 = self.evaluate(val_loader, use_ema=True, use_swa=False)
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if ep % 10 == 0 or ep == 1:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"  Ep {ep:3d} | Loss: {tr_loss:.4f} | Val AUC: {val_auc:.4f} | "
                      f"Best: {self.best_val_auc:.4f} | LR: {lr:.2e}")
            if no_improve >= patience:
                print(f"  Early stop at epoch {ep}")
                break
        if self.best_state:
            self.model.load_state_dict(self.best_state)
        if self.ema:
            self.ema.apply(self.model)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        preds = []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch).squeeze()
            preds.append(torch.sigmoid(out).cpu())
        return torch.cat(preds).numpy()
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
# Experiment runner
# ============================================
def run_experiment(name, cfg, use_desc, use_pos_weight=True, model_cls=GeoTransformerModel,
                   df=None, desc_arr=None, train_idx=None, val_idx=None, test_idx=None):
    print(f"\n{'=' * 70}\n  {name}")
    print(f"  Model: {model_cls.__name__} | Desc: {use_desc} | "
          f"VN: {cfg.get('vn', True)} | JK: {cfg.get('jk', True)} | pos_weight: {use_pos_weight}")
    print(f"{'=' * 70}")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    def build_set(idx_set):
        out = []
        for i in tqdm(idx_set, desc="Build", leave=False):
            g = build_pyg_data(
                df['mol'].iloc[i], df['Class'].iloc[i], desc_arr[i],
                use_desc=use_desc
            )
            if g:
                out.append(g)
        return out

    graphs_train = build_set(train_idx)
    graphs_val = build_set(val_idx)
    graphs_test = build_set(test_idx)

    train_loader = PyGDataLoader(
        PyGDataset(graphs_train), batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(SEED)
    )
    val_loader = PyGDataLoader(PyGDataset(graphs_val), batch_size=64, shuffle=False)
    test_loader = PyGDataLoader(PyGDataset(graphs_test), batch_size=64, shuffle=False)
all_test_preds = []
    for m_idx in range(cfg['n_models']):
        model_seed = SEED + m_idx * 100
        torch.manual_seed(model_seed); np.random.seed(model_seed); random.seed(model_seed)

        model = model_cls(
            hidden_dim=256, num_layers=cfg['layers'], dropout=0.2, desc_dim=140,
            use_virtual_node=cfg.get('vn', True), use_jk=cfg.get('jk', True),
            use_descriptors=use_desc
        )

        lr = 8e-4
        pw = pos_weight if use_pos_weight else torch.tensor([1.0], dtype=torch.float32).to(DEVICE)
        trainer = Trainer(
            model, lr=lr, wd=1e-4, pos_weight=pw,
            use_ema=True, use_swa=True, device=DEVICE
        )
        trainer.fit(train_loader, val_loader, epochs=cfg['epochs'], patience=40)
        preds = trainer.predict(test_loader)
        all_test_preds.append(preds)

    ens_pred = np.mean(all_test_preds, axis=0)
    test_y = np.array([int(g.y.item()) for g in graphs_test])
    auc = roc_auc_score(test_y, ens_pred)
    acc = accuracy_score(test_y, (ens_pred > 0.5).astype(int))
    f1 = f1_score(test_y, (ens_pred > 0.5).astype(int))
    print(f"\n[RESULT] {name} | AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}\n")
    return {'name': name, 'auc': auc, 'acc': acc, 'f1': f1}
# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HIV', choices=['BACE', 'BBBP', 'HIV'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--full_ablation', action='store_true', help='Run all ablation experiments')
    args = parser.parse_args()

    global SEED, pos_weight, TRAIN_IDX, VAL_IDX, TEST_IDX
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

    # Split
    TRAIN_IDX, VAL_IDX, TEST_IDX = scaffold_split_indices(df['mol'].tolist(), seed=SEED)
    print(f"[SPLIT] Train: {len(TRAIN_IDX)} | Val: {len(VAL_IDX)} | Test: {len(TEST_IDX)}")
 # Normalize
    train_mean = raw_descs[TRAIN_IDX].mean(0)
    train_std = raw_descs[TRAIN_IDX].std(0) + 1e-8
    desc_arr = (raw_descs - train_mean) / train_std

    # Pos weight
    train_labels = df['Class'].iloc[TRAIN_IDX].values
    n_inactive = (train_labels == 0).sum()
    n_active = (train_labels == 1).sum()
    pos_weight = n_inactive / max(1, n_active)
    pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    print(f"[TRAIN CLASS] Inactive: {n_inactive} | Active: {n_active} | pos_weight: {pos_weight.item():.3f}")

    BASE = {'vn': True, 'jk': True, 'n_models': 3, 'epochs': 200, 'layers': 5}

    if args.full_ablation:
        EXPERIMENTS = [
            {'name': '1. Full 2D + BCE (Baseline)', 'vn': True, 'jk': True, 'desc': True, 'pos_weight': True},
            {'name': '2. - Descriptors (BCE)', 'vn': True, 'jk': True, 'desc': False, 'pos_weight': True},
            {'name': '3. - Virtual Node (BCE)', 'vn': False, 'jk': True, 'desc': True, 'pos_weight': True},
            {'name': '4. - Jumping Knowledge (BCE)', 'vn': True, 'jk': False, 'desc': True, 'pos_weight': True},
            {'name': '5. - BCE only (no pos_weight)', 'vn': True, 'jk': True, 'desc': True, 'pos_weight': False},
        ]

        results = []
        for exp in EXPERIMENTS:
            cfg = BASE.copy()
            cfg['vn'] = exp['vn']
            cfg['jk'] = exp['jk']
            r = run_experiment(
                exp['name'], cfg,
                use_desc=exp['desc'],
                use_pos_weight=exp['pos_weight'],
                model_cls=GeoTransformerModel,
                df=df, desc_arr=desc_arr,
                train_idx=TRAIN_IDX, val_idx=VAL_IDX, test_idx=TEST_IDX
            )
            results.append(r)

        # GIN Baseline
        print("\n" + "=" * 70)
        print("  EXTERNAL BASELINE: GIN (5-layer)")
        print("=" * 70)
      gin_result = run_experiment(
            "GIN Baseline (same protocol)",
            BASE, use_desc=True, use_pos_weight=True,
            model_cls=GINBaseline,
            df=df, desc_arr=desc_arr,
            train_idx=TRAIN_IDX, val_idx=VAL_IDX, test_idx=TEST_IDX
        )

        # Summary
        print("=" * 70)
        print("FINAL ABLATION RESULTS")
        print("=" * 70)
        for r in results:
            print(f"  {r['name']:45s}  AUC: {r['auc']:.4f}  Acc: {r['acc']:.4f}  F1: {r['f1']:.4f}")
        print("-" * 70)
        print(f"  {gin_result['name']:45s}  AUC: {gin_result['auc']:.4f}  Acc: {gin_result['acc']:.4f}  F1: {gin_result['f1']:.4f}")
        delta = results[0]['auc'] - gin_result['auc']
        print(f"  {'Delta (Ours - GIN)':45s}  AUC: {delta:+.4f}")
        print("=" * 70)
    else:
        # Single run: Full model vs GIN baseline
        cfg = BASE.copy()
        r1 = run_experiment(
            "GeoTransformer-GINE (Full 2D)", cfg,
            use_desc=True, use_pos_weight=True,
            model_cls=GeoTransformerModel,
            df=df, desc_arr=desc_arr,
            train_idx=TRAIN_IDX, val_idx=VAL_IDX, test_idx=TEST_IDX
        )
        r2 = run_experiment(
            "GIN Baseline", cfg,
            use_desc=True, use_pos_weight=True,
            model_cls=GINBaseline,
            df=df, desc_arr=desc_arr,
            train_idx=TRAIN_IDX, val_idx=VAL_IDX, test_idx=TEST_IDX
        )
        print(f"\nDelta (Ours - GIN): {r1['auc'] - r2['auc']:+.4f}")

if __name__ == "__main__":
    main()
      
