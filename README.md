# GeoTransformer-GINE: Rethinking Virtual Nodes in Molecular Graph Transformers

Official implementation of **GeoTransformer-GINE**, a hybrid GINE + Graph Transformer architecture for 2D molecular property prediction on MoleculeNet datasets (BACE, BBBP, HIV) under scaffold-split evaluation.

> **Key Finding:** Virtual nodes (VN) can destabilize attention in graph transformers on imbalanced molecular datasets, causing attention-score collapse. Our VN-free variant achieves superior and more robust performance on HIV (+0.032 AUC) and BBBP while remaining competitive on BACE.

---

## Architecture

- **5-layer hybrid stack:** GINE → GINE → GraphTransformerConv → GraphTransformerConv → GINE
- **Attentional pooling:** Gated attention + mean + max pooling
- **Molecular descriptors:** 12 physicochemical properties + 128-bit Morgan fingerprint (projected to 32)
- **Training tricks:** EMA, SWA, DropEdge, LayerScale, StochasticDepth
- **Scaffold split:** Proportional allocation (random scaffold shuffle, no size sorting) to preserve class balance

---

## Results
### HIV (seed 42)

| Model | AUC | Acc | F1 |
|-------|-----|-----|-----|
| **Full 2D (with VN)** | 0.7216 ±0.0143| 0.9700 | 0.2583 |
| **–Descriptors** | 0.7352±0.121 | 0.9648 | 0.3439 |
| **–VN** | **0.7538**±0.0198 | 0.9672 | 0.3602 |
| **–JK** | 0.700±0.0321 | 0.9718 | 0.3256 |
| **–pos_weight** | 0.7539 ±0.0132 | 0.9732 | 0.2079 |
| **GIN Baseline** | 0.7349 ±0.0120 | 0.9693 | 0.3253 |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/GeoTransformer-GINE.git
cd GeoTransformer-GINE
pip install -r requirements.txt
```

## Usage

### Run a single experiment
```bash
# BACE
python train.py --dataset BACE --seed 42

# HIV
python train.py --dataset HIV --seed 42

# BBBP
python train.py --dataset BBBP --seed 42
```

### Run full ablation (all 5 experiments + GIN baseline)
```bash
python train.py --dataset HIV --seed 42 --full_ablation
```

---

## Project Structure

```
.
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

---

## Dependencies

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric
- RDKit
- scikit-learn, pandas, numpy, tqdm

---

## License

This project is licensed under the MIT License
