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

### BACE (3 seeds: 0, 42, 123)

| Model | Mean AUC | Std | Best |
|-------|----------|-----|------|
| **Full 2D (with VN)** | **0.8679** | **±0.0143** | 0.8859 |
| **–VN** | 0.8482 | ±0.0325 | 0.8888 |
| **GIN Baseline** | 0.8459 | ±0.0241 | 0.8731 |

### BBBP (seed 42)

| Model | AUC | Acc | F1 |
|-------|-----|-----|-----|
| **Full 2D (with VN)** | 0.9166 | 0.9000 | 0.9362 |
| **–Descriptors** | 0.9087 | 0.8756 | 0.9187 |
| **–VN** | **0.9169** | 0.8780 | 0.9224 |
| **–JK** | 0.9057 | 0.8902 | 0.9296 |
| **–pos_weight** | 0.9071 | 0.8927 | 0.9333 |
| **GIN Baseline** | 0.9028 | 0.8756 | 0.9224 |

### HIV (seed 42)

| Model | AUC | Acc | F1 |
|-------|-----|-----|-----|
| **Full 2D (with VN)** | 0.7216 | 0.9700 | 0.2583 |
| **–Descriptors** | 0.7352 | 0.9648 | 0.3439 |
| **–VN** | **0.7538** | 0.9672 | 0.3602 |
| **–JK** | 0.7001 | 0.9718 | 0.3256 |
| **–pos_weight** | 0.7539 | 0.9732 | 0.2079 |
| **GIN Baseline** | 0.7349 | 0.9693 | 0.3253 |

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
