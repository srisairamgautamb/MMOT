# Multi-Marginal Optimal Transport: Theory, Algorithms, and Neural Approximation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2512.#####-b31b1b.svg)](https://arxiv.org/abs/2512.#####)

Official implementation of the paper **"Multi-Period Martingale Optimal Transport: Classical Theory, Neural Acceleration, and Financial Applications"**.

## 🎯 Comparison with State-of-the-Art
This work bridges the gap between theoretical MMOT and production deployment:

| Feature | Benamou et al. (2024) | This Work |
|---------|------------------------|-----------|
| **Convergence Rate** | Qualitative ($\Gamma$-convergence) | **Quantitative**: $O(\sqrt{\Delta t \log(1/\Delta t)})$ |
| **Algorithm** | Sinkhorn (Classical) | **Hybrid**: Transformer + Newton-Raphson |
| **Inference Speed** | Seconds/Minutes | **2.9 ms** ($1597\times$ speedup) |
| **Martingale Error** | Exact | $< 10^{-6}$ (via Projection Layer) |

## 🚀 Key Results Reproduction

### 1. Installation
```bash
git clone https://github.com/srisairamgautamb/MMOT.git
cd MMOT
pip install -r requirements.txt
```

### 2. Main Paper Experiments
To reproduce Figures 9-13 and Tables 2-4 (Classical & Hybrid Comparisons):
```bash
python run_all_experiments.py
```
*Outputs are saved to `figures/phase2b/`.*

### 3. Neural Solver Validation (SOTA Comparison)
To reproduce the **Deep Learning results** (Table 6, Table 7, and the 1,597x speedup):
```bash
# Validate Real Market Data (Table 7) & SOTA Benchmarks
python neural/validation/validate_real_data_200.py

# Run Comprehensive Drift Check (Table 3 / Synthetic)
python neural/validation/comprehensive_drift_check.py
```

## 📂 Repository Structure

```
MMOT/
├── mmot/                    # Classical Solver Package
│   ├── core/               # Sinkhorn & ADMM Solvers
│   └── validation/         # Theoretical Verification Scripts
│
├── neural/                  # Neural Solver Package
│   ├── models/             # Transformer Architecture (Physics-Informed)
│   ├── training/           # Training Loops & Loss Functions
│   └── validation/         # Speedup & Accuracy Tests
│
├── experiments/             # Paper Experiments (Tasks 1-7)
├── submission_final/        # LaTeX Source & PDFs
└── run_all_experiments.py   # Master Reproduction Script
```

## 🛠️ Usage Example

**Neural Inference (Fast):**
```python
import torch
from neural.models.architecture import NeuralDualSolver

# Load Pre-trained Model
model = NeuralDualSolver(grid_size=150, hidden_dim=256, num_layers=3)
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location='cpu'))
model.eval()

# Solve for N=10 marginals (Batch Inference)
marginals = torch.randn(1, 11, 150).abs()  # Example input
with torch.no_grad():
    u_pred, h_pred = model(marginals)
```

## 📜 Citation

If you use this code, please cite:

```bibtex
@article{gautamb2026mmot,
  title={Multi-Period Martingale Optimal Transport: Classical Theory, Neural Acceleration, and Financial Applications},
  author={Gautamb, Sri Sairam},
  journal={Submitted to Mathematical Finance},
  year={2026}
}
```

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.
