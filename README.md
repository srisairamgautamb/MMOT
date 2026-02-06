# Multi-Period Martingale Optimal Transport

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.05290-b31b1b.svg)](https://arxiv.org/abs/2601.05290)

Reference implementation for the paper:

> **Multi-Period Martingale Optimal Transport: Classical Theory, Neural Acceleration, and Financial Applications**  
> Sri Sairam Gautam B.  
> School of Engineering, Jawaharlal Nehru University, New Delhi, India  
> arXiv:2601.05290

---

## Overview

This repository provides a unified computational framework for multi-period martingale optimal transport (MMOT). The implementation includes:

1. **Classical Solvers**: Sinkhorn-Knopp and ADMM-based algorithms with incremental update schemes.
2. **Neural Solver**: A hybrid Transformer architecture with Newton-Raphson projection for real-time inference.
3. **Validation Suite**: Comprehensive tests on synthetic (GBM, Merton, Heston) and real market data (S&P 500 options).

---

## Key Results

| Metric | Value |
|--------|-------|
| Discrete Convergence Rate | O(sqrt(dt) log(1/dt)) |
| Algorithmic Convergence | (1 - kappa)^(2/3) |
| Martingale Constraint Error | < 1e-6 |
| Neural Inference Time | 2.9 ms |
| Speedup vs Classical | 1,597x |

Validated on 12,000 synthetic instances and 120 real market scenarios.

---

## Installation

```bash
git clone https://github.com/srisairamgautamb/MMOT.git
cd MMOT
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, PyTorch 2.0+, NumPy, SciPy, Matplotlib.

---

## Usage

### Reproduce Paper Experiments

Run the master script to generate Figures 9-13 and Tables 2-4:

```bash
python run_all_experiments.py
```

Outputs are saved to `figures/phase2b/`.

### Neural Solver Validation

```bash
# Real market data validation (Table 7)
python neural/validation/validate_real_data_200.py

# Synthetic drift analysis (Table 3)
python neural/validation/comprehensive_drift_check.py
```

### Programmatic Usage

```python
import torch
from neural.models.architecture import NeuralDualSolver

# Initialize model
model = NeuralDualSolver(grid_size=150, hidden_dim=256, num_layers=3)
model.load_state_dict(torch.load('checkpoints/best_model.pt', map_location='cpu'))
model.eval()

# Inference
marginals = torch.randn(1, 11, 150).abs()
with torch.no_grad():
    u_pred, h_pred = model(marginals)
```

---

## Repository Structure

```
MMOT/
├── mmot/                       # Classical solver package
│   ├── core/                   # Sinkhorn, ADMM, and stable solver implementations
│   ├── utils/                  # Grid construction and helper functions
│   └── validation/             # Theoretical verification scripts
│
├── neural/                     # Neural solver package
│   ├── models/                 # Transformer architecture with projection layers
│   ├── training/               # Training loops, loss functions, and metrics
│   └── validation/             # Ablation studies, drift analysis, and backtests
│
├── experiments/                # Paper experiment scripts (Tasks 1-7)
│   ├── task1_algorithm_comparison.py
│   ├── task2_donsker_rate.py
│   ├── task3_lp_comparison.py
│   ├── task4_heston_comparison.py
│   ├── task5_robustness.py
│   ├── task6_transaction_costs.py
│   └── task7_scalability.py
│
├── figures/                    # Generated figures and plots
├── checkpoints/                # Pre-trained model weights
├── submission_final/           # LaTeX source and compiled PDF
├── requirements.txt            # Python dependencies
├── run_all_experiments.py      # Master reproduction script
└── LICENSE                     # MIT License
```

---

## Citation

```bibtex
@article{gautamb2026mmot,
  title={Multi-Period Martingale Optimal Transport: Classical Theory, 
         Neural Acceleration, and Financial Applications},
  author={Gautam B., Sri Sairam},
  journal={arXiv preprint arXiv:2601.05290},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Sri Sairam Gautam B.  
School of Engineering, Jawaharlal Nehru University  
bsrisa59_soe@jnu.ac.in
