# KpH-EDMD: Koopman-Port-Hamiltonian Identification and Control

KpH-EDMD is a data-driven system identification framework that fits a
Koopman operator approximation with enforced port-Hamiltonian structure
via a convex SDP. The generator is decomposed as F = KJ − KR, where KJ
is skew-symmetric (lossless J structure) and KR ≥ 0 (dissipative R
structure), and passivity is guaranteed at every discrete time step
through a Schur-complement LMI. This repository accompanies the paper
listed below and provides a fully reproducible two-link planar robot
example with fully coupled inertial dynamics, demonstrating that
KpH-EDMD achieves a 100% RMSE improvement over unconstrained EDMD
while maintaining exact passivity by construction.

## Citation

```bibtex
@inproceedings{preciado2026kph,
  author    = {Victor M. Preciado},
  title     = {A Koopman-Port-Hamiltonian Framework for
               Data-Driven Modeling and Control},
  booktitle = {submitted to IEEE Conference on Decision and Control (CDC)},
  year      = {2026}
}
```

## Requirements

- Python >= 3.9
- numpy
- scipy
- cvxpy >= 1.3
- matplotlib
- mosek (optional, recommended for speed; falls back to clarabel)

## Installation

```bash
git clone https://github.com/vmpreciado/KpH-EDMD.git
cd KpH-EDMD
pip install -r requirements.txt
```

## Reproducing the Paper Results

```bash
python examples/two_link_robot.py
```

This script reproduces Table I and Figures 3–4 of Preciado (CDC 2025).
Paper figures are saved as PDF to `figures/` and key numerical results
(RMSE, passivity residuals, spectral radii, settling times) are printed
to the console.

## Repository Structure

```
KpH-EDMD/
├── README.md                  # this file
├── requirements.txt           # Python dependencies
├── kph/
│   ├── __init__.py            # package entry point; exports KpHEDMD
│   └── edmd.py                # KpHEDMD class + fit_edmd + passivity_residual
├── examples/
│   └── two_link_robot.py      # self-contained reproduction script
└── figures/
    ├── fig_robot_joints.pdf   # joint angles under damping-injection control
    └── fig_robot_energy.pdf   # total energy + passivity residual
```

### `kph/edmd.py`

| Symbol | Description |
|--------|-------------|
| `KpHEDMD` | Main class: solves the KpH SDP, exposes `K_disc`, `Ku_disc`, `predict()` |
| `fit_edmd` | Unconstrained EDMD baseline via ridge regression |
| `passivity_residual` | One-step PR_k computation |

## License

MIT License. Copyright (c) 2026 Victor M. Preciado.
