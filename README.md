# Quantum Control Optimization for Single-Qubit Gates (X, Y, X90, Y90)

This repository contains a modular, research-oriented framework for optimizing and benchmarking single-qubit control pulses in a driven cavity–qubit system. It implements IQ (two-quadrature) Fourier controls, QuTiP-based open-system dynamics, PTM-style gate costs, and a simplified randomized benchmarking-style sequence fidelity analysis. For citation and archival reference, a Zenodo DOI is available: <https://doi.org/10.5281/zenodo.17499171>.

Key features:

- Process-inspired PTM cost for single-qubit gates (X, Y, √X, √Y, H, S, T)
- IQ control (I/Q) parameterization with real cosine/sine Fourier series
- QuTiP-based time dynamics with optional dissipation and dephasing
- Multi-start (global) optimization with L-BFGS-B or Nelder–Mead
- Simplified randomized benchmarking sequence fidelity scans with exponential fits
- Publication-oriented plotting utilities and summary figures

## Repository layout

```text
public/
├─ src/                        # Library code (importable as `src`)
│  ├─ config.py                # Physical constants and defaults
│  ├─ dynamics.py              # Lab-frame simulation for IQ controls
│  ├─ optimization.py          # Cost builders and gate optimizers
│  ├─ benchmarking.py          # Sequence fidelity scans and fits
│  ├─ plotting.py              # Publication-quality plots
│  ├─ operators.py, states.py, utils.py, pulse_shapes.py
│  └─ __init__.py              # Convenient exports
├─ data/                       # Sample .npz payloads (kept small)
│  └─ payloads/                # Large payloads (ignored by Git)
├─ results/                    # Generated results (ignored by Git)
├─ 1-...ipynb .. 5-...ipynb    # Notebooks demonstrating optimization workflows
├─ create_dataset.ipynb        # Notebook to create/generate the example dataset
├─ random_benchmarking_*.py    # Script variants (optional)
├─ run_rnn_training.py         # Optional NN-assisted initial guesses
├─ README.md                   # This file
├─ LICENSE                     # MIT License
├─ CITATION.cff                # Citation metadata
├─ CONTRIBUTING.md             # Contributing guidelines
├─ CODE_OF_CONDUCT.md          # Community expectations
├─ SECURITY.md                 # Reporting security issues
├─ requirements.txt            # Python dependencies
└─ .gitignore                  # Ignored files and folders
```

## Installation

1) Create and activate a Python 3.10+ environment (conda, venv, etc.), then install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- QuTiP may require system packages (FFTW/OpenMP) depending on your OS. See <https://qutip.org> for platform-specific install tips.
- TensorFlow and scikit-learn are only required if you plan to run the NN-related parts (e.g., `run_rnn_training.py` or NN-initialized optimizations). You can comment them out in `requirements.txt` if not needed.
- Matplotlib LaTeX rendering is enabled by default in `src/plotting.py`. If you do not have LaTeX installed, instantiate the plotter with `use_latex=False` or toggle it off in the code.

## Quickstart (Python API)

Below is a minimal example that optimizes a Y gate using the IQ PTM cost and prints a short summary.

```python
import numpy as np
from src import (
    time_array, CAVITY_FREQUENCY, QUBIT_FREQUENCY, COUPLING_STRENGTH,
    optimize_y_gate, QuantumControlPlotter
)
from src.operators import get_collapse_operators
from src.dynamics import simulate_controlled_evolution

# Dissipators (tunable; values from src/config.py)
collapse_ops = get_collapse_operators()

# System parameters for the simulator
system_params = {
    'omega_r': float(CAVITY_FREQUENCY),
    'omega_t': float(QUBIT_FREQUENCY),
    'coupling_g': float(COUPLING_STRENGTH),
}

# Optimize Y gate (π rotation around Y)
payload = optimize_y_gate(
    system_params=system_params,
    collapse_ops=collapse_ops,
    fourier_modes=16,
    max_iterations=3,
    coefficient_bounds=(-0.32, 0.32),
    leakage_weight=0.05,
    save=True,
)

print("Final cost:", payload['optimization_result'].fun)
print("Fourier modes:", payload['fourier_modes'])
print("Control period (ns):", payload['control_period'])

# (Optional) plot using the helper
plotter = QuantumControlPlotter(show_plots=True, use_latex=False, figure_format='png')
# You can simulate and plot Bloch components using the notebook examples in 4-Quantum_Control_Optimization-Y.ipynb
```

## Reproducing the workflows (Notebooks)

The numbered notebooks demonstrate the gate-optimization and evaluation workflows:

- `1-Quantum_Control_Optimization-DummySignals.ipynb`
- `2-Quantum_Control_Optimization-X.ipynb`
- `3-Quantum_Control_Optimization-X90.ipynb`
- `4-Quantum_Control_Optimization-Y.ipynb`
- `5-Quantum_Control_Optimization-Y90.ipynb`

Additionally:

- `create_dataset.ipynb` — builds and saves the example dataset used by the notebooks under `data/` and `results/`.

Open them in Jupyter Lab/VS Code and run sequentially. Each notebook:

- Builds/loads problem configuration from `src/config.py`
- Runs the corresponding gate optimization from `src/optimization.py`
- Optionally runs a sequence fidelity scan from `src/benchmarking.py`
- Generates figures with `src/plotting.py`

Outputs and intermediate artifacts are written to `results/` (ignored by Git) and small curated payloads in `data/`.
The dataset creation notebook (`create_dataset.ipynb`) exports reproducible payloads into `data/` (and larger run artifacts into `data/payloads/`).

## Data and results

- Small example `.npz` payloads are included in `data/` for convenience. Larger or run-specific payloads are ignored under `data/payloads/` (see `.gitignore`).
- New runs store metadata as JSON plus arrays as NPZ in `results/` with timestamped filenames; these are regenerable and therefore not tracked.

## Citing

If this work is useful, please cite it. See `CITATION.cff` for a ready-to-use citation (GitHub’s “Cite this repository”) and the Zenodo record:

- DOI: <https://doi.org/10.5281/zenodo.17499171>
- Author: Denys Derlian Carvalho Brito (ORCID: <https://orcid.org/0009-0007-1669-0340>)
- Advisor: André Jorge Carvalho Chaves (ORCID: <https://orcid.org/0000-0003-1381-8568>)

## License

This project is licensed under the MIT License — see `LICENSE` for details.

## Acknowledgements

- Built on [QuTiP](https://qutip.org/)
- Uses SciPy optimizers and NumPy/SciPy/Matplotlib for scientific computing and visualization
