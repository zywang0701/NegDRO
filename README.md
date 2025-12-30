# NegDRO: Negative Distributionally Robust Optimization for Causal Inference

This repository contains the implementation and experiments for NegDRO (Negative Distributionally Robust Optimization), a method for causal inference in the presence of distributional shifts across environments.

## Project Structure

```
NegDRO-simus/
├── src/                    # Source code
│   ├── negdro.py          # Main NegDRO algorithm
│   ├── negdro_limited.py   # NegDRO for limited interventions
│   ├── methods_wrappers.py # Wrapper functions for comparison methods
│   └── pre_methods/       # Pre-existing methods for comparison
│       ├── cdanzig.py     # Causal Dantzig
│       ├── eills.py       # EILLS method
│       └── others.py      # Other baseline methods (ICP, Anchor, DRIG, etc.)
├── simus/                  # Simulation experiments
│   ├── data.py            # Data generation (SCM models)
│   ├── exp1-gamma.py      # Experiment 1: Varying regularization parameter γ
│   ├── exp1-n.py          # Experiment 1: Varying sample size n
│   ├── exp1-T.py          # Experiment 1: Convergence trajectory over iterations
│   ├── exp2.py            # Experiment 2: Comparison with alternative methods
│   └── exp3-limited.py     # Experiment 3: Limited interventions
├── notebooks/              # Jupyter notebooks for result analysis
│   ├── summary_exp1.ipynb
│   ├── summary_exp2.ipynb
│   └── summary_exp3-limited.ipynb
└── Figures/                # Generated figures

```

## Installation

### Requirements

- Python 3.7+
- NumPy
- PyTorch
- SciPy
- scikit-learn
- cvxpy (for limited interventions)

Install dependencies:
```bash
pip install numpy torch scipy scikit-learn cvxpy
```

## Usage

### Experiment 1: Parameter Sensitivity Analysis

#### Varying regularization parameter γ
```bash
python simus/exp1-gamma.py --p 5 --roundnum 1
```
- Evaluates NegDRO with different γ values in the range [0, 20]
- Parameters: `p ∈ {5, 10, 40}`, `round_num ∈ {1, 2, 3, 4, 5}`

#### Varying sample size n
```bash
python simus/exp1-n.py --p 5 --roundnum 1
```
- Evaluates NegDRO with different sample sizes n ∈ {500, 1000, ..., 20000}
- Parameters: `p ∈ {5, 10, 40}`, `round_num ∈ {1, 2, 3, 4, 5}`

#### Convergence trajectory
```bash
python simus/exp1-T.py --p 5 --roundnum 1
```
- Tracks convergence behavior over iterations
- Parameters: `p ∈ {5, 10, 40}`, `round_num ∈ {1, 2, 3, 4, 5}`

### Experiment 2: Method Comparison

Compare NegDRO with alternative methods (ERM, ICP, EILLS, Anchor) across different dimensions:

```bash
python simus/exp2.py --mode 2 --round_num 1 --num_repeats_per 20
```

- `--mode`: Simulation mode (1: with hidden confounder, 2: no hidden confounder, default=2)
- `--round_num`: Round number starting from 1 (default=1)
- `--num_repeats_per`: Number of repeats per p value (default=20, use 1 for quick testing)
- Varies dimension p ∈ {5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100}
- Timeout: 30 minutes per method
- If a method times out, it is skipped for all larger p values

### Experiment 3: Limited Interventions

```bash
python simus/exp3-limited.py --mode 1 --sim_round 1
```

- Evaluates NegDRO for limited interventions
- Parameters: `mode` (simulation mode), `sim_round` (round number)


### Basic Usage

```python
from src.negdro import negDRO
from simus.data import StructuralCausalModelSimu1

# Generate data
SCM = StructuralCausalModelSimu1(p=10)
x_list, y_list = SCM.sample(n=10000, mode=2, seed=42)

# Run NegDRO
b_final, _ = negDRO(
    x_list, y_list, 
    gamma=20.0,           # Regularization parameter
    early_stop=True,      # Enable early stopping
    num_iter=1500,        # Maximum iterations
    seed=42               # Random seed
)
```

## Results

Results are saved as pickle files (not included in repository due to size). Use the Jupyter notebooks in `notebooks/` to analyze and visualize the results.


