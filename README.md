# GMRF-VAC

GMRF-VAC is a sparse Gaussian Markov random field framework for lattice relaxation around a true vacancy in a harmonic crystal.

The implementation treats a vacancy through physical node-and-bond deletion in a periodic harmonic lattice. It evaluates normalized pseudo-determinant contributions, solves the prescribed Kanzaki-type force relaxation problem, and includes numerical checks for translational zero modes, regularization sensitivity, and pseudo-determinant evaluation.

## Current status

This repository accompanies the manuscript:

> Mahmoud Hinnawi, “Gaussian Markov Random Field Model for Lattice Relaxation Around a True Vacancy in a Harmonic Crystal: Normalized Harmonic Contributions and Computational Implementation,” submitted to *Computer Physics Communications*.

## Requirements

* Python 3.10 or later
* NumPy
* SciPy

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Quick start

The main reproduction script is:

```bash
python code/GMRF-VAC_Table2.py
```

This script constructs the perfect and true-vacancy periodic square-lattice Laplacians, computes the normalized pseudo-determinant ratio, evaluates the vacancy-induced relaxation contribution, and performs the finite-size extrapolation.

The reference calculation uses:

* Linear dimension: `L = 32`
* Perfect-lattice site count: `N = 1024`
* Vacancy coordinate: `(16, 16)`
* Prescribed force magnitude: `f0 = 0.05`

Additional validation and benchmarking scripts are provided in the `code/` directory:

```text
GMRF-VAC_regularization_sensitivity.py
GMRF-VAC_SparseLU_Benchmark.py
GMRF-VAC_run_grounded_node_comparison.py
GMRF-VAC_PseudoDet_Spectral_Validation.py
```

These scripts reproduce the regularization-sensitivity check, sparse-LU performance benchmark, grounded-node comparison, and pseudo-determinant spectral validation reported in the manuscript.

## Repository structure

```text
GMRF-VAC/
├── README.md
├── code/
│   ├── GMRF-VAC_Table2.py
│   ├── GMRF-VAC_regularization_sensitivity.py
│   ├── GMRF-VAC_SparseLU_Benchmark.py
│   ├── GMRF-VAC_run_grounded_node_comparison.py
│   └── GMRF-VAC_PseudoDet_Spectral_Validation.py
```


## License

GMRF-VAC is released under the MIT License. See `LICENSE` for the full license text.

## Citation

If you use GMRF-VAC in academic work, please cite the accompanying manuscript listed above. Full publication details and a DOI will be added after publication.
