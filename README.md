# GMRF-VAC

GMRF-VAC is a sparse Gaussian Markov random field framework for lattice relaxation around a true vacancy in a harmonic crystal.

The implementation treats a vacancy through physical node-and-bond deletion in a periodic harmonic lattice. It evaluates normalized pseudo-determinant contributions, solves the prescribed Kanzaki-type force relaxation problem, and includes numerical checks for translational zero modes and regularization sensitivity.

## Current status

This repository is being prepared to accompany the manuscript:

> M. A. Hinnawi, “Gaussian Markov Random Field Model for Lattice Relaxation Around a True Vacancy in a Harmonic Crystal: Normalized Harmonic Contributions and Computational Implementation,” submitted to *Computer Physics Communications*.

## Requirements

- Python 3.10 or later
- NumPy
- SciPy

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Quick start

Run the reference-system validation script:

```bash
python scripts/run_L32_validation.py
```

The reference calculation uses a periodic square lattice with:

- Linear dimension: `L = 32`
- Perfect-lattice site count: `N = 1024`
- Vacancy coordinate: `(16, 16)`
- Prescribed force magnitude: `f0 = 0.05`

The completed validation script will reproduce the determinant ratio, relaxation contribution, force-balance check, and normalized result reported in the accompanying manuscript.

## License

GMRF-VAC is released under the MIT License. See `LICENSE` for the full license text.

## Citation

If you use GMRF-VAC in academic work, please cite the accompanying manuscript listed above. Full publication details and a DOI will be added after publication.
