
"""
GMRF-VAC Pseudo-Determinant / Spectral Validation

Independently validates the pseudo-determinant calculation used
in the GMRF-VAC model by comparing:

    1. Cofactor + sparse LU evaluation
    2. Product of the non-zero eigenvalues

The validation is performed for small perfect periodic square
lattices with L = 3,...,8.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ============================================================
# GMRF-VAC
# Pseudo-determinant / Spectral Validation
#
# Purpose:
#   Independently validate the pseudo-determinant calculation
#   used in the GMRF-VAC model by comparing:
#
#       1. Cofactor + sparse LU calculation
#       2. Product of the non-zero eigenvalues
#
# The test is performed for small perfect periodic
# square lattices, 3 <= L <= 8.
# ============================================================


# ============================================================
# 1. Build perfect periodic square-lattice Laplacian
# ============================================================

def build_perfect_laplacian(L: int) -> sp.csr_matrix:
    """
    Construct the N x N graph Laplacian of an L x L
    periodic square lattice using 0-based indexing.

    N = L^2

    Each node has four periodic nearest-neighbour
    contributions.
    """

    N = L * L

    rows = []
    cols = []
    data = []

    def idx(i, j):
        return i * L + j

    for i in range(L):
        for j in range(L):

            p = idx(i, j)

            # Diagonal degree
            rows.append(p)
            cols.append(p)
            data.append(4.0)

            # Periodic nearest neighbours
            neighbours = [
                ((i - 1) % L, j),      # up
                ((i + 1) % L, j),      # down
                (i, (j - 1) % L),      # left
                (i, (j + 1) % L)       # right
            ]

            for ni, nj in neighbours:

                q = idx(ni, nj)

                rows.append(p)
                cols.append(q)
                data.append(-1.0)

    Lmat = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(N, N)
    ).tocsr()

    return Lmat


# ============================================================
# 2. Pseudo-determinant from cofactor + sparse LU
# ============================================================

def log_pdet_lu(Lmat: sp.csr_matrix) -> float:
    """
    Compute

        ln pdet(L)
        =
        ln(N) + ln(det(L_reduced))

    using a Laplacian cofactor and sparse LU factorization.
    """

    N = Lmat.shape[0]

    # Delete first row and first column
    Lred = Lmat[1:, 1:].tocsc()

    # Sparse LU factorization
    lu = spla.splu(Lred)

    # Determinant of U from diagonal pivots
    log_det_reduced = np.sum(
        np.log(np.abs(lu.U.diagonal()))
    )

    return float(
        np.log(N) + log_det_reduced
    )


# ============================================================
# 3. Pseudo-determinant from non-zero eigenvalues
# ============================================================

def log_pdet_eigenvalues(Lmat: sp.csr_matrix) -> float:
    """
    Compute

        ln pdet(L) = sum ln(lambda_i)

    over the positive (non-zero) eigenvalues.
    """

    # Small systems only: convert to dense matrix
    A = Lmat.toarray()

    # Symmetric Laplacian -> use eigh/eigvalsh
    eigenvalues = np.linalg.eigvalsh(A)

    # Numerical tolerance for the translational zero mode
    tolerance = 1e-12

    positive_eigenvalues = eigenvalues[
        eigenvalues > tolerance
    ]

    if len(positive_eigenvalues) != A.shape[0] - 1:
        raise ValueError(
            "Unexpected number of non-zero eigenvalues."
        )

    return float(
        np.sum(np.log(positive_eigenvalues))
    )


# ============================================================
# 4. Validation for one lattice size
# ============================================================

def validate_one_L(L: int):
    """
    Compare the two independent pseudo-determinant
    calculations for one lattice size.
    """

    N = L * L

    Lmat = build_perfect_laplacian(L)

    # Basic structural checks
    symmetry_error = (
        np.max(np.abs((Lmat - Lmat.T).data))
        if (Lmat - Lmat.T).nnz > 0
        else 0.0
    )

    row_sums = np.asarray(
        Lmat.sum(axis=1)
    ).ravel()

    row_sum_error = np.max(
        np.abs(row_sums)
    )

    # Two independent calculations
    log_pdet_lu_value = log_pdet_lu(Lmat)
    log_pdet_eig_value = log_pdet_eigenvalues(Lmat)

    # Difference between the two log pseudo-determinants
    absolute_log_difference = abs(
        log_pdet_lu_value - log_pdet_eig_value
    )

    # Also compare the pseudo-determinants themselves
    pdet_lu = np.exp(log_pdet_lu_value)
    pdet_eig = np.exp(log_pdet_eig_value)

    absolute_pdet_difference = abs(
        pdet_lu - pdet_eig
    )

    relative_pdet_difference = (
        absolute_pdet_difference / abs(pdet_eig)
    )

    return {
        "L": L,
        "N": N,
        "symmetry_error": symmetry_error,
        "row_sum_error": row_sum_error,
        "log_pdet_lu": log_pdet_lu_value,
        "log_pdet_eigenvalues": log_pdet_eig_value,
        "absolute_log_difference": absolute_log_difference,
        "absolute_pdet_difference": absolute_pdet_difference,
        "relative_pdet_difference": relative_pdet_difference,
    }


# ============================================================
# 5. Run complete validation
# ============================================================

def run_validation():

    lattice_sizes = [3, 4, 5, 6, 7, 8]

    print()
    print("=" * 100)
    print("GMRF-VAC PSEUDO-DETERMINANT / SPECTRAL VALIDATION")
    print("=" * 100)

    print()
    print("Comparison:")
    print("  Method 1: Cofactor + sparse LU")
    print("  Method 2: Product of all non-zero eigenvalues")

    print()
    print(
        f"{'L':>4}"
        f"{'N':>6}"
        f"{'ln pdet (LU)':>20}"
        f"{'ln pdet (eig)':>20}"
        f"{'|delta log|':>16}"
        f"{'relative pdet diff.':>22}"
    )

    print("-" * 100)

    results = []

    for L in lattice_sizes:

        result = validate_one_L(L)

        results.append(result)

        print(
            f"{result['L']:4d}"
            f"{result['N']:6d}"
            f"{result['log_pdet_lu']:20.12f}"
            f"{result['log_pdet_eigenvalues']:20.12f}"
            f"{result['absolute_log_difference']:16.3e}"
            f"{result['relative_pdet_difference']:22.3e}"
        )

    print("-" * 100)

    # Maximum discrepancies
    max_log_difference = max(
        r["absolute_log_difference"]
        for r in results
    )

    max_absolute_pdet_difference = max(
        r["absolute_pdet_difference"]
        for r in results
    )

    max_relative_pdet_difference = max(
        r["relative_pdet_difference"]
        for r in results
    )

    max_symmetry_error = max(
        r["symmetry_error"]
        for r in results
    )

    max_row_sum_error = max(
        r["row_sum_error"]
        for r in results
    )

    print()
    print("FINAL VALIDATION RESULTS")
    print("-" * 60)

    print(
        f"Maximum symmetry error       : "
        f"{max_symmetry_error:.3e}"
    )

    print(
        f"Maximum row-sum error        : "
        f"{max_row_sum_error:.3e}"
    )

    print(
        f"Maximum |delta log pdet|     : "
        f"{max_log_difference:.3e}"
    )

    print(
        f"Maximum absolute pdet diff.  : "
        f"{max_absolute_pdet_difference:.3e}"
    )

    print(
        f"Maximum relative pdet diff.  : "
        f"{max_relative_pdet_difference:.3e}"
    )

    print()
    print(
        "Validation criterion:"
    )

    print(
        "  The LU/cofactor and spectral pseudo-determinants"
    )

    print(
        "  should agree to numerical precision for these"
    )

    print(
        "  small perfect lattices."
    )

    print()

    if max_log_difference < 1e-10:
        print("VALIDATION PASSED")
        print(
            "The two pseudo-determinant calculations agree "
            "to better than 1e-10 in log space."
        )
    else:
        print("VALIDATION FAILED")
        print(
            "The discrepancy exceeds the 1e-10 validation threshold."
        )

    print()
    print("=" * 100)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_validation()
