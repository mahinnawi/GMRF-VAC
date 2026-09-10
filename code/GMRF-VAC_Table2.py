
"""GMRF-VAC: Authoritative Table 2 Generator

Reproduces the finite-size GMRF-VAC results for
L = 8, 12, 16, 20, 24, 32, 48, 64, including
the C_L values, relaxation contributions, total
normalized free-energy contributions, and the
C_L = C_inf + A/L^2 extrapolation.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as csgraph
import csv
from pathlib import Path


# ============================================================
# GMRF-VAC
# Authoritative Table 2 Generator
#
# Computes:
#   1. Perfect periodic square-lattice Laplacian
#   2. True-vacancy defective Laplacian
#   3. ln[pdet(L_def) / pdet(L)]
#   4. Kanzaki-force relaxation
#   5. Total normalized free-energy contribution
#   6. Finite-size extrapolation
#
# Default parameters:
#   beta  = 1
#   kappa = 1
#   f0    = 0.05
#   epsilon = 1e-10
#
# L values:
#   8, 12, 16, 20, 24, 32, 48, 64
# ============================================================


# ============================================================
# 1. Build perfect periodic square-lattice Laplacian
# ============================================================

def build_perfect_laplacian(L: int) -> sp.csr_matrix:
    """
    Construct the graph Laplacian of an L x L periodic
    square lattice.

    N = L^2.

    Each node has four nearest neighbours.

    The matrix is

        L = D - A

    with diagonal entries +4 and nearest-neighbour
    off-diagonal entries -1.
    """

    N = L * L

    row_indices = []
    col_indices = []
    data = []

    for i in range(L):
        for j in range(L):

            idx = i * L + j

            neighbours = [
                ((i - 1) % L) * L + j,      # Up
                ((i + 1) % L) * L + j,      # Down
                i * L + ((j - 1) % L),      # Left
                i * L + ((j + 1) % L)       # Right
            ]

            # Diagonal degree
            row_indices.append(idx)
            col_indices.append(idx)
            data.append(4.0)

            # Off-diagonal bonds
            for nbr in neighbours:
                row_indices.append(idx)
                col_indices.append(nbr)
                data.append(-1.0)

    Lmat = sp.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(N, N)
    ).tocsr()

    return Lmat


# ============================================================
# 2. Build true-vacancy defective Laplacian
# ============================================================

def build_defective_laplacian(L: int, vac_idx: int):
    """
    Construct the true-vacancy graph Laplacian.

    One node is physically deleted together with all
    incident bonds.

    The remaining graph therefore has N-1 nodes.

    The four nearest neighbours of the vacancy have
    degree 3 instead of degree 4.

    Returns
    -------
    L_def : csr_matrix
        (N-1) x (N-1) defective Laplacian.

    old_to_new : dict
        Mapping from original node indices to defective
        node indices.

    """

    N = L * L

    old_nodes = [
        idx for idx in range(N)
        if idx != vac_idx
    ]

    old_to_new = {
        old: new
        for new, old in enumerate(old_nodes)
    }

    N_def = N - 1

    row_indices = []
    col_indices = []
    data = []

    degrees = np.zeros(N_def, dtype=float)

    for i in range(L):
        for j in range(L):

            old_idx = i * L + j

            # Vacancy is physically removed
            if old_idx == vac_idx:
                continue

            new_idx = old_to_new[old_idx]

            neighbours = [
                ((i - 1) % L) * L + j,      # Up
                ((i + 1) % L) * L + j,      # Down
                i * L + ((j - 1) % L),      # Left
                i * L + ((j + 1) % L)       # Right
            ]

            # Remove bonds connecting to vacancy
            valid_nbrs = [
                nbr for nbr in neighbours
                if nbr != vac_idx
            ]

            # Physical degree after vacancy removal
            degree = len(valid_nbrs)

            degrees[new_idx] = degree

            # Diagonal degree contribution
            row_indices.append(new_idx)
            col_indices.append(new_idx)
            data.append(float(degree))

            # Off-diagonal neighbour contributions
            for nbr in valid_nbrs:

                new_nbr = old_to_new[nbr]

                row_indices.append(new_idx)
                col_indices.append(new_nbr)
                data.append(-1.0)

    L_def = sp.coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(N_def, N_def)
    ).tocsr()

    return L_def, old_to_new


# ============================================================
# 3. Basic Laplacian validation
# ============================================================

def validate_laplacian(Lmat, name="Laplacian"):
    """
    Validate symmetry, zero row sums, and node degrees.
    """

    print()
    print(f"Validation: {name}")
    print("-" * 60)

    # Symmetry
    symmetry_error = np.max(
        np.abs(
            (Lmat - Lmat.T).data
        )
    ) if (Lmat - Lmat.T).nnz > 0 else 0.0

    # Row sums
    row_sums = np.asarray(
        Lmat.sum(axis=1)
    ).ravel()

    row_sum_error = np.max(
        np.abs(row_sums)
    )

    # Degree = diagonal
    degrees = Lmat.diagonal()

    degree_min = int(np.min(degrees))
    degree_max = int(np.max(degrees))

    print(
        f"Maximum symmetry error: {symmetry_error:.3e}"
    )

    print(
        f"Maximum row-sum error:  "
        f"{row_sum_error:.3e}"
    )

    print(
        f"Degree range:             "
        f"{degree_min} - {degree_max}"
    )

    unique, counts = np.unique(
        degrees.astype(int),
        return_counts=True
    )

    for degree, count in zip(unique, counts):
        print(
            f"Degree {degree}: "
            f"{count} nodes"
        )

    passed = (
        symmetry_error < 1e-12
        and row_sum_error < 1e-12
    )

    print(
        "Validation: "
        + ("PASSED" if passed else "FAILED")
    )

    return passed


# ============================================================
# 4. Connectivity validation
# ============================================================

def validate_connectivity(Lmat, name="Laplacian"):
    """
    Check that the graph represented by the Laplacian
    is connected.
    """

    print()
    print(f"Connectivity check: {name}")
    print("-" * 60)

    # Graph connectivity based on off-diagonal structure
    adjacency = Lmat.copy()

    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    adjacency.data = np.ones_like(adjacency.data)

    n_components, labels = csgraph.connected_components(
        adjacency,
        directed=False
    )

    print(
        f"Number of connected components: "
        f"{n_components}"
    )

    passed = (n_components == 1)

    print(
        "Connectivity check: "
        + ("PASSED" if passed else "FAILED")
    )

    return passed


# ============================================================
# 5. Reduced Laplacian
# ============================================================

def reduced_laplacian(Lmat):
    """
    Remove the first row and first column.

    For a connected graph Laplacian with n vertices,

        pdet(L) = n * det(L_reduced)

    by the matrix-tree theorem.
    """

    return Lmat[1:, 1:].tocsc()


# ============================================================
# 6. Compute log pseudo-determinant
# ============================================================

def compute_log_pdet(
    Lmat: sp.csr_matrix,
    validate=True,
    name="Laplacian"
):
    """
    Compute

        ln pdet(L)

    using the cofactor formulation:

        ln pdet(L)
        =
        ln(n)
        +
        ln det(L_reduced)

    The determinant of the reduced Laplacian is obtained
    from sparse LU factorization.

    Returns
    -------
    log_pdet : float
    """

    n = Lmat.shape[0]

    L_red = reduced_laplacian(Lmat)

    if validate:

        validate_laplacian(
            Lmat,
            name=name
        )

        validate_connectivity(
            Lmat,
            name=name
        )

        print()
        print(
            f"Reduced matrix check: "
            f"{name}"
        )
        print("-" * 60)

        print(
            f"Reduced matrix dimension: "
            f"{L_red.shape[0]} x {L_red.shape[1]}"
        )

        symmetry_error = np.max(
            np.abs(
                (L_red - L_red.T).data
            )
        ) if (L_red - L_red.T).nnz > 0 else 0.0

        print(
            f"Maximum symmetry error: "
            f"{symmetry_error:.3e}"
        )

        print(
            "Reduced matrix check: "
            + (
                "PASSED"
                if symmetry_error < 1e-12
                else "FAILED"
            )
        )

    # Sparse LU
    lu = spla.splu(L_red)

    
    # The reduced Laplacian is positive definite, so its
    # determinant is positive. The logarithm is evaluated
    # from the absolute values of the LU diagonal factors.

    diag_U = lu.U.diagonal()

    log_det = np.sum(
        np.log(np.abs(diag_U))
    )

    log_pdet = (
        np.log(float(n))
        + log_det
    )

    return float(log_pdet)


# ============================================================
# 7. Construct Kanzaki force vector
# ============================================================

def build_kanzaki_force(
    L: int,
    vac_idx: int,
    old_to_new,
    f0: float
):
    """
    Construct the external Kanzaki-force vector used by
    the supplied GMRF-VAC formulation.

    Each of the four nearest neighbours receives a force
    of magnitude f0 directed inward toward the vacancy.
    """

    N = L * L

    i_v = vac_idx // L
    j_v = vac_idx % L

    f_ext = np.zeros(
        2 * (N - 1),
        dtype=float
    )

    nbr_forces = [

        (
            ((i_v - 1) % L, j_v),
            np.array([0.0, -1.0])
        ),

        (
            ((i_v + 1) % L, j_v),
            np.array([0.0, 1.0])
        ),

        (
            (i_v, (j_v - 1) % L),
            np.array([1.0, 0.0])
        ),

        (
            (i_v, (j_v + 1) % L),
            np.array([-1.0, 0.0])
        )
    ]

    for (r, c), force_dir in nbr_forces:

        old_idx = r * L + c

        new_idx = old_to_new[old_idx]

        f_ext[
            2 * new_idx:
            2 * new_idx + 2
        ] = f0 * force_dir

    return f_ext


# ============================================================
# 8. Compute relaxation contribution
# ============================================================

def compute_relaxation(
    L_def,
    f_ext,
    beta=1.0,
    kappa=1.0,
    epsilon=1e-10
):
    """
    Solve

        (Q_def + epsilon I) mu = beta * f_ext

    with

        Q_def = beta*kappa*kron(L_def, I_2)

    and compute

        Delta F_relax
        =
        -1/2 f_ext^T mu.
    """

    N_def = L_def.shape[0]

    Q_def = (
        beta
        * kappa
        * sp.kron(
            L_def,
            sp.eye(2),
            format="csr"
        )
    )

    Q_reg = (
        Q_def
        + epsilon
        * sp.eye(
            2 * N_def,
            format="csr"
        )
    )

    rhs = beta * f_ext

    mu = spla.spsolve(
        Q_reg,
        rhs
    )

    f_ext_T_mu = float(
        np.dot(
            f_ext,
            mu
        )
    )

    delta_F_relax = (
        -0.5
        * f_ext_T_mu
    )

    # Maximum displacement magnitude
    mu_xy = mu.reshape(
        (-1, 2)
    )

    displacement_magnitudes = np.linalg.norm(
        mu_xy,
        axis=1
    )

    max_displacement = float(
        np.max(
            displacement_magnitudes
        )
    )

    return (
        f_ext_T_mu,
        max_displacement,
        delta_F_relax
    )


# ============================================================
# 9. Single complete GMRF-VAC calculation
# ============================================================

def calculate_one_L(
    L: int,
    beta=1.0,
    kappa=1.0,
    f0=0.05,
    epsilon=1e-10,
    verbose=True
):
    """
    Complete calculation for one lattice size.
    """

    N = L * L

    # --------------------------------------------------------
    # Vacancy location
    # --------------------------------------------------------

    i_v = L // 2
    j_v = L // 2

    vac_idx = i_v * L + j_v

    if verbose:
        print()
        print("=" * 90)
        print(
            f"Processing L = {L}, N = {N}"
        )
        print("=" * 90)

    if verbose:
        print(
            f"Vacancy coordinate: "
            f"({i_v}, {j_v})"
        )

        print(
            f"Vacancy original index: "
            f"{vac_idx}"
        )

    # --------------------------------------------------------
    # Perfect lattice
    # --------------------------------------------------------

    L_ideal = build_perfect_laplacian(L)

    log_pdet_ideal = compute_log_pdet(
        L_ideal,
        validate=verbose,
        name="Perfect Laplacian"
    )

    # --------------------------------------------------------
    # Defective lattice
    # --------------------------------------------------------

    L_def, old_to_new = build_defective_laplacian(
        L,
        vac_idx
    )

    log_pdet_def = compute_log_pdet(
        L_def,
        validate=verbose,
        name="Defective Laplacian"
    )

    # --------------------------------------------------------
    # Pseudo-determinant ratio
    # --------------------------------------------------------

    ln_pdet_ratio = (
        log_pdet_def
        - log_pdet_ideal
    )

    pdet_ratio = np.exp(
        ln_pdet_ratio
    )

    # Since beta*kappa = 1 in the
    # default formulation:
    delta_F_det = (
         -(1.0 / beta) * np.log(beta * kappa)
         + (1.0 / beta) * ln_pdet_ratio
    )

    # --------------------------------------------------------
    # Kanzaki force
    # --------------------------------------------------------

    f_ext = build_kanzaki_force(
        L=L,
        vac_idx=vac_idx,
        old_to_new=old_to_new,
        f0=f0
    )

    # --------------------------------------------------------
    # Relaxation
    # --------------------------------------------------------

    (
        f_ext_T_mu,
        max_displacement,
        delta_F_relax
    ) = compute_relaxation(
        L_def=L_def,
        f_ext=f_ext,
        beta=beta,
        kappa=kappa,
        epsilon=epsilon
    )

    # --------------------------------------------------------
    # Total free-energy contribution
    # --------------------------------------------------------

    delta_F_total = (
        delta_F_det
        + delta_F_relax
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if verbose:

        print()
        print(
            "GMRF-VAC numerical results"
        )
        print("-" * 60)

        print(
            f"ln[pdet(L_def)/pdet(L)] : "
            f"{ln_pdet_ratio:.9f}"
        )

        print(
            f"pdet ratio               : "
            f"{pdet_ratio:.9e}"
        )

        print(
            f"f_ext^T mu               : "
            f"{f_ext_T_mu:.9f}"
        )

        print(
            f"Maximum displacement     : "
            f"{max_displacement:.9e}"
        )

        print(
            f"Delta F determinant      : "
            f"{delta_F_det:.9f}"
        )

        print(
            f"Delta F relaxation      : "
            f"{delta_F_relax:.9f}"
        )

        print(
            f"Total normalized Delta F: "
            f"{delta_F_total:.9f}"
        )

    return {
        "L": L,
        "N": N,
        "vacancy_i": i_v,
        "vacancy_j": j_v,
        "vacancy_index": vac_idx,
        "log_pdet_ideal": log_pdet_ideal,
        "log_pdet_defective": log_pdet_def,
        "C_L": ln_pdet_ratio,
        "pdet_ratio": pdet_ratio,
        "f_ext_T_mu": f_ext_T_mu,
        "max_displacement": max_displacement,
        "delta_F_det": delta_F_det,
        "delta_F_relax": delta_F_relax,
        "delta_F_total": delta_F_total
    }


# ============================================================
# 10. Finite-size L^-2 fit
# ============================================================

def fit_L_minus_2(results, minimum_L=None):
    """
    Fit

        C_L = C_inf + A/L^2

    by ordinary least squares.
    """

    if minimum_L is None:
        selected = results
    else:
        selected = [
            r for r in results
            if r["L"] >= minimum_L
        ]

    L_values = np.array(
        [r["L"] for r in selected],
        dtype=float
    )

    C_values = np.array(
        [r["C_L"] for r in selected],
        dtype=float
    )

    x = 1.0 / L_values**2
    y = C_values

    A, C_inf = np.polyfit(
        x,
        y,
        1
    )

    y_fit = (
        C_inf
        + A * x
    )

    residuals = (
        y - y_fit
    )

    ss_res = np.sum(
        residuals**2
    )

    ss_tot = np.sum(
        (y - np.mean(y))**2
    )

    if ss_tot > 0:
        R2 = (
            1.0
            - ss_res / ss_tot
        )
    else:
        R2 = np.nan

    return {
        "minimum_L": (
            min(r["L"] for r in selected)
        ),
        "number_of_points": len(selected),
        "C_inf": float(C_inf),
        "A": float(A),
        "R2": float(R2)
    }


# ============================================================
# 11. Save authoritative Table 2
# ============================================================

def save_results_csv(
    results,
    filename="authoritative_table2_data.csv"
):
    """
    Save complete numerical results.
    """

    fieldnames = [
        "L",
        "N",
        "vacancy_i",
        "vacancy_j",
        "vacancy_index",
        "log_pdet_ideal",
        "log_pdet_defective",
        "C_L",
        "pdet_ratio",
        "f_ext_T_mu",
        "max_displacement",
        "delta_F_det",
        "delta_F_relax",
        "delta_F_total"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print()
    print(
        f"Authoritative numerical data saved to:"
    )
    print(
        f"  {Path(filename).resolve()}"
    )


# ============================================================
# 12. Print manuscript-ready Table 2
# ============================================================

def print_table2(results):

    print()
    print("=" * 115)
    print("AUTHORITATIVE TABLE 2 — GMRF-VAC FINITE-SIZE RESULTS")
    print("=" * 115)

    print(
        f"{'L':>5}"
        f"{'N':>8}"
        f"{'C_L':>14}"
        f"{'f_ext^T mu':>16}"
        f"{'Max |mu|':>16}"
        f"{'Delta F_relax':>18}"
        f"{'Delta F_total':>18}"
    )

    print("-" * 115)

    for r in results:

        print(
            f"{r['L']:5d}"
            f"{r['N']:8d}"
            f"{r['C_L']:14.9f}"
            f"{r['f_ext_T_mu']:16.9f}"
            f"{r['max_displacement']:16.9e}"
            f"{r['delta_F_relax']:18.9f}"
            f"{r['delta_F_total']:18.9f}"
        )

    print("=" * 115)


# ============================================================
# 13. Main finite-size study
# ============================================================

def run_authoritative_table2():

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    lattice_sizes = [
        8,
        12,
        16,
        20,
        24,
        32,
        48,
        64
    ]

    beta = 1.0
    kappa = 1.0
    f0 = 0.05
    epsilon = 1e-10

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("GMRF-VAC AUTHORITATIVE TABLE 2 CALCULATION")
    print("=" * 90)

    print()
    print("Parameters")
    print("-" * 60)
    print(f"beta      = {beta}")
    print(f"kappa     = {kappa}")
    print(f"f0        = {f0}")
    print(f"epsilon   = {epsilon}")
    print(
        "L values  = "
        + str(lattice_sizes)
    )

    # --------------------------------------------------------
    # Run all sizes
    # --------------------------------------------------------

    results = []

    for L in lattice_sizes:

        result = calculate_one_L(
            L=L,
            beta=beta,
            kappa=kappa,
            f0=f0,
            epsilon=epsilon,
            verbose=True
        )

        results.append(result)

    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print_table2(results)

    # --------------------------------------------------------
    # Finite-size extrapolations
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("FINITE-SIZE EXTRAPOLATION")
    print("=" * 90)

    print()
    print(
        "Model:"
    )

    print(
        "    C_L = C_inf + A/L^2"
    )

    fit_ranges = [
        ("All L", None),
        ("L >= 16", 16),
        ("L >= 24", 24)
    ]

    print()
    print(
        f"{'Fit range':<15}"
        f"{'C_inf':>18}"
        f"{'A':>18}"
        f"{'R^2':>15}"
    )

    print("-" * 70)

    fit_results = []

    for label, minimum_L in fit_ranges:

        fit = fit_L_minus_2(
            results,
            minimum_L=minimum_L
        )

        fit_results.append(
            (label, fit)
        )

        print(
            f"{label:<15}"
            f"{fit['C_inf']:18.9f}"
            f"{fit['A']:18.9f}"
            f"{fit['R2']:15.9f}"
        )

    # --------------------------------------------------------
    # Recommended estimate
    # --------------------------------------------------------

    fit_large = fit_L_minus_2(
        results,
        minimum_L=24
    )

    print()
    print(
        "Recommended large-L extrapolation:"
    )

    print(
        f"    C_inf = "
        f"{fit_large['C_inf']:.9f}"
    )

    print(
        f"    A     = "
        f"{fit_large['A']:.9f}"
    )

    print(
        f"    R^2   = "
        f"{fit_large['R2']:.9f}"
    )

    # --------------------------------------------------------
    # Save numerical data
    # --------------------------------------------------------

    save_results_csv(
        results,
        filename="authoritative_table2_data.csv"
    )

    # --------------------------------------------------------
    # Save extrapolation data
    # --------------------------------------------------------

    with open(
        "finite_size_extrapolation.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "fit_range",
            "minimum_L",
            "number_of_points",
            "C_inf",
            "A",
            "R2"
        ])

        for label, fit in fit_results:

            writer.writerow([
                label,
                fit["minimum_L"],
                fit["number_of_points"],
                fit["C_inf"],
                fit["A"],
                fit["R2"]
            ])

    print()
    print(
        "Finite-size extrapolation data saved to:"
    )

    print(
        f"  {Path('finite_size_extrapolation.csv').resolve()}"
    )

    # --------------------------------------------------------
    # Final diagnostic statement
    # --------------------------------------------------------

    C64 = results[-1]["C_L"]

    print()
    print("=" * 90)
    print("FINAL NUMERICAL DIAGNOSTIC")
    print("=" * 90)

    print(
        f"C_64 = {C64:.9f}"
    )

    print(
        f"C_64 rounded to four decimals = "
        f"{C64:.4f}"
    )

    print()
    print(
        "The value C_64 is the directly calculated finite-size"
    )

    print(
        "result for L = 64."
    )

    print()
    print(
        "The quantity C_inf is an extrapolated thermodynamic-"
    )

    print(
        "limit estimate and must not be identified with C_64."
    )

    print()
    print(
        "No manuscript values are hard-coded into this program."
    )

    print(
        "All reported values are generated directly from the"
    )

    print(
        "GMRF-VAC calculation."
    )

    print()
    print(
        "AUTHORITATIVE TABLE 2 CALCULATION COMPLETED"
    )

    print("=" * 90)

    return results, fit_results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_authoritative_table2()
