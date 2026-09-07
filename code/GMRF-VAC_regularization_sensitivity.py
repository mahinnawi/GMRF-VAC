
"""
GMRF-VAC Regularization Sensitivity

Verifies the sensitivity of the regularized displacement solution
to the choice of epsilon for the L = 32 true-vacancy model.

The script evaluates

    (Q_def + epsilon I) mu_epsilon = beta f_ext

for epsilon = 1e-6, 1e-8, 1e-10, and 1e-12, using the
epsilon = 1e-12 solution as the reference.

The calculation reports the relative displacement difference,
f_ext^T mu, and the corresponding relaxation contribution.
"""

import csv
import gc

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ============================================================
# GMRF-VAC
# Regularization Sensitivity Verification
#
# Purpose:
#   Verify the sensitivity of the regularized displacement
#   solution to the choice of epsilon for the L = 32
#   true-vacancy model.
#
# The script evaluates:
#
#   (Q_def + epsilon I) mu_epsilon = beta f_ext
#
# for
#
#   epsilon = 1e-6, 1e-8, 1e-10, 1e-12
#
# using the epsilon = 1e-12 solution as the reference.
#
# It reports:
#   1. Relative displacement difference
#   2. f_ext^T mu
#   3. Relaxation contribution
#
# The calculation reproduces the manuscript values:
#
#   Maximum relative displacement difference = 2.02e-5
#   Maximum change in f_ext^T mu              = 1.06e-8
# ============================================================


# ============================================================
# 1. Build the true-vacancy defective Laplacian
# ============================================================

def build_defective_laplacian(L, vacancy=None):
    """
    Build the (N-1) x (N-1) true-vacancy graph Laplacian.

    One lattice node is physically deleted together with all
    incident bonds.

    The four nearest neighbours of the vacancy therefore have
    degree 3 instead of degree 4.

    Returns
    -------
    Ldef : scipy.sparse.csc_matrix
        Defective Laplacian.

    vacancy : int
        Original 0-based vacancy index.
    """

    N = L * L

    if vacancy is None:
        i0 = L // 2
        j0 = L // 2
        vacancy = i0 * L + j0

    old_to_new = -np.ones(N, dtype=int)

    new_index = 0

    for old_index in range(N):
        if old_index != vacancy:
            old_to_new[old_index] = new_index
            new_index += 1

    N_def = N - 1

    rows = []
    cols = []
    data = []

    def idx(i, j):
        return i * L + j

    for i in range(L):
        for j in range(L):

            old_p = idx(i, j)

            if old_p == vacancy:
                continue

            p = old_to_new[old_p]

            neighbours = [
                ((i - 1) % L, j),      # up
                ((i + 1) % L, j),      # down
                (i, (j - 1) % L),      # left
                (i, (j + 1) % L)       # right
            ]

            valid_neighbours = []

            for ni, nj in neighbours:
                old_q = idx(ni, nj)

                if old_q != vacancy:
                    valid_neighbours.append(old_q)

            degree = len(valid_neighbours)

            # Diagonal degree
            rows.append(p)
            cols.append(p)
            data.append(float(degree))

            # Off-diagonal bonds
            for old_q in valid_neighbours:

                q = old_to_new[old_q]

                rows.append(p)
                cols.append(q)
                data.append(-1.0)

    Ldef = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(N_def, N_def)
    ).tocsc()

    return Ldef, vacancy


# ============================================================
# 2. Build the prescribed Kanzaki force vector
# ============================================================

def build_kanzaki_force_vector(L, vacancy, f0=0.05):
    """
    Construct the 2(N-1)-component prescribed Kanzaki-force
    vector for the true-vacancy model.

    Ordering:
        [u0_x, u0_y, u1_x, u1_y, ...]

    Cartesian convention:
        x = j
        y = -i

    The four neighbours receive inward forces of magnitude f0.
    """

    N = L * L

    old_to_new = -np.ones(N, dtype=int)

    new_index = 0

    for old_index in range(N):

        if old_index != vacancy:

            old_to_new[old_index] = new_index
            new_index += 1

    vacancy_i = vacancy // L
    vacancy_j = vacancy % L

    neighbours = {
        "up": (
            (vacancy_i - 1) % L,
            vacancy_j,
            np.array([0.0, -1.0])
        ),
        "down": (
            (vacancy_i + 1) % L,
            vacancy_j,
            np.array([0.0, 1.0])
        ),
        "left": (
            vacancy_i,
            (vacancy_j - 1) % L,
            np.array([1.0, 0.0])
        ),
        "right": (
            vacancy_i,
            (vacancy_j + 1) % L,
            np.array([-1.0, 0.0])
        )
    }

    N_def = N - 1

    f_ext = np.zeros(
        2 * N_def,
        dtype=float
    )

    mapping_info = []

    for direction, (ni, nj, direction_vector) in neighbours.items():

        original_index = ni * L + nj
        defective_index = old_to_new[original_index]

        force_vector = f0 * direction_vector

        f_ext[
            2 * defective_index
        ] = force_vector[0]

        f_ext[
            2 * defective_index + 1
        ] = force_vector[1]

        mapping_info.append(
            (
                direction,
                (ni, nj),
                original_index,
                int(defective_index),
                force_vector[0],
                force_vector[1]
            )
        )

    return f_ext, mapping_info


# ============================================================
# 3. Regularization sensitivity test
# ============================================================

def run_regularization_sensitivity_test(
    L=32,
    beta=1.0,
    kappa=1.0,
    f0=0.05,
    eps_values=(1e-6, 1e-8, 1e-10, 1e-12),
    output_csv="regularization_sensitivity_L32.csv"
):
    """
    Evaluate regularization sensitivity for the L = 32
    true-vacancy system.

    For each epsilon, solve

        (Q_def + epsilon I) mu_epsilon = beta f_ext

    and compare the solution with the smallest-epsilon
    reference solution.
    """

    print()
    print("=" * 90)
    print("GMRF-VAC REGULARIZATION SENSITIVITY VERIFICATION")
    print("=" * 90)

    N = L * L

    vacancy_i = L // 2
    vacancy_j = L // 2
    vacancy = vacancy_i * L + vacancy_j

    print(f"L = {L}, N = {N}")
    print(
        f"Vacancy coordinate = "
        f"({vacancy_i}, {vacancy_j})"
    )
    print(
        f"Vacancy original index = "
        f"{vacancy}"
    )

    # --------------------------------------------------------
    # Defective Laplacian
    # --------------------------------------------------------

    L_def, _ = build_defective_laplacian(
        L,
        vacancy=vacancy
    )

    # --------------------------------------------------------
    # Kanzaki force vector
    # --------------------------------------------------------

    f_ext, mapping_info = build_kanzaki_force_vector(
        L,
        vacancy,
        f0=f0
    )

    # --------------------------------------------------------
    # Verify zero total force
    # --------------------------------------------------------

    total_force = np.array([
        np.sum(f_ext[0::2]),
        np.sum(f_ext[1::2])
    ])

    total_force_norm = np.linalg.norm(
        total_force
    )

    print()
    print(
        "Total force vector = "
        f"[{total_force[0]:.6e}, "
        f"{total_force[1]:.6e}]"
    )

    print(
        f"Total-force norm = "
        f"{total_force_norm:.6e}"
    )

    if total_force_norm > 1e-14:

        raise ValueError(
            "The prescribed Kanzaki force vector does not "
            "satisfy the zero-total-force condition."
        )

    # --------------------------------------------------------
    # Print force mapping
    # --------------------------------------------------------

    print()
    print("Kanzaki-force neighbour mapping")
    print("-" * 90)

    print(
        f"{'Direction':<10}"
        f"{'Original index':>18}"
        f"{'Defective index':>18}"
        f"{'Fx':>16}"
        f"{'Fy':>16}"
    )

    for (
        direction,
        coordinate,
        original_index,
        defective_index,
        fx,
        fy
    ) in mapping_info:

        print(
            f"{direction:<10}"
            f"{original_index:>18d}"
            f"{defective_index:>18d}"
            f"{fx:>16.6e}"
            f"{fy:>16.6e}"
        )

    # --------------------------------------------------------
    # Two-dimensional defective precision matrix
    # --------------------------------------------------------

    Q_def = (
        beta
        * kappa
        * sp.kron(
            L_def,
            sp.eye(2, format="csc"),
            format="csc"
        )
    )

    rhs = beta * f_ext

    identity = sp.eye(
        Q_def.shape[0],
        format="csc"
    )

    # --------------------------------------------------------
    # Solve all regularized systems
    # --------------------------------------------------------

    mu_solutions = {}

    for eps in eps_values:

        A_reg = (
            Q_def
            + eps * identity
        )

        mu_solutions[eps] = spla.spsolve(
            A_reg,
            rhs
        )


        del A_reg
        gc.collect()

    # Smallest epsilon is the reference
    eps_ref = min(eps_values)
    mu_ref = mu_solutions[eps_ref]

    norm_mu_ref = np.linalg.norm(
        mu_ref
    )

    if norm_mu_ref == 0.0:

        raise ValueError(
            "Reference displacement norm is zero."
        )

    # --------------------------------------------------------
    # Compute diagnostics
    # --------------------------------------------------------

    results = []

    print()
    print(
        f"{'epsilon':<14}"
        f"{'relative mu difference':>28}"
        f"{'f_ext^T mu':>20}"
        f"{'DeltaF_relax':>20}"
    )

    print("-" * 90)

    for eps in eps_values:

        mu_eps = mu_solutions[eps]

        relative_difference = (
            np.linalg.norm(
                mu_eps - mu_ref
            )
            / norm_mu_ref
        )

        fmu = float(
            f_ext @ mu_eps
        )

        delta_f_relax = (
            -0.5 * fmu
        )

        results.append([
            L,
            N,
            eps,
            relative_difference,
            fmu,
            delta_f_relax
        ])

        print(
            f"{eps:<14.0e}"
            f"{relative_difference:>28.12e}"
            f"{fmu:>20.12e}"
            f"{delta_f_relax:>20.12e}"
        )

    # --------------------------------------------------------
    # Summary diagnostics
    # --------------------------------------------------------

    max_relative_difference = max(
        row[3] for row in results
    )

    # Use the explicitly identified reference solution,
    # rather than relying on the order of eps_values.
    reference_fmu = float(
        f_ext @ mu_ref
    )

    max_fmu_change = max(
        abs(row[4] - reference_fmu)
        for row in results
    )

    # Identify epsilon producing the maximum displacement
    # difference.
    max_relative_row = max(
        results,
        key=lambda row: row[3]
    )

    max_relative_epsilon = (
        max_relative_row[2]
    )

    print()
    print(
        "Maximum relative displacement difference = "
        f"{max_relative_difference:.12e}"
    )

    print(
        "Occurring at epsilon = "
        f"{max_relative_epsilon:.0e}"
    )

    print(
        "Maximum absolute change in f_ext^T mu = "
        f"{max_fmu_change:.12e}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        output_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "L",
            "N",
            "epsilon",
            "relative_mu_difference",
            "f_ext_T_mu",
            "DeltaF_relax"
        ])

        for row in results:

            writer.writerow([
                row[0],
                row[1],
                f"{row[2]:.12e}",
                f"{row[3]:.12e}",
                f"{row[4]:.12e}",
                f"{row[5]:.12e}"
            ])

    print()
    print(
        f"Diagnostic results saved to: "
        f"{output_csv}"
    )

    print("=" * 90)

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    run_regularization_sensitivity_test()