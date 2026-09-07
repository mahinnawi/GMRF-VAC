
"""GMRF-VAC: Sparse LU Performance Benchmark

Benchmarks sparse LU factorization of the reduced perfect and
true-vacancy Laplacians for L = 16, 32, 64, 128, and 256.

The script reports minimum and median factorization times,
fits empirical power laws t = C N^alpha, and generates the
normalized O(N^(3/2)) reference-slope figure used for the
computational-scaling analysis.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time
import gc
import csv


# ============================================================
# Configuration
# ============================================================

LATTICE_SIZES = [16, 32, 64, 128, 256]

# Number of independent LU timing repetitions for each matrix.
N_REPEATS = 5

# Use the minimum time as the primary benchmark value.
# Median is also retained for diagnostic/reproducibility purposes.
USE_MINIMUM_FOR_SCALING = True

# Output files
FIGURE_FILE = "performance_scaling.pdf"
CSV_FILE = "performance_scaling_data.csv"


# ============================================================
# 1. Build the perfect periodic square-lattice Laplacian
# ============================================================

def build_perfect_laplacian(L):
    """
    Build the N x N graph Laplacian of an L x L periodic
    square lattice using 0-based indexing.

    N = L^2

    Every lattice site has four nearest neighbours.
    The resulting matrix is symmetric, sparse, and has
    one translational zero mode.
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

            # Degree of a perfect square-lattice node
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
    ).tocsc()

    return Lmat


# ============================================================
# 2. Build the true-vacancy defective Laplacian
# ============================================================

def build_defective_laplacian(L, vacancy=None):
    """
    Build the (N-1) x (N-1) graph Laplacian after physically
    deleting one vacancy node and all bonds incident to it.

    The remaining nodes are re-indexed consecutively.

    This is NOT a principal submatrix of the perfect Laplacian.

    Consequently, the four nearest neighbours of the vacancy
    have degree 3 rather than degree 4.

    Returns
    -------
    Ldef : scipy.sparse.csc_matrix
        Defective Laplacian of dimension (N-1) x (N-1).

    vacancy : int
        Original 0-based index of the deleted node.
    """

    N = L * L

    if vacancy is None:

        # Geometric centre for even L.
        #
        # For L = 32:
        # coordinate = (16, 16)
        # index      = 16*32 + 16 = 528
        #
        i0 = L // 2
        j0 = L // 2
        vacancy = i0 * L + j0

    vacancy_i = vacancy // L
    vacancy_j = vacancy % L

    # --------------------------------------------------------
    # Map original node indices to new defective indices
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Construct the defective graph from scratch
    # --------------------------------------------------------

    for i in range(L):
        for j in range(L):

            old_p = idx(i, j)

            # The vacancy itself is removed
            if old_p == vacancy:
                continue

            p = old_to_new[old_p]

            # Original periodic nearest neighbours
            neighbours = [
                ((i - 1) % L, j),      # up
                ((i + 1) % L, j),      # down
                (i, (j - 1) % L),      # left
                (i, (j + 1) % L)       # right
            ]

            # Remove the vacancy and its incident bonds
            valid_neighbours = []

            for ni, nj in neighbours:

                old_q = idx(ni, nj)

                if old_q != vacancy:
                    valid_neighbours.append(old_q)

            # Physical degree after vacancy deletion
            degree = len(valid_neighbours)

            rows.append(p)
            cols.append(p)
            data.append(float(degree))

            # Off-diagonal entries
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
# 3. Reduced Laplacian
# ============================================================

def reduced_laplacian(Lmat):
    """
    Construct a reduced Laplacian by deleting the first row
    and first column.

    For a connected graph Laplacian with n vertices,

        pdet(L) = n * det(L_red)

    where L_red is any Laplacian cofactor.
    """

    return Lmat[1:, 1:].tocsc()


# ============================================================
# 4. Structural validation of a Laplacian
# ============================================================

def validate_laplacian(
    Lmat,
    expected_degree_counts,
    matrix_name
):
    """
    Validate the structural properties required for the
    perfect and defective graph Laplacians.

    Checks:
        1. Square matrix
        2. Symmetry
        3. Zero row sums
        4. Expected diagonal degrees
    """

    print()
    print(f"Validation: {matrix_name}")
    print("-" * 60)

    # --------------------------------------------------------
    # Square matrix
    # --------------------------------------------------------

    if Lmat.shape[0] != Lmat.shape[1]:
        raise ValueError(
            f"{matrix_name} is not square: {Lmat.shape}"
        )

    # --------------------------------------------------------
    # Symmetry
    # --------------------------------------------------------

    symmetry_difference = Lmat - Lmat.T

    if symmetry_difference.nnz > 0:
        symmetry_error = np.max(
            np.abs(symmetry_difference.data)
        )
    else:
        symmetry_error = 0.0

    print(
        f"Maximum symmetry error: "
        f"{symmetry_error:.3e}"
    )

    if symmetry_error > 1e-14:
        raise ValueError(
            f"{matrix_name} is not symmetric."
        )

    # --------------------------------------------------------
    # Zero row sums
    # --------------------------------------------------------

    row_sums = np.asarray(
        Lmat.sum(axis=1)
    ).ravel()

    row_sum_error = np.max(
        np.abs(row_sums)
    )

    print(
        f"Maximum row-sum error:  "
        f"{row_sum_error:.3e}"
    )

    if row_sum_error > 1e-12:
        raise ValueError(
            f"{matrix_name} does not have zero row sums."
        )

    # --------------------------------------------------------
    # Degree structure
    # --------------------------------------------------------

    degrees = Lmat.diagonal()

    print(
        f"Degree range:             "
        f"{degrees.min():.0f} - {degrees.max():.0f}"
    )

    for degree, expected_count in expected_degree_counts.items():

        actual_count = np.count_nonzero(
            np.isclose(degrees, degree)
        )

        print(
            f"Degree {degree}: "
            f"{actual_count} nodes "
            f"(expected {expected_count})"
        )

        if actual_count != expected_count:
            raise ValueError(
                f"{matrix_name}: unexpected number "
                f"of degree-{degree} nodes."
            )

    print("Validation: PASSED")


# ============================================================
# 5. Check connectivity of the graph
# ============================================================

def validate_connectivity(Lmat, matrix_name):
    """
    Verify that the graph represented by the Laplacian
    is connected.

    For a connected graph Laplacian there is exactly one
    zero eigenvalue.

    For the defective graph this check is performed on the
    full defective Laplacian.
    """

    print()
    print(f"Connectivity check: {matrix_name}")
    print("-" * 60)

    # Convert Laplacian to an adjacency-like matrix.
    #
    # Off-diagonal negative entries correspond to edges.
    adjacency = Lmat.copy().tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    # Use scipy's graph connectivity routine.
    from scipy.sparse.csgraph import connected_components

    n_components, labels = connected_components(
        adjacency,
        directed=False
    )

    print(
        f"Number of connected components: "
        f"{n_components}"
    )

    if n_components != 1:
        raise ValueError(
            f"{matrix_name} is not connected."
        )

    print("Connectivity check: PASSED")


# ============================================================
# 6. Validate the reduced Laplacian
# ============================================================

def validate_reduced_laplacian(Lred, matrix_name):
    """
    Verify that the reduced Laplacian is symmetric and
    suitable for sparse LU factorization.
    """

    print()
    print(f"Reduced matrix check: {matrix_name}")
    print("-" * 60)

    # Symmetry
    diff = Lred - Lred.T

    if diff.nnz > 0:
        symmetry_error = np.max(
            np.abs(diff.data)
        )
    else:
        symmetry_error = 0.0

    print(
        f"Maximum symmetry error: "
        f"{symmetry_error:.3e}"
    )

    if symmetry_error > 1e-14:
        raise ValueError(
            f"{matrix_name} is not symmetric."
        )

    # Structural rank check through SuperLU
    #
    # We do this once outside the timing loop to ensure that
    # the matrix is nonsingular.
    lu = spla.splu(Lred)
    del lu
    gc.collect()

    print(
        f"Reduced matrix dimension: "
        f"{Lred.shape[0]} x {Lred.shape[1]}"
    )

    print("Reduced matrix check: PASSED")


# ============================================================
# 7. Benchmark sparse LU factorization
# ============================================================

def benchmark_splu(A, repeats=N_REPEATS):
    """
    Benchmark sparse LU factorization.

    The matrix A is already constructed and converted to CSC,
    so matrix construction and format conversion are excluded
    from the timing.

    Returns
    -------
    minimum_time : float
        Minimum factorization time.

    median_time : float
        Median factorization time.

    all_times : list
        Individual timing measurements.
    """

    times = []

    for repetition in range(repeats):

        # Explicit garbage collection occurs BEFORE timing,
        # so cleanup overhead is not included in the measured
        # LU factorization time.
        gc.collect()

        start_time = time.perf_counter()

        lu = spla.splu(A)

        end_time = time.perf_counter()

        elapsed = end_time - start_time

        times.append(elapsed)

        # Release the factorization before the next run.
        del lu

    minimum_time = min(times)
    median_time = float(np.median(times))

    return minimum_time, median_time, times


# ============================================================
# 8. Empirical power-law fit
# ============================================================

def fit_power_law(N_values, times):
    """
    Fit

        t = C * N^alpha

    using linear regression in log-log space.

    Returns
    -------
    alpha : float
        Empirical scaling exponent.

    C : float
        Fitted prefactor.

    r_squared : float
        Coefficient of determination.
    """

    N_values = np.asarray(N_values, dtype=float)
    times = np.asarray(times, dtype=float)

    log_N = np.log(N_values)
    log_t = np.log(times)

    alpha, log_C = np.polyfit(
        log_N,
        log_t,
        1
    )

    C = np.exp(log_C)

    # Predicted log-times
    predicted_log_t = (
        log_C + alpha * log_N
    )

    ss_res = np.sum(
        (log_t - predicted_log_t) ** 2
    )

    ss_tot = np.sum(
        (log_t - np.mean(log_t)) ** 2
    )

    if ss_tot > 0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = np.nan

    return alpha, C, r_squared


# ============================================================
# 9. Save benchmark data
# ============================================================

def save_results(
    N_values,
    perfect_min,
    perfect_median,
    defective_min,
    defective_median,
    total_min,
    total_median
):
    """
    Save benchmark results to a CSV file.
    """

    with open(
        CSV_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "L",
            "N",
            "Perfect_LU_min_s",
            "Perfect_LU_median_s",
            "Defective_LU_min_s",
            "Defective_LU_median_s",
            "Total_LU_min_s",
            "Total_LU_median_s"
        ])

        for i, N in enumerate(N_values):

            L = int(np.sqrt(N))

            writer.writerow([
                L,
                N,
                f"{perfect_min[i]:.9f}",
                f"{perfect_median[i]:.9f}",
                f"{defective_min[i]:.9f}",
                f"{defective_median[i]:.9f}",
                f"{total_min[i]:.9f}",
                f"{total_median[i]:.9f}"
            ])

    print()
    print(
        f"Benchmark data saved to: {CSV_FILE}"
    )


# ============================================================
# 10. Plot computational scaling
# ============================================================

def plot_scaling(
    N_values,
    perfect_times,
    defective_times,
    total_times
):
    """
    Plot measured LU factorization scaling.

    The O(N^(3/2)) curve is only a normalized reference
    slope. It is NOT fitted to the data and is NOT a
    prediction for the particular SuperLU configuration.
    """

    plt.figure(figsize=(8, 6))

    # --------------------------------------------------------
    # Measured data
    # --------------------------------------------------------

    plt.loglog(
        N_values,
        perfect_times,
        marker='o',
        linestyle='-',
        label='Perfect reduced Laplacian'
    )

    plt.loglog(
        N_values,
        defective_times,
        marker='s',
        linestyle='-',
        label='Defective reduced Laplacian'
    )

    plt.loglog(
        N_values,
        total_times,
        marker='^',
        linestyle='-',
        label='Total LU factorization'
    )

    # --------------------------------------------------------
    # O(N^(3/2)) reference slope
    # --------------------------------------------------------

    N0 = N_values[0]
    t0 = total_times[0]

    reference_times = [
        t0 * (N / N0) ** 1.5
        for N in N_values
    ]

    plt.loglog(
        N_values,
        reference_times,
        linestyle='--',
        label=r'Reference slope $O(N^{3/2})$'
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.xlabel(
        'System Size (Total Nodes $N$)',
        fontsize=12
    )

    plt.ylabel(
        'Wall-clock LU Factorization Time (s)',
        fontsize=12
    )

    plt.title(
        'Computational Scaling of Sparse LU Factorization',
        fontsize=14
    )

    plt.legend(fontsize=10)

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_FILE,
        bbox_inches='tight'
    )

    print()
    print(
        f"Scaling figure saved to: {FIGURE_FILE}"
    )

    plt.show()


# ============================================================
# 11. Main benchmark
# ============================================================

def run_benchmarks():

    perfect_min_times = []
    perfect_median_times = []

    defective_min_times = []
    defective_median_times = []

    total_min_times = []
    total_median_times = []

    N_values = []

    print()
    print("=" * 90)
    print("GMRF-VAC SPARSE LU PERFORMANCE BENCHMARK")
    print("=" * 90)

    print()
    print(
        f"Timing repetitions per matrix: {N_REPEATS}"
    )

    print(
        "Primary benchmark statistic: minimum LU factorization time"
    )

    print(
        "Additional diagnostic statistic: median LU factorization time"
    )

    print()
    print(
        f"{'L':>6}"
        f"{'N':>10}"
        f"{'Perfect min (s)':>18}"
        f"{'Perfect med. (s)':>20}"
        f"{'Defective min (s)':>20}"
        f"{'Defective med. (s)':>22}"
        f"{'Total min (s)':>18}"
        f"{'Total med. (s)':>20}"
    )

    print("-" * 140)

    # ========================================================
    # Loop over lattice sizes
    # ========================================================

    for L in LATTICE_SIZES:

        N = L * L

        N_values.append(N)

        print()
        print("=" * 90)
        print(f"Processing L = {L}, N = {N}")
        print("=" * 90)

        # ====================================================
        # Perfect lattice
        # ====================================================

        L_perfect = build_perfect_laplacian(L)

        validate_laplacian(
            L_perfect,
            expected_degree_counts={
                4: N
            },
            matrix_name="Perfect Laplacian"
        )

        validate_connectivity(
            L_perfect,
            matrix_name="Perfect lattice"
        )

        L_perfect_red = reduced_laplacian(
            L_perfect
        )

        validate_reduced_laplacian(
            L_perfect_red,
            matrix_name="Perfect reduced Laplacian"
        )

        # ====================================================
        # Benchmark perfect reduced Laplacian
        # ====================================================

        (
            t_perfect_min,
            t_perfect_median,
            perfect_runs
        ) = benchmark_splu(
            L_perfect_red,
            repeats=N_REPEATS
        )

        # ====================================================
        # True-vacancy defective lattice
        # ====================================================

        L_defective, vacancy = (
            build_defective_laplacian(L)
        )

        vacancy_i = vacancy // L
        vacancy_j = vacancy % L

        print()
        print(
            f"Vacancy coordinate: "
            f"({vacancy_i}, {vacancy_j})"
        )

        print(
            f"Vacancy original index: "
            f"{vacancy}"
        )

        # Expected defective degree structure:
        #
        # 4 neighbours have degree 3.
        # Remaining N-5 nodes have degree 4.
        #
        validate_laplacian(
            L_defective,
            expected_degree_counts={
                3: 4,
                4: N - 5
            },
            matrix_name="Defective Laplacian"
        )

        validate_connectivity(
            L_defective,
            matrix_name="Defective lattice"
        )

        L_defective_red = reduced_laplacian(
            L_defective
        )

        validate_reduced_laplacian(
            L_defective_red,
            matrix_name="Defective reduced Laplacian"
        )

        # ====================================================
        # Benchmark defective reduced Laplacian
        # ====================================================

        (
            t_defective_min,
            t_defective_median,
            defective_runs
        ) = benchmark_splu(
            L_defective_red,
            repeats=N_REPEATS
        )

        # ====================================================
        # Total factorization time
        # ====================================================

        t_total_min = (
            t_perfect_min
            +
            t_defective_min
        )

        t_total_median = (
            t_perfect_median
            +
            t_defective_median
        )

        # ====================================================
        # Store results
        # ====================================================

        perfect_min_times.append(
            t_perfect_min
        )

        perfect_median_times.append(
            t_perfect_median
        )

        defective_min_times.append(
            t_defective_min
        )

        defective_median_times.append(
            t_defective_median
        )

        total_min_times.append(
            t_total_min
        )

        total_median_times.append(
            t_total_median
        )

        # ====================================================
        # Print individual runs for transparency
        # ====================================================

        print()
        print("Perfect LU individual runs:")
        print(
            "  "
            +
            ", ".join(
                f"{t:.6f}"
                for t in perfect_runs
            )
            +
            " s"
        )

        print(
            f"  Minimum = {t_perfect_min:.6f} s"
        )

        print(
            f"  Median  = {t_perfect_median:.6f} s"
        )

        print()
        print("Defective LU individual runs:")
        print(
            "  "
            +
            ", ".join(
                f"{t:.6f}"
                for t in defective_runs
            )
            +
            " s"
        )

        print(
            f"  Minimum = {t_defective_min:.6f} s"
        )

        print(
            f"  Median  = {t_defective_median:.6f} s"
        )

        # ====================================================
        # Summary row
        # ====================================================

        print()
        print(
            f"{L:6d}"
            f"{N:10d}"
            f"{t_perfect_min:18.6f}"
            f"{t_perfect_median:20.6f}"
            f"{t_defective_min:20.6f}"
            f"{t_defective_median:22.6f}"
            f"{t_total_min:18.6f}"
            f"{t_total_median:20.6f}"
        )

        # ====================================================
        # Release large matrices before next size
        # ====================================================

        del L_perfect
        del L_perfect_red
        del L_defective
        del L_defective_red

        gc.collect()

    # ========================================================
    # Empirical scaling analysis
    # ========================================================

    N_values_array = np.asarray(
        N_values,
        dtype=float
    )

    # --------------------------------------------------------
    # Choose primary statistic
    # --------------------------------------------------------

    if USE_MINIMUM_FOR_SCALING:

        scaling_perfect = perfect_min_times
        scaling_defective = defective_min_times
        scaling_total = total_min_times

        scaling_label = "minimum"

    else:

        scaling_perfect = perfect_median_times
        scaling_defective = defective_median_times
        scaling_total = total_median_times

        scaling_label = "median"

    # --------------------------------------------------------
    # Fit power laws
    # --------------------------------------------------------

    (
        alpha_perfect,
        C_perfect,
        R2_perfect
    ) = fit_power_law(
        N_values_array,
        scaling_perfect
    )

    (
        alpha_defective,
        C_defective,
        R2_defective
    ) = fit_power_law(
        N_values_array,
        scaling_defective
    )

    (
        alpha_total,
        C_total,
        R2_total
    ) = fit_power_law(
        N_values_array,
        scaling_total
    )

    # ========================================================
    # Print final scaling results
    # ========================================================

    print()
    print("=" * 90)
    print("EMPIRICAL SCALING RESULTS")
    print("=" * 90)

    print()
    print(
        f"Primary statistic used for scaling: "
        f"{scaling_label}"
    )

    print()
    print(
        "Power-law model:"
    )

    print(
        "    t = C * N^alpha"
    )

    print()
    print(
        f"Perfect lattice:"
    )

    print(
        f"    alpha = {alpha_perfect:.6f}"
    )

    print(
        f"    C     = {C_perfect:.6e}"
    )

    print(
        f"    R^2   = {R2_perfect:.6f}"
    )

    print()
    print(
        f"Defective lattice:"
    )

    print(
        f"    alpha = {alpha_defective:.6f}"
    )

    print(
        f"    C     = {C_defective:.6e}"
    )

    print(
        f"    R^2   = {R2_defective:.6f}"
    )

    print()
    print(
        f"Total sparse LU factorization:"
    )

    print(
        f"    alpha = {alpha_total:.6f}"
    )

    print(
        f"    C     = {C_total:.6e}"
    )

    print(
        f"    R^2   = {R2_total:.6f}"
    )

    # ========================================================
    # Save data
    # ========================================================

    save_results(
        N_values,
        perfect_min_times,
        perfect_median_times,
        defective_min_times,
        defective_median_times,
        total_min_times,
        total_median_times
    )

    # ========================================================
    # Plot
    # ========================================================

    plot_scaling(
        N_values,
        scaling_perfect,
        scaling_defective,
        scaling_total
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 90)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 90)

    print()
    print(
        "The benchmark measures only the sparse LU factorization"
    )

    print(
        "stage of the pseudo-determinant calculation."
    )

    print(
        "Matrix construction, reduced-matrix construction,"
    )

    print(
        "and other GMRF-VAC operations are excluded from the"
    )

    print(
        "reported LU timing."
    )

    print()
    print(
        "The O(N^(3/2)) curve in the figure is a normalized"
    )

    print(
        "reference slope, not a fitted prediction or an"
    )

    print(
        "asymptotic complexity claim for the particular"
    )

    print(
        "SuperLU configuration used here."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_benchmarks()