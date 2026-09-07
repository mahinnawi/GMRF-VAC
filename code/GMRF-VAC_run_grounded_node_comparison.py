
"""
GMRF-VAC Grounded-Node Comparison

Reproduces the grounded-node comparison reported in Section 4.7
of the GMRF-VAC manuscript.

The grounded model is constructed as the principal submatrix
obtained by deleting the vacancy row and column from the perfect
periodic square-lattice Laplacian, while retaining the original
degree entries of the four vacancy neighbours.

Default parameters:
    L = 32
    beta = 1.0
    kappa = 1.0
    f0 = 0.05
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_perfect_laplacian(L: int) -> sp.csr_matrix:
    """
    Constructs the graph Laplacian for an L x L periodic square lattice.

    Node numbering:
        idx = i * L + j

    with periodic nearest neighbours:
        Up, Down, Left, Right.

    The Laplacian is:
        L = D - A
    with diagonal degree equal to 4.
    """

    N = L * L

    row_indices = []
    col_indices = []
    data = []

    for i in range(L):
        for j in range(L):

            idx = i * L + j

            neighbors = [
                ((i - 1) % L) * L + j,        # Up
                ((i + 1) % L) * L + j,        # Down
                i * L + ((j - 1) % L),        # Left
                i * L + ((j + 1) % L)         # Right
            ]

            for nbr in neighbors:
                row_indices.append(idx)
                col_indices.append(nbr)
                data.append(-1.0)

    A = sp.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(N, N)
    )

    D = sp.diags(
        [4.0] * N,
        0,
        shape=(N, N),
        format="csr"
    )

    return D + A


def build_grounded_laplacian(
    L_ideal: sp.csr_matrix,
    vac_idx: int
):
    """
    Constructs the grounded-node / principal-submatrix approximation.

    The vacancy is represented by deleting the corresponding row
    and column from the PERFECT Laplacian.

    IMPORTANT:
    The diagonal entries of the four neighbours are NOT reduced.
    Therefore, this is NOT the true-vacancy Laplacian.

    Returns:
        L_grounded : (N-1) x (N-1) principal submatrix
        old_to_new : dictionary mapping original node indices
                     to grounded-system indices
    """

    N = L_ideal.shape[0]

    old_nodes = [i for i in range(N) if i != vac_idx]

    old_to_new = {
        old: new
        for new, old in enumerate(old_nodes)
    }

    # Principal submatrix:
    # delete vacancy row and vacancy column
    L_grounded = L_ideal[
        old_nodes, :
    ][:, old_nodes].tocsr()

    return L_grounded, old_to_new


def run_grounded_comparison(
    L: int = 32,
    beta: float = 1.0,
    kappa: float = 1.0,
    f0: float = 0.05
):
    """
    Reproduces the grounded-node comparison described in
    Section 4.7.

    The procedure is:

    1. Construct the perfect periodic L x L Laplacian.
    2. Delete the vacancy row and column WITHOUT changing
       neighbour diagonal degrees.
    3. Construct the same Kanzaki force field used in the
       true-vacancy calculation.
    4. Solve the positive-definite grounded system directly.
    5. Compute:
           f_ext^T mu^(p)
           Delta F_relax^(p)
           Delta F_det^(p) = -ln(N)
           Delta F^(p)
    """

    N = L * L

    # ----------------------------------------------------------
    # 1. Vacancy location
    # ----------------------------------------------------------

    i_v = L // 2
    j_v = L // 2

    vac_idx = i_v * L + j_v

    # ----------------------------------------------------------
    # 2. Construct perfect Laplacian
    # ----------------------------------------------------------

    L_ideal = build_perfect_laplacian(L)

    # ----------------------------------------------------------
    # 3. Construct grounded-node principal submatrix
    # ----------------------------------------------------------

    L_grounded, old_to_new = build_grounded_laplacian(
        L_ideal,
        vac_idx
    )

    # ----------------------------------------------------------
    # 4. Construct the same Kanzaki force vector
    #
    # Cartesian convention:
    #     x = j
    #     y = -i
    #
    # Therefore the forces point inward toward the vacancy.
    # ----------------------------------------------------------

    f_ext = np.zeros(2 * (N - 1))

    nbr_forces = [

        # Up neighbour: (i_v - 1, j_v)
        ((i_v - 1, j_v),
         np.array([0.0, -1.0])),

        # Down neighbour: (i_v + 1, j_v)
        ((i_v + 1, j_v),
         np.array([0.0, 1.0])),

        # Left neighbour: (i_v, j_v - 1)
        ((i_v, j_v - 1),
         np.array([1.0, 0.0])),

        # Right neighbour: (i_v, j_v + 1)
        ((i_v, j_v + 1),
         np.array([-1.0, 0.0]))
    ]

    print("=" * 60)
    print("GROUNDED-NODE COMPARISON")
    print("=" * 60)

    print(f"Lattice size L               : {L}")
    print(f"Total nodes N                : {N}")
    print(
        f"Vacancy index (0-based)      : "
        f"{vac_idx} at ({i_v}, {j_v})"
    )

    print("\nForce mapping:")

    for (r, c), force_dir in nbr_forces:

        old_idx = r * L + c

        new_idx = old_to_new[old_idx]

        force = f0 * force_dir

        f_ext[
            2 * new_idx:
            2 * new_idx + 2
        ] = force

        print(
            f"  Original node {old_idx:4d} "
            f"at ({r:2d}, {c:2d}) "
            f"-> grounded index {new_idx:4d}, "
            f"force = {force}"
        )

    # ----------------------------------------------------------
    # 5. Check force balance
    # ----------------------------------------------------------

    total_force = np.zeros(2)

    for n in range(N - 1):
        total_force += f_ext[2 * n: 2 * n + 2]

    print("\nTotal force vector           :", total_force)
    print(
        "Total-force norm             : "
        f"{np.linalg.norm(total_force):.12e}"
    )

    # ----------------------------------------------------------
    # 6. Construct grounded precision matrix
    #
    # Unlike the true-vacancy matrix, this principal submatrix
    # is positive definite, so no regularization is required.
    # ----------------------------------------------------------

    Q_grounded = (
        beta
        * kappa
        * sp.kron(
            L_grounded,
            sp.eye(2),
            format="csr"
        )
    )

    rhs = beta * f_ext

    # Direct solution
    mu_p = spla.spsolve(Q_grounded, rhs)

    # ----------------------------------------------------------
    # 7. Relaxation contribution
    # ----------------------------------------------------------

    f_ext_T_mu_p = float(
        np.dot(f_ext, mu_p)
    )

    delta_F_relax_p = (
        -0.5 * f_ext_T_mu_p
    )

    # ----------------------------------------------------------
    # 8. Determinant contribution
    #
    # Matrix-tree theorem:
    #
    # det(L^(p)) / pdet(L) = 1 / N
    #
    # Hence:
    #
    # Delta F_det^(p) = ln(1/N) = -ln(N)
    #
    # for beta = kappa = 1.
    # ----------------------------------------------------------

    delta_F_det_p = -np.log(N)

    # ----------------------------------------------------------
    # 9. Total grounded-node contribution
    # ----------------------------------------------------------

    delta_F_p = (
        delta_F_det_p
        + delta_F_relax_p
    )

    # ----------------------------------------------------------
    # 10. Maximum displacement magnitude
    # ----------------------------------------------------------

    mu_p_2d = mu_p.reshape(
        (N - 1, 2)
    )

    displacement_magnitudes = np.linalg.norm(
        mu_p_2d,
        axis=1
    )

    max_displacement = np.max(
        displacement_magnitudes
    )

    max_index_new = int(
        np.argmax(displacement_magnitudes)
    )

    # Find corresponding original node
    new_to_old = {
        new: old
        for old, new in old_to_new.items()
    }

    max_index_old = new_to_old[max_index_new]

    max_i = max_index_old // L
    max_j = max_index_old % L

    # ----------------------------------------------------------
    # 11. Print final results
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("NUMERICAL RESULTS")
    print("=" * 60)

    print(
        "f_ext^T mu^(p)               : "
        f"{f_ext_T_mu_p:.12f}"
    )

    print(
        "Delta F_relax^(p)             : "
        f"{delta_F_relax_p:.12f}"
    )

    print(
        "Delta F_det^(p) = -ln(N)      : "
        f"{delta_F_det_p:.12f}"
    )

    print(
        "Total Delta F^(p)             : "
        f"{delta_F_p:.12f}"
    )

    print(
        "\nMaximum displacement          : "
        f"{max_displacement:.12f}"
    )

    print(
        "Maximum displacement at node  : "
        f"{max_index_old} "
        f"at ({max_i}, {max_j})"
    )

    print("=" * 60)

    # ----------------------------------------------------------
    # 12. Return all results for independent checking
    # ----------------------------------------------------------

    return {
        "L": L,
        "N": N,
        "vacancy_index": vac_idx,
        "L_grounded": L_grounded,
        "Q_grounded": Q_grounded,
        "f_ext": f_ext,
        "mu_p": mu_p,
        "f_ext_T_mu_p": f_ext_T_mu_p,
        "delta_F_relax_p": delta_F_relax_p,
        "delta_F_det_p": delta_F_det_p,
        "delta_F_p": delta_F_p,
        "max_displacement": max_displacement,
        "max_displacement_old_index": max_index_old
    }


if __name__ == "__main__":

    results = run_grounded_comparison(
        L=32,
        beta=1.0,
        kappa=1.0,
        f0=0.05
    )