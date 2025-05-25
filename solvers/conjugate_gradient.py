import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags
from numpy.linalg import norm

from utils.grid_initialization import init_T_grid
from utils.create_coordinate_axes import create_coordinate_axes


def conjugate_gradient(
    length_x,
    length_y,
    nx,
    ny,
    convergence_threshold,
    T_bottom,
    T_top,
    T_left,
    T_right,
    plot=True,
    verbose=True,
):
    num_cells = nx * ny

    main_diag = -4.0 * np.ones(num_cells)
    off_diag = np.ones(num_cells)
    A = diags(
        [main_diag, off_diag, off_diag, off_diag, off_diag],
        [0, -1, 1, -nx, nx],
        shape=(num_cells, num_cells),
    ).tocsc()

    A = A.tolil()
    for i in range(ny):
        if i < ny - 1:
            A[i * nx + (nx - 1), i * nx + nx] = 0
        if i > 0:
            A[i * nx + 0, i * nx - 1] = 0
    A = A.tocsc()

    b = np.zeros(num_cells)
    for i in range(ny):
        for j in range(nx):
            idx = i * nx + j
            if i == 0:
                b[idx] -= T_bottom
            if i == ny - 1:
                b[idx] -= T_top
            if j == 0:
                b[idx] -= T_left
            if j == nx - 1:
                b[idx] -= T_right

    error_history = []

    x_vec = np.zeros(num_cells)
    r = b - A.dot(x_vec)
    p = r.copy()
    rs_old = r.dot(r)

    start_time = time.time()
    while norm(r, np.inf) > convergence_threshold:
        Ap = A.dot(p)
        alpha = rs_old / p.dot(Ap)
        x_vec = x_vec + alpha * p
        r = r - alpha * Ap
        rs_new = r.dot(r)
        error_history.append(norm(r, np.inf))
        if rs_new == 0:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    elapsed_time = time.time() - start_time

    if verbose:
        print("Conjugate Gradient Method")
        print(f"Computation time: {elapsed_time:.4f}")
        print(f"This solver converged after {len(error_history)} iterations.\n")

    T_grid = x_vec.reshape((ny, nx))

    if plot:
        x, y, _, _ = create_coordinate_axes(length_x, length_y, nx, ny)
        plt.figure()
        plt.contourf(x, y, T_grid, levels=50)
        plt.colorbar()
        plt.title(
            "Steady-State Temperature Distribution in a 2D Plane\nUsing the Conjugate Gradient Method"
        )
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.tight_layout()
        plt.show()

    return T_grid, error_history, elapsed_time


if __name__ == "__main__":
    _, _, _ = conjugate_gradient(1, 1, 20, 20, 1e-4, 400, 200, 300, 300)
