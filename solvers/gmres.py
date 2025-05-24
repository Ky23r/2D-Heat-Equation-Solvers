import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags
from scipy.sparse.linalg import gmres
from numpy.linalg import norm

from utils.grid_initialization import init_T_grid
from utils.create_coordinate_axes import create_coordinate_axes


def gmres_solver(
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

    def _gmres_callback(rk):
        error_history.append(norm(rk, np.inf))

    start_time = time.time()
    x_vec, info = gmres(A, b, rtol=convergence_threshold, callback=_gmres_callback)
    elapsed_time = time.time() - start_time

    if verbose:
        print("GMRES Method")
        print(f"Computation time: {elapsed_time:.4f}")
        # print(f"Final residual ∞-norm: {error_history[-1]:.4e}")
        print(f"Iterations: {len(error_history)}\n")

    T_grid = x_vec.reshape((ny, nx))

    if plot:
        x, y, _, _ = create_coordinate_axes(length_x, length_y, nx, ny)
        plt.figure()
        plt.contourf(x, y, T_grid, levels=50)
        plt.colorbar()
        plt.title(
            "Steady-State Temperature Distribution in a 2D Plane\nUsing the GMRES Method"
        )
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.tight_layout()
        plt.show()

    return T_grid, error_history, elapsed_time
