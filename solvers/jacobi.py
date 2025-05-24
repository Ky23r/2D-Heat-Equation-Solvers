import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.grid_initialization import init_T_grid
from utils.create_coordinate_axes import create_coordinate_axes
import numpy as np
import matplotlib.pyplot as plt
import time


def jacobi(
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
    T_grid = init_T_grid(nx, ny, T_bottom, T_top, T_left, T_right)

    error_history = []
    max_error = np.inf

    start_time = time.time()

    while max_error > convergence_threshold:
        previous_T_grid = T_grid.copy()

        T_grid[1:-1, 1:-1] = 0.25 * (
            previous_T_grid[2:, 1:-1]
            + previous_T_grid[:-2, 1:-1]
            + previous_T_grid[1:-1, 2:]
            + previous_T_grid[1:-1, :-2]
        )

        max_error = np.max(np.abs(T_grid - previous_T_grid))
        error_history.append(max_error)

    elapsed_time = time.time() - start_time

    if verbose:
        print("Jacobi Iterative Method")
        print(f"Computation time: {elapsed_time:.4f}")
        print(f"This solver converged after {len(error_history)} iterations.\n")

    if plot:
        x, y, _, _ = create_coordinate_axes(length_x, length_y, nx, ny)
        plt.figure()
        plt.contourf(x, y, T_grid, levels=50)
        plt.colorbar()
        plt.title(
            "Steady-State Temperature Distribution in a 2D Plane\nUsing the Jacobi Iterative Method"
        )
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.tight_layout()
        plt.show()

    return T_grid, error_history, elapsed_time


if __name__ == "__main__":
    _, _, _ = jacobi(1, 1, 50, 50, 1e-4, 400, 200, 300, 300)
