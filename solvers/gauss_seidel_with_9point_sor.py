import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.grid_initialization import init_T_grid
from utils.create_coordinate_axes import create_coordinate_axes
import numpy as np
import matplotlib.pyplot as plt
import time


def gauss_seidel_with_9point_sor(
    length_x,
    length_y,
    nx,
    ny,
    convergence_threshold,
    omega,
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

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                sum_of_adjacent_cells = (
                    T_grid[i - 1, j]
                    + previous_T_grid[i + 1, j]
                    + T_grid[i, j - 1]
                    + previous_T_grid[i, j + 1]
                )
                sum_of_diagonal_cells = (
                    T_grid[i - 1, j - 1]
                    + previous_T_grid[i + 1, j - 1]
                    + T_grid[i - 1, j + 1]
                    + previous_T_grid[i + 1, j + 1]
                )
                T_grid[i, j] = (
                    (omega / 5) * sum_of_adjacent_cells
                    + (omega / 20) * sum_of_diagonal_cells
                    + (1 - omega) * previous_T_grid[i, j]
                )

        max_error = np.max(np.abs(T_grid - previous_T_grid))
        error_history.append(max_error)

    elapsed_time = time.time() - start_time

    if verbose:
        print("Gauss-Seidel Iterative Method with 9-Point SOR")
        print(f"Computation time: {elapsed_time:.4f}")
        print(f"This solver converged after {len(error_history)} iterations.\n")

    if plot:
        x, y, _, _ = create_coordinate_axes(length_x, length_y, nx, ny)
        plt.figure()
        plt.contourf(x, y, T_grid, levels=50)
        plt.colorbar()
        plt.title(
            "Steady-State Temperature Distribution in a 2D Plane\nUsing the Gauss-Seidel Iterative Method with 9-Point SOR"
        )
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.tight_layout()
        plt.show()

    return T_grid, error_history, elapsed_time


if __name__ == "__main__":
    _, _, _ = gauss_seidel_with_9point_sor(1, 1, 50, 50, 1e-4, 1.87, 400, 200, 300, 300)
