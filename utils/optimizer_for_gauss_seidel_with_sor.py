import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from solvers.gauss_seidel_with_5point_sor import gauss_seidel_with_5point_sor
from solvers.gauss_seidel_with_9point_sor import gauss_seidel_with_9point_sor

SOLVER_NAMES = {
    "gauss_seidel_with_5point_sor": "Gauss-Seidel Iterative Method with 5-Point SOR",
    "gauss_seidel_with_9point_sor": "Gauss-Seidel Iterative Method with 9-Point SOR",
}


def find_optimal_omega(
    solver,
    length_x,
    length_y,
    nx,
    ny,
    convergence_threshold,
    T_bottom,
    T_top,
    T_left,
    T_right,
    omega_start=1,
    omega_stop=2,
    omega_step=0.01,
    plot=True,
    verbose=True,
):
    if verbose:
        print(
            f"Find optimal relaxation parameter (ω) for {SOLVER_NAMES[solver.__name__]}\n"
        )

    omega_values = np.arange(omega_start, omega_stop, omega_step)
    iteration_counts = np.zeros_like(omega_values)

    for idx, omega in enumerate(omega_values):
        _, error_history, _ = solver(
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
            False,
            False,
        )
        iteration_counts[idx] = len(error_history)
        if verbose:
            print(
                f"With ω = {omega:.2f}, the {SOLVER_NAMES[solver.__name__]} converged in {len(error_history)} iterations."
            )

    best_idx = np.argmin(iteration_counts)
    best_omega = omega_values[best_idx]
    best_iters = iteration_counts[best_idx]

    if verbose:
        print(
            f"\nOptimal relaxation parameter: ω = {best_omega:.2f}, converging in only {int(best_iters)} iterations."
        )

    if plot:
        plt.figure()
        plt.plot(omega_values, iteration_counts)
        plt.xlabel("Relaxation Parameter (ω)")
        plt.ylabel("Iterations to Converge")
        plt.title(
            f"Influence of the Relaxation Parameter on Convergence Speed\n{SOLVER_NAMES[solver.__name__]}"
        )
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return best_omega, omega_values, iteration_counts


if __name__ == "__main__":
    length_x, length_y = 1, 1
    nx = ny = 50
    convergence_threshold = 1e-4

    T_bottom, T_top, T_left, T_right = 400, 200, 300, 300

    # _, _, _ = find_optimal_omega(
    #     gauss_seidel_with_5point_sor,
    #     length_x,
    #     length_y,
    #     nx,
    #     ny,
    #     convergence_threshold,
    #     T_bottom,
    #     T_top,
    #     T_left,
    #     T_right,
    # )

    _, _, _ = find_optimal_omega(
        gauss_seidel_with_9point_sor,
        length_x,
        length_y,
        nx,
        ny,
        convergence_threshold,
        T_bottom,
        T_top,
        T_left,
        T_right,
    )
