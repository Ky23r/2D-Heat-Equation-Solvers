import matplotlib.pyplot as plt

from solvers.jacobi import jacobi
from solvers.gauss_seidel import gauss_seidel
from solvers.gauss_seidel_with_5point_sor import gauss_seidel_with_5point_sor
from solvers.gauss_seidel_with_9point_sor import gauss_seidel_with_9point_sor
from utils.optimizer_for_gauss_seidel_with_sor import find_optimal_omega
from utils.convergence_visualization import plot_convergence_curves


def get_float(prompt: str, default: float = None) -> float:
    """
    Prompt the user to enter a floating-point number.
    If default is provided and the user presses Enter without typing anything,
    returns the default value.
    Repeats until a valid float is entered.
    """
    while True:
        entry = input(prompt).strip()
        if entry == "" and default is not None:
            return default
        try:
            value = float(entry)
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def get_int(prompt: str) -> int:
    """
    Prompt the user to enter an integer.
    Repeats until a valid integer is entered.
    """
    while True:
        entry = input(prompt).strip()
        try:
            value = int(entry)
            return value
        except ValueError:
            print("Invalid input. Please enter a valid integer.")


length_x = get_float("Enter the domain length in the x-direction (length_x): ")
length_y = get_float("Enter the domain length in the y-direction (length_y): ")

nx = get_int("Enter the number of grid points along x (nx, positive integer): ")
ny = get_int("Enter the number of grid points along y (ny, positive integer): ")

convergence_threshold = get_float(
    "Enter the convergence threshold [press Enter for default 1e-4]: ", default=1e-4
)

T_bottom = get_float("Enter the temperature at the bottom boundary (T_bottom): ")
T_top = get_float("Enter the temperature at the top boundary (T_top): ")
T_left = get_float("Enter the temperature at the left boundary (T_left): ")
T_right = get_float("Enter the temperature at the right boundary (T_right): ")

# optimal_omega_for_gs_with_5point_sor = find_optimal_omega(
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
#     1,
#     2,
#     0.01,
#     True,
#     False,
# )

# optimal_omega_for_gs_with_9point_sor = find_optimal_omega(
#     gauss_seidel_with_9point_sor,
#     length_x,
#     length_y,
#     nx,
#     ny,
#     convergence_threshold,
#     T_bottom,
#     T_top,
#     T_left,
#     T_right,
#     1,
#     2,
#     0.01,
#     True,
#     False,
# )

_, jacobi_error_history, _ = jacobi(
    length_x,
    length_y,
    nx,
    ny,
    convergence_threshold,
    T_bottom,
    T_top,
    T_left,
    T_right,
    False,
    True,
)

_, gs_error_history, _ = gauss_seidel(
    length_x,
    length_y,
    nx,
    ny,
    convergence_threshold,
    T_bottom,
    T_top,
    T_left,
    T_right,
    False,
    True,
)

_, gs_with_5point_sor_error_history, _ = gauss_seidel_with_5point_sor(
    length_x,
    length_y,
    nx,
    ny,
    convergence_threshold,
    1.97,
    T_bottom,
    T_top,
    T_left,
    T_right,
    False,
    True,
)

_, gs_with_9point_sor_error_history, _ = gauss_seidel_with_9point_sor(
    length_x,
    length_y,
    nx,
    ny,
    convergence_threshold,
    1.97,
    T_bottom,
    T_top,
    T_left,
    T_right,
    False,
    True,
)

plot_convergence_curves(
    jacobi_error_history,
    gs_error_history,
    gs_with_5point_sor_error_history,
    gs_with_9point_sor_error_history,
    labels=[
        "Jacobi Iterative Method",
        "Gauss-Seidel Iterative Method",
        "Gauss-Seidel Iterative Method with 5-Point SOR",
        "Gauss-Seidel Iterative Method with 9-Point SOR",
    ],
)
