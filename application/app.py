import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import matplotlib.pyplot as plt

from utils.create_coordinate_axes import create_coordinate_axes
from utils.optimizer_for_sor_method import find_optimal_omega
from solvers.jacobi import jacobi
from solvers.gauss_seidel import gauss_seidel
from solvers.gauss_seidel_with_5point_sor import gauss_seidel_with_5point_sor
from solvers.gauss_seidel_with_9point_sor import gauss_seidel_with_9point_sor
from solvers.conjugate_gradient import conjugate_gradient


def clear_optimal_omegas():
    for key in ["best_omega_for_gs_with_5pt_sor", "best_omega_for_gs_with_9pt_sor"]:
        if key in st.session_state:
            del st.session_state[key]


st.sidebar.header("Input Values")

length_x_str = st.sidebar.text_input(
    "Enter the domain length in the x-direction",
    value="1.0",
    key="length_x",
    on_change=clear_optimal_omegas,
)
length_y_str = st.sidebar.text_input(
    "Enter the domain length in the y-direction",
    value="1.0",
    key="length_y",
    on_change=clear_optimal_omegas,
)
nx_str = st.sidebar.text_input(
    "Enter the number of grid points along x (positive integer)",
    value="50",
    key="nx",
    on_change=clear_optimal_omegas,
)
ny_str = st.sidebar.text_input(
    "Enter the number of grid points along y (positive integer)",
    value="50",
    key="ny",
    on_change=clear_optimal_omegas,
)
T_bottom_str = st.sidebar.text_input(
    "Enter the temperature at the bottom boundary",
    value="400.0",
    key="T_bottom",
    on_change=clear_optimal_omegas,
)
T_top_str = st.sidebar.text_input(
    "Enter the temperature at the top boundary",
    value="200.0",
    key="T_top",
    on_change=clear_optimal_omegas,
)
T_left_str = st.sidebar.text_input(
    "Enter the temperature at the left boundary",
    value="300.0",
    key="T_left",
    on_change=clear_optimal_omegas,
)
T_right_str = st.sidebar.text_input(
    "Enter the temperature at the right boundary",
    value="300.0",
    key="T_right",
    on_change=clear_optimal_omegas,
)
convergence_threshold_str = st.sidebar.text_input(
    "Enter the convergence threshold",
    value="1e-4",
    key="convergence_threshold",
    on_change=clear_optimal_omegas,
)


def parse_float(name, text):
    try:
        return float(text), None
    except ValueError:
        return None, f"`{name}` must be a valid number."


def parse_positive_int(name, text):
    try:
        val = int(text)
        if val < 1:
            raise ValueError
        return val, None
    except ValueError:
        return None, f"`{name}` must be a positive integer."


errors = []

length_x, error = parse_float("The domain length in the x-direction", length_x_str)
if error:
    errors.append(error)

length_y, error = parse_float("The domain length in the y-direction", length_y_str)
if error:
    errors.append(error)

nx, error = parse_positive_int("The number of grid points along x", nx_str)
if error:
    errors.append(error)

ny, error = parse_positive_int("The number of grid points along y", ny_str)
if error:
    errors.append(error)

T_bottom, error = parse_float("The temperature at the bottom boundary", T_bottom_str)
if error:
    errors.append(error)

T_top, error = parse_float("The temperature at the top boundary", T_top_str)
if error:
    errors.append(error)

T_left, error = parse_float("The temperature at the left boundary", T_left_str)
if error:
    errors.append(error)

T_right, error = parse_float("The temperature at the right boundary", T_right_str)
if error:
    errors.append(error)

convergence_threshold, error = parse_float(
    "The convergence threshold", convergence_threshold_str
)
if error:
    errors.append(error)

if errors:
    for error in errors:
        st.sidebar.error(error)
    st.stop()

st.sidebar.success("✔︎ All input values are valid.")

x, y, _, _ = create_coordinate_axes(length_x, length_y, nx, ny)

execution_mode = st.sidebar.selectbox(
    "Execution Mode",
    (
        "Optimize ω for SOR Method",
        "Single-Solver Run",
        "Convergence Performance Analysis",
    ),
    index=1,
)

if execution_mode == "Optimize ω for SOR Method":
    solver = st.sidebar.selectbox(
        "Select Solver",
        [
            "Gauss-Seidel Iterative Method with 5-Point SOR",
            "Gauss-Seidel Iterative Method with 9-Point SOR",
        ],
    )

    verbose = st.sidebar.checkbox("Details", value=True)

    find_opt_omega = st.sidebar.button("Find optimal ω")

    if find_opt_omega:
        if solver == "Gauss-Seidel Iterative Method with 5-Point SOR":
            func = gauss_seidel_with_5point_sor
        elif solver == "Gauss-Seidel Iterative Method with 9-Point SOR":
            func = gauss_seidel_with_9point_sor

        best_omega, omega_values, iteration_counts = find_optimal_omega(
            func,
            length_x,
            length_y,
            nx,
            ny,
            convergence_threshold,
            T_bottom,
            T_top,
            T_left,
            T_right,
            omega_start=1.0,
            omega_stop=2.0,
            omega_step=0.01,
            plot=False,
            log_writer=st.write,
            verbose=verbose,
        )

        if solver == "Gauss-Seidel Iterative Method with 5-Point SOR":
            st.session_state.best_omega_for_gs_with_5pt_sor = best_omega
        elif solver == "Gauss-Seidel Iterative Method with 9-Point SOR":
            st.session_state.best_omega_for_gs_with_9pt_sor = best_omega

        st.sidebar.success(f"Optimal ω = {best_omega:.2f}")

        st.subheader(
            f"Influence of the Relaxation Parameter on Convergence Speed ({solver})"
        )

        fig, ax = plt.subplots()
        ax.plot(omega_values, iteration_counts)
        ax.set_xlabel("Relaxation Parameter (ω)")
        ax.set_ylabel("Iterations to Converge")
        ax.grid(True)
        st.pyplot(fig)

elif execution_mode == "Single-Solver Run":
    solver = st.sidebar.selectbox(
        "Select Solver",
        [
            "Jacobi Iterative Method",
            "Gauss-Seidel Iterative Method",
            "Gauss-Seidel Iterative Method with 5-Point SOR",
            "Gauss-Seidel Iterative Method with 9-Point SOR",
            "Conjugate Gradient Method",
        ],
    )

    omega = None

    if solver == "Gauss-Seidel Iterative Method with 5-Point SOR":
        current_omega = 1.5
        if "best_omega_for_gs_with_5pt_sor" in st.session_state:
            current_omega = st.session_state.best_omega_for_gs_with_5pt_sor
        omega = st.sidebar.slider(
            "Relaxation parameter (ω)",
            1.0,
            1.99,
            current_omega,
            0.01,
        )
        if "best_omega_for_gs_with_5pt_sor" in st.session_state:
            st.sidebar.success(
                f"The optimal relaxation parameter (ω) is {st.session_state.best_omega_for_gs_with_5pt_sor:.2f}."
            )
    elif solver == "Gauss-Seidel Iterative Method with 9-Point SOR":
        current_omega = 1.5
        if "best_omega_for_gs_with_9pt_sor" in st.session_state:
            current_omega = st.session_state.best_omega_for_gs_with_9pt_sor
        omega = st.sidebar.slider(
            "Relaxation parameter (ω)",
            1.0,
            1.99,
            current_omega,
            0.01,
        )
        if "best_omega_for_gs_with_9pt_sor" in st.session_state:
            st.sidebar.success(
                f"The optimal relaxation parameter (ω) is {st.session_state.best_omega_for_gs_with_9pt_sor:.2f}."
            )

    solve = st.sidebar.button("Solve")

    if solve:
        if solver == "Jacobi Iterative Method":
            T_grid, error_history, elasped_time = jacobi(
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
        elif solver == "Gauss-Seidel Iterative Method":
            T_grid, error_history, elasped_time = gauss_seidel(
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
        elif solver == "Gauss-Seidel Iterative Method with 5-Point SOR":
            T_grid, error_history, elasped_time = gauss_seidel_with_5point_sor(
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
                True,
            )
        elif solver == "Gauss-Seidel Iterative Method with 9-Point SOR":
            T_grid, error_history, elasped_time = gauss_seidel_with_9point_sor(
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
                True,
            )
        elif solver == "Conjugate Gradient Method":
            T_grid, error_history, elasped_time = conjugate_gradient(
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

        st.subheader(f"Steady-State Temperature Distribution in a 2D Plane ({solver})")
        st.write(f"Computation time: {elasped_time:.4f}")
        st.write(f"The {solver} converged after {len(error_history)} iterations.\n")
        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, T_grid, levels=50)
        fig.colorbar(cs, ax=ax)
        st.pyplot(fig)

elif execution_mode == "Convergence Performance Analysis":
    st.sidebar.subheader("Evaluating Convergence Rates Across Different Solvers")

    selected_solvers = []

    select_jacobi = st.sidebar.checkbox("Jacobi Iterative Method", value=False)
    select_gs = st.sidebar.checkbox("Gauss-Seidel Iterative Method", value=False)
    select_gs_with_5pt_sor = st.sidebar.checkbox(
        "Gauss-Seidel Iterative Method with 5-Point SOR", value=False
    )
    select_gs_with_9pt_sor = st.sidebar.checkbox(
        "Gauss-Seidel Iterative Method with 9-Point SOR", value=False
    )
    select_cg = st.sidebar.checkbox("Conjugate Gradient Method", value=False)

    if select_jacobi:
        selected_solvers.append("Jacobi Iterative Method")
    if select_gs:
        selected_solvers.append("Gauss-Seidel Iterative Method")
    if select_gs_with_5pt_sor:
        selected_solvers.append("Gauss-Seidel Iterative Method with 5-Point SOR")
    if select_gs_with_9pt_sor:
        selected_solvers.append("Gauss-Seidel Iterative Method with 9-Point SOR")
    if select_cg:
        selected_solvers.append("Conjugate Gradient Method")

    omega_for_gs_with_5pt_sor = None
    omega_for_gs_with_9pt_sor = None

    if "Gauss-Seidel Iterative Method with 5-Point SOR" in selected_solvers:
        current_omega = 1.5
        if "best_omega_for_gs_with_5pt_sor" in st.session_state:
            current_omega = st.session_state.best_omega_for_gs_with_5pt_sor
        omega_for_gs_with_5pt_sor = st.sidebar.slider(
            "Relaxation parameter (ω) for Gauss-Seidel Iterative Method with 5-Point SOR",
            1.0,
            1.99,
            current_omega,
            0.01,
        )
        if "best_omega_for_gs_with_5pt_sor" in st.session_state:
            st.sidebar.success(
                f"The optimal relaxation parameter (ω) is {st.session_state.best_omega_for_gs_with_5pt_sor:.2f}."
            )
    if "Gauss-Seidel Iterative Method with 9-Point SOR" in selected_solvers:
        current_omega = 1.5
        if "best_omega_for_gs_with_9pt_sor" in st.session_state:
            current_omega = st.session_state.best_omega_for_gs_with_9pt_sor
        omega_for_gs_with_9pt_sor = st.sidebar.slider(
            "Relaxation parameter (ω) for Gauss-Seidel Iterative Method with 9-Point SOR",
            1.0,
            1.99,
            current_omega,
            0.01,
        )
        if "best_omega_for_gs_with_9pt_sor" in st.session_state:
            st.sidebar.success(
                f"The optimal relaxation parameter (ω) is {st.session_state.best_omega_for_gs_with_9pt_sor:.2f}."
            )

    evaluate = st.sidebar.button("Evaluate")

    if evaluate and selected_solvers:
        error_histories, labels = [], []
        for solver in selected_solvers:
            if solver == "Jacobi Iterative Method":
                _, error_history, _ = jacobi(
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
                    False,
                )
            elif solver == "Gauss-Seidel Iterative Method":
                _, error_history, _ = gauss_seidel(
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
                    False,
                )
            elif solver == "Gauss-Seidel Iterative Method with 5-Point SOR":
                _, error_history, _ = gauss_seidel_with_5point_sor(
                    length_x,
                    length_y,
                    nx,
                    ny,
                    convergence_threshold,
                    omega_for_gs_with_5pt_sor,
                    T_bottom,
                    T_top,
                    T_left,
                    T_right,
                    False,
                    False,
                )
            elif solver == "Gauss-Seidel Iterative Method with 9-Point SOR":
                _, error_history, _ = gauss_seidel_with_9point_sor(
                    length_x,
                    length_y,
                    nx,
                    ny,
                    convergence_threshold,
                    omega_for_gs_with_9pt_sor,
                    T_bottom,
                    T_top,
                    T_left,
                    T_right,
                    False,
                    False,
                )
            elif solver == "Conjugate Gradient Method":
                _, error_history, _ = conjugate_gradient(
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
                    False,
                )
            error_histories.append(error_history)
            labels.append(solver)
        max_iter = 400
        st.subheader(
            f"Convergence Behavior: Maximum Absolute Error Over Iterations (First {max_iter} Iterations)"
        )
        fig, ax = plt.subplots()
        for error_history, label in zip(error_histories, labels):
            iters_to_plot = min(len(error_history), max_iter)
            ax.plot(
                range(1, iters_to_plot + 1), error_history[:iters_to_plot], label=label
            )
        ax.set_xlabel("Iteration Number")
        ax.set_ylabel("Maximum Absolute Error")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
