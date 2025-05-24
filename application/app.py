import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from utils.create_coordinate_axes import create_coordinate_axes
from solvers.jacobi import jacobi
from solvers.gauss_seidel import gauss_seidel
from solvers.gauss_seidel_with_5point_sor import gauss_seidel_with_5point_sor
from solvers.gauss_seidel_with_9point_sor import gauss_seidel_with_9point_sor
from solvers.conjugate_gradient import conjugate_gradient
from solvers.gmres import gmres

st.set_page_config(page_title="2D Heat Equation Solvers", layout="wide")
st.title("2D Heat Equation Solvers")

length_x_str = st.sidebar.text_input(
    "Enter the domain length in the x-direction", value="1.0"
)
length_y_str = st.sidebar.text_input(
    "Enter the domain length in the y-direction", value="1.0"
)
nx_str = st.sidebar.text_input(
    "Enter the number of grid points along x (positive integer)", value="50"
)
ny_str = st.sidebar.text_input(
    "Enter the number of grid points along y (positive integer)", value="50"
)
T_bottom_str = st.sidebar.text_input(
    "Enter the temperature at the bottom boundary", value="400.0"
)
T_top_str = st.sidebar.text_input(
    "Enter the temperature at the top boundary", value="200.0"
)
T_left_str = st.sidebar.text_input(
    "Enter the temperature at the left boundary", value="300.0"
)
T_right_str = st.sidebar.text_input(
    "Enter the temperature at the right boundary", value="300.0"
)
convergence_threshold_str = st.sidebar.text_input(
    "Enter the convergence threshold", value="1e-4"
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

st.sidebar.success("✔︎ All parameters are valid.")

x, y, _, _ = create_coordinate_axes(length_x, length_y, nx, ny)

solver = st.sidebar.selectbox(
    "Solver Method",
    [
        "Jacobi Iterative Method",
        "Gauss-Seidel Iterative Method",
        "Gauss-Seidel Iterative Method with 5-Point SOR",
        "Gauss-Seidel Iterative Method with 9-Point SOR",
        "Conjugate Gradient Method",
        "GMRES Method",
    ],
)

omega = None
if "SOR" in solver:
    omega = st.sidebar.slider("Relaxation parameter (ω) \u03c9", 1.0, 1.99, 1.5, 0.01)

run = st.sidebar.button("Solve")


if run:
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
            False,
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
            False,
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
            False,
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
            False,
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
            False,
        )
    elif solver == "GMRES Method":
        T_grid, error_history, elasped_time = gmres(
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

    st.subheader(
        f"Steady-State Temperature Distribution in a 2D Plane Using the {solver}"
    )
    st.write(f"Computation time: {elasped_time:.4f}")
    if error_history:
        st.write(f"The {solver} converged after {len(error_history)} iterations.\n")
    fig, ax = plt.subplots()
    cs = ax.contourf(x, y, T_grid, levels=50)
    fig.colorbar(cs, ax=ax)
    st.pyplot(fig)
