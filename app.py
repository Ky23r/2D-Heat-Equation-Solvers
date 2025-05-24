import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import diags
from scipy.sparse.linalg import cg, gmres

st.set_page_config(page_title="2D Heat Equation Solvers", layout="wide")
st.title("2D Steady-State Heat Equation Solver")

# --- Sidebar for global parameters ---
st.sidebar.header("Global Parameters")
length_x = st.sidebar.number_input("Plate length in X", min_value=0.1, value=1.0)
length_y = st.sidebar.number_input("Plate length in Y", min_value=0.1, value=1.0)
nx = st.sidebar.number_input(
    "Grid points in X", min_value=10, max_value=500, value=50, step=1
)
ny = st.sidebar.number_input(
    "Grid points in Y", min_value=10, max_value=500, value=50, step=1
)

temp_bottom = st.sidebar.number_input("Temperature Bottom", value=400.0)
temp_top = st.sidebar.number_input("Temperature Top", value=200.0)
temp_left = st.sidebar.number_input("Temperature Left", value=300.0)
temp_right = st.sidebar.number_input("Temperature Right", value=300.0)

tol = st.sidebar.number_input("Error tolerance", value=1e-4, format="%.1e")

# Generate spatial grid
x = np.linspace(0, length_x, int(nx))
y = np.linspace(0, length_y, int(ny))

method = st.sidebar.selectbox(
    "Solver Method",
    [
        "Jacobi",
        "Gauss-Seidel",
        "SOR 5-point",
        "SOR 9-point",
        "Conjugate Gradient",
        "GMRES",
    ],
)

# Method-specific parameter
omega = None
if "SOR" in method:
    omega = st.sidebar.slider("Relaxation factor \u03c9", 1.0, 1.99, 1.5, 0.01)

run = st.sidebar.button("Run Solver")


# Helper: initialize temperature grid
def init_grid():
    T = np.full(
        (int(ny), int(nx)), (temp_bottom + temp_top + temp_left + temp_right) * 0.25
    )
    T[0, :] = temp_bottom
    T[-1, :] = temp_top
    T[1:-1, 0] = temp_left
    T[1:-1, -1] = temp_right
    T[0, 0] = (temp_bottom + temp_left) / 2
    T[-1, -1] = (temp_top + temp_right) / 2
    T[-1, 0] = (temp_top + temp_left) / 2
    T[0, -1] = (temp_bottom + temp_right) / 2
    return T


# Solver implementations


def solver_jacobi():
    T = init_grid()
    errors = []
    start = time.time()
    error = np.inf
    while error > tol:
        T0 = T.copy()
        T[1:-1, 1:-1] = 0.25 * (
            T0[2:, 1:-1] + T0[:-2, 1:-1] + T0[1:-1, 2:] + T0[1:-1, :-2]
        )
        error = np.max(np.abs(T - T0))
        errors.append(error)
    return T, errors, time.time() - start


def solver_gauss():
    T = init_grid()
    errors = []
    start = time.time()
    error = np.inf
    while error > tol:
        T0 = T.copy()
        for i in range(1, int(ny) - 1):
            for j in range(1, int(nx) - 1):
                T[i, j] = 0.25 * (
                    T0[i + 1, j] + T[i - 1, j] + T0[i, j + 1] + T[i, j - 1]
                )
        error = np.max(np.abs(T - T0))
        errors.append(error)
    return T, errors, time.time() - start


def solver_sor5():
    T = init_grid()
    errors = []
    start = time.time()
    error = np.inf
    while error > tol:
        T0 = T.copy()
        for i in range(1, int(ny) - 1):
            for j in range(1, int(nx) - 1):
                T[i, j] = (1 - omega) * T0[i, j] + omega * 0.25 * (
                    T[i, j - 1] + T0[i, j + 1] + T[i - 1, j] + T0[i + 1, j]
                )
        error = np.max(np.abs(T - T0))
        errors.append(error)
    return T, errors, time.time() - start


def solver_sor9():
    T = init_grid()
    errors = []
    start = time.time()
    error = np.inf
    while error > tol:
        Told = T.copy()
        for i in range(1, int(ny) - 1):
            for j in range(1, int(nx) - 1):
                T[i, j] = (
                    (omega / 5)
                    * (T[i - 1, j] + Told[i + 1, j] + T[i, j - 1] + Told[i, j + 1])
                    + (omega / 20)
                    * (
                        T[i - 1, j - 1]
                        + Told[i + 1, j - 1]
                        + T[i - 1, j + 1]
                        + Told[i + 1, j + 1]
                    )
                    + (1 - omega) * Told[i, j]
                )
        error = np.max(np.abs(T - Told))
        errors.append(error)
    return T, errors, time.time() - start


# Conjugate Gradient with correct stencil
from scipy.sparse import csr_matrix


def solver_cg():
    N = int(nx * ny)
    main = -4 * np.ones(N)
    off = np.ones(N)
    # initial full A including wrap connections
    A_full = diags(
        [main, off, off, off, off], [0, -1, 1, -int(nx), int(nx)], shape=(N, N)
    ).toarray()
    # zero out boundary wraps
    idx = lambda i, j: i * int(nx) + j
    for i in range(int(ny)):
        if i < ny - 1:
            A_full[idx(i, nx - 1), idx(i, nx)] = 0
        if i > 0:
            A_full[idx(i, 0), idx(i, 0) - 1] = 0
    A = csr_matrix(A_full)
    # build b with boundary contributions
    b = np.zeros(N)
    for i in range(int(ny)):
        for j in range(int(nx)):
            k = idx(i, j)
            if i == 0:
                b[k] -= temp_bottom
            if i == int(ny) - 1:
                b[k] -= temp_top
            if j == 0:
                b[k] -= temp_left
            if j == int(nx) - 1:
                b[k] -= temp_right
    start = time.time()
    x, info = cg(A, b, rtol=tol)
    Tcg = x.reshape((int(ny), int(nx)))
    return Tcg, [], time.time() - start


def solver_gmres():
    N = int(nx * ny)
    main = -4 * np.ones(N)
    off = np.ones(N)
    A_full = diags(
        [main, off, off, off, off], [0, -1, 1, -int(nx), int(nx)], shape=(N, N)
    ).toarray()
    idx = lambda i, j: i * int(nx) + j
    for i in range(int(ny)):
        if i < ny - 1:
            A_full[idx(i, nx - 1), idx(i, nx)] = 0
        if i > 0:
            A_full[idx(i, 0), idx(i, 0) - 1] = 0
    A = csr_matrix(A_full)
    b = np.zeros(N)
    for i in range(int(ny)):
        for j in range(int(nx)):
            k = idx(i, j)
            if i == 0:
                b[k] -= temp_bottom
            if i == int(ny) - 1:
                b[k] -= temp_top
            if j == 0:
                b[k] -= temp_left
            if j == int(nx) - 1:
                b[k] -= temp_right
    start = time.time()
    x, info = gmres(A, b, rtol=tol)
    Tgm = x.reshape((int(ny), int(nx)))
    return Tgm, [], time.time() - start


# --- Run and display ---
if run:
    if method == "Jacobi":
        Tsol, errors, tcost = solver_jacobi()
    elif method == "Gauss-Seidel":
        Tsol, errors, tcost = solver_gauss()
    elif method == "SOR 5-point":
        Tsol, errors, tcost = solver_sor5()
    elif method == "SOR 9-point":
        Tsol, errors, tcost = solver_sor9()
    elif method == "Conjugate Gradient":
        Tsol, errors, tcost = solver_cg()
    else:
        Tsol, errors, tcost = solver_gmres()

    st.subheader(f"Results for {method}")
    st.write(f"Computation time: {tcost:.4f} seconds")
    if errors:
        st.write(f"Iterations to converge: {len(errors)}")
    fig, ax = plt.subplots()
    cs = ax.contourf(x, y, Tsol, levels=50)
    fig.colorbar(cs, ax=ax)
    st.pyplot(fig)
