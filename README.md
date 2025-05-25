# 2D Heat Equation Solvers

## Introduction

The steady‐state two‐dimensional heat (Laplace) equation is a classical partial differential equation that models the distribution of temperature in a thin, homogeneous plate with fixed boundary temperatures. Numerical solution of this equation is a fundamental problem in computational physics and engineering, with applications ranging from thermal management in electronics to geophysical heat flow.

This project provides a comprehensive Python toolkit and an interactive Streamlit application for solving the 2D steady‐state heat equation on a rectangular domain. It implements a suite of state‐of‐the‐art iterative algorithms—including Jacobi, Gauss–Seidel, Successive Over‐Relaxation (SOR) with both 5-point and 9-point stencils, and the Conjugate Gradient method—allowing users to experiment with different approaches and compare their convergence behavior. For SOR methods, an automated optimizer searches for the optimal relaxation parameter, eliminating guesswork and reducing time to solution.

## Installation

### Requirements

- Python 3.10 or higher

### Step 1: Clone the repository

```bash
git clone https://github.com/Ky23r/2D-Heat-Equation-Solvers.git
cd 2D-Heat-Equation-Solvers
```

### Step 2: Install required libraries

```bash
pip install -r requirements.txt
```

## Usage

The project can be used via a web interface (Streamlit) or directly through a command-line Python script.

### As a web interface (Streamlit)

```bash
cd application
streamlit run app.py
```

1. **Input Values**: Set domain lengths, grid points, convergence threshold, and boundary temperatures in the sidebar.

2. **Execution Mode**:

- Single-Solver Run: Select one algorithm, optionally tune or auto-find ω, then click Solve.

- Convergence Performance Analysis: Tick multiple solvers, set ω values for SOR if needed, then click Evaluate.

3. **Output**: View contour plots of the steady-state temperature and/or convergence curves.

### As a Python script-based version

```bash
python main.py
```

This version provides a command-line interface for running and comparing different iterative solvers for the 2D steady-state heat equation without the need for a graphical interface. Users are prompted to input domain parameters, boundary conditions, convergence criteria, and select which solvers to run.

**Available solvers include**:

- Jacobi Iterative Method

- Gauss–Seidel Iterative Method

- Gauss–Seidel Iterative Method with 5-Point Successive Over-Relaxation (SOR)

- Gauss–Seidel Iterative Method with 9-Point SOR

- Conjugate Gradient Method

**Workflow**:

1. **Input** – Define the domain size, grid resolution, convergence threshold, and fixed boundary temperatures directly in the terminal.

2. **Solver Selection** – Choose one or more solvers to run from a numbered list.

3. **Execution** – Each selected solver computes the steady-state temperature distribution using the specified parameters.

4. **Output** – A convergence behavior plot is generated showing the error decay of each selected solver, allowing users to visually compare their performance.

## Project Structure

```
2D-Heat-Equation-Solvers/
│
├── main.py
├── requirements.txt
│
├── solvers/
│   ├── jacobi.py
│   ├── gauss_seidel.py
│   ├── gauss_seidel_with_5point_sor.py
│   ├── gauss_seidel_with_9point_sor.py
│   └── conjugate_gradient.py
│
└── utils/
    ├── create_coordinate_axes.py
    ├── grid_initialization.py
    ├── optimizer_for_gauss_seidel_with_sor.py
    └── convergence_visualization.py
```
