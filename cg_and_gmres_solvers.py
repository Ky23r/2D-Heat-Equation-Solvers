import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, linalg
import time

# Parameters
nx = ny = 50
Lx = Ly = 1.0
N = nx * ny
hx = Lx / (nx - 1)
hy = Ly / (ny - 1)

# Boundary temperatures
T_bottom = 400
T_top = 200
T_left = 300
T_right = 300

# Helper to convert 2D index to 1D
idx = lambda i, j: i * nx + j

# Create coefficient matrix A and right-hand side b
main_diag = -4.0 * np.ones(N)
off_diag = np.ones(N)

# A in sparse format
A = diags(
    [main_diag, off_diag, off_diag, off_diag, off_diag],
    [0, -1, 1, -nx, nx],
    shape=(N, N),
).tocsc()

# Zero out connections across row boundaries
temp = A.toarray()
for i in range(ny):
    if i < ny - 1:
        temp[idx(i, nx - 1), idx(i, nx)] = 0
    if i > 0:
        temp[idx(i, 0), idx(i, 0) - 1] = 0
A = (
    diags(temp.diagonal(0), 0)
    + diags(temp.diagonal(-1), -1)
    + diags(temp.diagonal(1), 1)
    + diags(temp.diagonal(-nx), -nx)
    + diags(temp.diagonal(nx), nx)
)
A = A.tocsc()

# Right-hand side vector b
b = np.zeros(N)
for i in range(ny):
    for j in range(nx):
        id = idx(i, j)
        if i == 0:
            b[id] -= T_bottom
        if i == ny - 1:
            b[id] -= T_top
        if j == 0:
            b[id] -= T_left
        if j == nx - 1:
            b[id] -= T_right

# Solve using Conjugate Gradient
start = time.time()
x_cg, info_cg = linalg.cg(A, b, rtol=1e-4)
cg_time = time.time() - start
print("Conjugate Gradient Method")
print(f"Computation time: {cg_time:.4f} s")
print(f"Convergence status: {'Success' if info_cg == 0 else 'Failure'}\n")

# Solve using GMRES
start = time.time()
x_gmres, info_gmres = linalg.gmres(A, b, rtol=1e-4)
gmres_time = time.time() - start
print("GMRES Method")
print(f"Computation time: {gmres_time:.4f} s")
print(f"Convergence status: {'Success' if info_gmres == 0 else 'Failure'}\n")

# Reshape and plot
T_cg = x_cg.reshape((ny, nx))
T_gmres = x_gmres.reshape((ny, nx))

plt.figure(1)
plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), T_cg, levels=50)
plt.colorbar()
plt.title("Temperature Distribution by Conjugate Gradient")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.figure(2)
plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), T_gmres, levels=50)
plt.colorbar()
plt.title("Temperature Distribution by GMRES")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()
