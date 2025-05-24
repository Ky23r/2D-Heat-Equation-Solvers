import numpy as np


def init_T_grid(nx, ny, T_bottom, T_top, T_left, T_right):
    """
    Return initial temperature grid with boundary conditions applied.
    """
    T_grid = np.full((ny, nx), (T_bottom + T_top + T_left + T_right) * 0.25)
    T_grid[0, :] = T_bottom
    T_grid[-1, :] = T_top
    T_grid[1:-1, 0] = T_left
    T_grid[1:-1, -1] = T_right
    T_grid[0, 0] = (T_bottom + T_left) / 2
    T_grid[-1, -1] = (T_top + T_right) / 2
    T_grid[-1, 0] = (T_top + T_left) / 2
    T_grid[0, -1] = (T_bottom + T_right) / 2
    return T_grid
