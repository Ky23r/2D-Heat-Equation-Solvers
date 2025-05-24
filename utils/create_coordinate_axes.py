import numpy as np


def create_coordinate_axes(length_x, length_y, nx, ny):
    x = np.linspace(0.0, length_x, nx)
    y = np.linspace(0.0, length_y, ny)
    dx = length_x / (nx - 1)
    dy = length_y / (ny - 1)
    return x, y, dx, dy
