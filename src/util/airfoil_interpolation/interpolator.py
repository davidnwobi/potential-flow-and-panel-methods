"""
B-spline approximation.

Author(s): Wei Chen (wchen459@umd.edu)

X : Each row contains the coordinates of B-spline control points and 
    the parameter at the leading edge: 
        `[x0, ..., xn; y0, ..., yn; u_head]`.
"""

import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from .cartesian import read_cartesian
from .interp import interpolate


def interpolator(file_path, num_points, degree=3):
    """
    Interpolate N points whose concentration is based on curvature.

    Parameters
    ----------

    file_path : str
        Path to the airfoil data file.

    num_points : int
        Number of interpolated points.

    degree : int
        Degree of B-splines.
    """
    Q = read_cartesian(file_path)
    x_new, y_new, fp, ier = interpolate(Q, num_points, degree)
    return np.vstack((x_new, y_new)).T
