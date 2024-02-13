import numpy as np
import numba as nb

__all__ = []


@nb.njit(cache=True)
def compute_geometric_integral(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, E: np.ndarray,
                               S: np.ndarray) -> np.ndarray:
    I = C * 0.5 * (np.log((S ** 2 + 2 * A * S + B) / B)) + (D - A * C) / E * (
            np.arctan2((S + A), E) - np.arctan2(A, E))
    return I


@nb.njit(cache=True)
def compute_repeating_terms(x_i: float, y_i: float, X_j: np.ndarray, Y_j: np.ndarray, phi_j: np.ndarray) -> (
        np.ndarray, np.ndarray, np.ndarray):
    A = -(x_i - X_j) * np.cos(phi_j) - (y_i - Y_j) * np.sin(phi_j)
    B = (x_i - X_j) ** 2 + (y_i - Y_j) ** 2
    E = np.sqrt(B - A ** 2)
    E = np.where(np.iscomplex(E) | np.isnan(E) | np.isinf(E), 0, E)
    return A, B, E


@nb.njit(cache=True)
def normal_geometric_integral_source(x_i: float, y_i: float, X_j: np.ndarray, Y_j: np.ndarray, phi_i: float,
                                     phi_j: np.ndarray, S_j: np.ndarray, A: np.ndarray, B: np.ndarray, E: np.ndarray
                                     ) -> np.ndarray:
    Cn = np.sin(phi_i - phi_j)
    Dn = -(x_i - X_j) * np.sin(phi_i) + (y_i - Y_j) * np.cos(phi_i)
    I_ij = compute_geometric_integral(A, B, Cn, Dn, E, S_j)

    return I_ij


@nb.njit(cache=True)
def tangential_geometric_integral_source(x_i, y_i, X_j, Y_j, phi_i, phi_j, S_j, A, B, E):
    Ct = -np.cos(phi_i - phi_j)
    Dt = (x_i - X_j) * np.cos(phi_i) + (y_i - Y_j) * np.sin(phi_i)
    J_ij = compute_geometric_integral(A, B, Ct, Dt, E, S_j)

    return J_ij


@nb.njit(cache=True)
def horizontal_geometric_integral_source(x_p, X_j, phi_j, S_j, A, B, E):
    Cx = -np.cos(phi_j)
    Dx = (x_p - X_j)
    M_xpj = compute_geometric_integral(A, B, Cx, Dx, E, S_j)
    return M_xpj


@nb.njit(cache=True)
def vertical_geometric_integral_source(y_p, Y_j, phi_j, S_j, A, B, E):
    Cy = -np.sin(phi_j)
    Dy = (y_p - Y_j)

    M_ypj = compute_geometric_integral(A, B, Cy, Dy, E, S_j)
    return M_ypj


@nb.njit(cache=True)
def normal_geometric_integral_vortex(x_i, y_i, X_j, Y_j, phi_i, phi_j, S_j, A, B, E):
    Cn = -np.cos(phi_i - phi_j)
    Dn = (x_i - X_j) * np.cos(phi_i) + (y_i - Y_j) * np.sin(phi_i)
    K_ij = compute_geometric_integral(A, B, Cn, Dn, E, S_j)

    return K_ij


@nb.njit(cache=True)
def tangential_geometric_integral_vortex(x_i, y_i, X_j, Y_j, phi_i, phi_j, S_j, A, B, E):
    Ct = np.sin(phi_j - phi_i)
    Dt = (x_i - X_j) * np.sin(phi_i) - (y_i - Y_j) * np.cos(phi_i)
    L_ij = compute_geometric_integral(A, B, Ct, Dt, E, S_j)

    return L_ij


@nb.njit(cache=True)
def horizontal_geometric_integral_vortex(y_p, Y_j, phi_j, S_j, A, B, E):
    Cx = np.sin(phi_j)
    Dx = -(y_p - Y_j)
    M_xpj = compute_geometric_integral(A, B, Cx, Dx, E, S_j)
    return M_xpj


@nb.njit(cache=True)
def vertical_geometric_integral_vortex(x_p, X_j, phi_j, S_j, A, B, E):
    Cy = -np.cos(phi_j)
    Dy = (x_p - X_j)
    M_ypj = compute_geometric_integral(A, B, Cy, Dy, E, S_j)
    return M_ypj


@nb.njit(cache=True)
def point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    n = len(polygon)
    inside = False

    x1, y1 = polygon[0]
    for i in range(n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                        if x1 == x2 or x <= x_intersect:
                            inside = not inside
        x1, y1 = x2, y2

    return inside
