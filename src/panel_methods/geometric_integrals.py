import numpy as np
from ..code_collections import data_collections as dc
import numba as nb
import h5py
from itertools import product
import tqdm
import pathlib
import typing as tp

__all__ = ['compute_panel_geometric_integrals_source_nb', 'compute_grid_geometric_integrals_source_nb',
           'compute_panel_geometric_integrals_vortex_nb', 'compute_grid_geometric_integrals_vortex_nb',
           'compute_grid_geometric_integrals_source', 'compute_grid_geometric_integrals_vortex']


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
def compute_panel_geometric_integrals_source_nb(panel_geometry: dc.PanelizedGeometryNb) -> (np.ndarray, np.ndarray):
    I = np.empty((panel_geometry.S.size, panel_geometry.S.size))
    J = np.empty((panel_geometry.S.size, panel_geometry.S.size))
    X = panel_geometry.xC - panel_geometry.S / 2 * np.cos(panel_geometry.phi)
    Y = panel_geometry.yC - panel_geometry.S / 2 * np.sin(panel_geometry.phi)
    for i in range(panel_geometry.S.size):
        A, B, E = compute_repeating_terms(panel_geometry.xC[i], panel_geometry.yC[i], X,
                                          Y, panel_geometry.phi)
        I[i, :] = normal_geometric_integral_source(panel_geometry.xC[i], panel_geometry.yC[i], X,
                                                   Y, panel_geometry.phi[i], panel_geometry.phi, panel_geometry.S,
                                                   A, B, E)

        J[i, :] = tangential_geometric_integral_source(panel_geometry.xC[i], panel_geometry.yC[i], X,
                                                       Y, panel_geometry.phi[i], panel_geometry.phi,
                                                       panel_geometry.S,
                                                       A, B, E)
        mask = np.where(np.isnan(E) | np.isinf(E) | np.iscomplex(E), True, False)
        I[i, mask] = 0.
        J[i, mask] = 0.

    I = np.where(np.isnan(I) | np.isinf(I) | np.iscomplex(I), 0, I)
    J = np.where(np.isnan(J) | np.isinf(J) | np.iscomplex(J), 0, J)

    np.fill_diagonal(I, np.pi)
    np.fill_diagonal(J, 0.)

    return I, J


@nb.njit(cache=True)
def compute_grid_geometric_integrals_source_nb(panel_geometry: dc.PanelizedGeometryNb, grid_x: np.ndarray,
                                               grid_y: np.ndarray):
    Ixpj = np.empty((grid_y.size, grid_x.size, panel_geometry.S.size))
    Jypj = np.empty((grid_y.size, grid_x.size, panel_geometry.S.size))
    X = panel_geometry.xC - panel_geometry.S / 2 * np.cos(panel_geometry.phi)
    Y = panel_geometry.yC - panel_geometry.S / 2 * np.sin(panel_geometry.phi)
    for i in range(grid_x.size):
        for j in range(grid_y.size):
            A, B, E = compute_repeating_terms(grid_x[i], grid_y[j], X,
                                              Y, panel_geometry.phi)

            M_xpj = horizontal_geometric_integral_source(grid_x[i], X, panel_geometry.phi, panel_geometry.S,
                                                         A, B, E)
            M_ypj = vertical_geometric_integral_source(grid_y[j], Y, panel_geometry.phi, panel_geometry.S,
                                                       A, B, E)
            mask_M_xpj = np.where(np.isnan(M_xpj))
            mask_M_ypj = np.where(np.isnan(M_ypj))
            M_xpj[mask_M_xpj] = 0.
            M_ypj[mask_M_ypj] = 0.
            Ixpj[j, i], Jypj[j, i] = M_xpj, M_ypj

    Ixpj = np.where(np.isnan(Ixpj) | np.isinf(Ixpj) | np.iscomplex(Ixpj), 0, Ixpj)
    Jypj = np.where(np.isnan(Jypj) | np.isinf(Jypj) | np.iscomplex(Jypj), 0, Jypj)
    return Ixpj, Jypj


@nb.njit(cache=True)
def compute_panel_geometric_integrals_vortex_nb(panel_geometry: dc.PanelizedGeometryNb) -> (np.ndarray, np.ndarray):
    K = np.empty((panel_geometry.S.size, panel_geometry.S.size))
    L = np.empty((panel_geometry.S.size, panel_geometry.S.size))
    X = panel_geometry.xC - panel_geometry.S / 2 * np.cos(panel_geometry.phi)
    Y = panel_geometry.yC - panel_geometry.S / 2 * np.sin(panel_geometry.phi)
    for i in range(panel_geometry.S.size):
        A, B, E = compute_repeating_terms(panel_geometry.xC[i], panel_geometry.yC[i], X,
                                          Y, panel_geometry.phi)
        K[i, :] = normal_geometric_integral_vortex(panel_geometry.xC[i], panel_geometry.yC[i], X,
                                                   Y, panel_geometry.phi[i], panel_geometry.phi, panel_geometry.S,
                                                   A, B, E)

        L[i, :] = tangential_geometric_integral_vortex(panel_geometry.xC[i], panel_geometry.yC[i], X,
                                                       Y, panel_geometry.phi[i], panel_geometry.phi,
                                                       panel_geometry.S,
                                                       A, B, E)
        mask = np.where(np.isnan(E) | np.isinf(E) | np.iscomplex(E), True, False)
        K[i, mask] = 0.
        L[i, mask] = 0.

    K = np.where(np.isnan(K) | np.isinf(K) | np.iscomplex(K), 0, K)
    L = np.where(np.isnan(L) | np.isinf(L) | np.iscomplex(L), 0, L)

    np.fill_diagonal(K, 0.)
    np.fill_diagonal(L, 0)

    return K, L


@nb.njit(cache=True)
def compute_grid_geometric_integrals_vortex_nb(panel_geometry: dc.PanelizedGeometryNb, grid_x: np.ndarray,
                                               grid_y: np.ndarray):
    Nxpj = np.empty((grid_y.size, grid_x.size, panel_geometry.S.size))
    Nypj = np.empty((grid_y.size, grid_x.size, panel_geometry.S.size))
    X = panel_geometry.xC - panel_geometry.S / 2 * np.cos(panel_geometry.phi)
    Y = panel_geometry.yC - panel_geometry.S / 2 * np.sin(panel_geometry.phi)
    for i in range(grid_x.size):
        for j in range(grid_y.size):
            A, B, E = compute_repeating_terms(grid_x[i], grid_y[j], X,
                                              Y, panel_geometry.phi)

            N_xpj = horizontal_geometric_integral_vortex(grid_y[j], Y, panel_geometry.phi, panel_geometry.S,
                                                         A, B, E)
            N_ypj = vertical_geometric_integral_vortex(grid_x[i], X, panel_geometry.phi, panel_geometry.S,
                                                       A, B, E)
            mask_N_xpj = np.where(np.isnan(N_xpj))
            mask_N_ypj = np.where(np.isnan(N_ypj))
            N_xpj[mask_N_xpj] = 0.
            N_ypj[mask_N_ypj] = 0.
            Nxpj[j, i], Nypj[j, i] = N_xpj, N_ypj

    Nxpj = np.where(np.isnan(Nxpj) | np.isinf(Nxpj) | np.iscomplex(Nxpj), 0, Nxpj)
    Nypj = np.where(np.isnan(Nypj) | np.isinf(Nypj) | np.iscomplex(Nypj), 0, Nypj)
    return Nxpj, Nypj


def compute_grid_geometric_integrals_source(panel_geometry: dc.PanelizedGeometryNb, grid_x: np.ndarray,
                                            grid_y: np.ndarray):
    print('\tGrid Geometric integrals for source panels')
    pathlib.Path.unlink('Mxpj.h5', missing_ok=True)
    pathlib.Path.unlink('Mypj.h5', missing_ok=True)
    Mxpjf = h5py.File('Mxpj.h5', 'a')
    Mypjf = h5py.File('Mypj.h5', 'a')
    Mxpj = Mxpjf.create_dataset('Mxpj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
    Mypj = Mypjf.create_dataset('Mypj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
    X = panel_geometry.xC - panel_geometry.S / 2 * np.cos(panel_geometry.phi)
    Y = panel_geometry.yC - panel_geometry.S / 2 * np.sin(panel_geometry.phi)
    array_points = product(range(grid_x.size), range(grid_y.size))
    for i, j in tqdm.tqdm(array_points, total=grid_x.size * grid_y.size, position=1):
        A, B, E = compute_repeating_terms(grid_x[i], grid_y[j], X,
                                          Y, panel_geometry.phi)

        M_xpj = horizontal_geometric_integral_source(grid_x[i], X, panel_geometry.phi, panel_geometry.S,
                                                     A, B, E)
        M_ypj = vertical_geometric_integral_source(grid_y[j], Y, panel_geometry.phi, panel_geometry.S,
                                                   A, B, E)

        del A, B, E
        mask_M_xpj = np.where(np.isnan(M_xpj))
        mask_M_ypj = np.where(np.isnan(M_ypj))
        M_xpj[mask_M_xpj] = 0.
        M_ypj[mask_M_ypj] = 0.
        M_xpj = np.where(np.isnan(M_xpj) | np.isinf(M_xpj) | np.iscomplex(M_xpj), 0, M_xpj)
        M_ypj = np.where(np.isnan(M_ypj) | np.isinf(M_ypj) | np.iscomplex(M_ypj), 0, M_ypj)
        Mxpj[j, i], Mypj[j, i] = M_xpj, M_ypj
        del M_xpj, M_ypj

    return Mxpj, Mypj


def compute_grid_geometric_integrals_vortex(panel_geometry: dc.PanelizedGeometryNb, grid_x: np.ndarray,
                                            grid_y: np.ndarray):
    print('\tGrid Geometric integrals for vortex panels')
    pathlib.Path.unlink('Nxpj.h5', missing_ok=True)
    pathlib.Path.unlink('Nypj.h5', missing_ok=True)
    Nxpjf = h5py.File('Nxpj.h5', 'a')
    Nypjf = h5py.File('Nypj.h5', 'a')
    Nxpj = Nxpjf.create_dataset('Nxpj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
    Nypj = Nypjf.create_dataset('Nypj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
    X = panel_geometry.xC - panel_geometry.S / 2 * np.cos(panel_geometry.phi)
    Y = panel_geometry.yC - panel_geometry.S / 2 * np.sin(panel_geometry.phi)
    array_points = product(range(grid_x.size), range(grid_y.size))
    for i, j in tqdm.tqdm(array_points, total=grid_x.size * grid_y.size, position=1):
        A, B, E = compute_repeating_terms(grid_x[i], grid_y[j], X,
                                          Y, panel_geometry.phi)

        N_xpj = horizontal_geometric_integral_vortex(grid_y[j], Y, panel_geometry.phi, panel_geometry.S,
                                                     A, B, E)
        N_ypj = vertical_geometric_integral_vortex(grid_x[i], X, panel_geometry.phi, panel_geometry.S,
                                                   A, B, E)

        del A, B, E
        mask_N_xpj = np.where(np.isnan(N_xpj))
        mask_N_ypj = np.where(np.isnan(N_ypj))
        N_xpj[mask_N_xpj] = 0.
        N_ypj[mask_N_ypj] = 0.
        N_xpj = np.where(np.isnan(N_xpj) | np.isinf(N_xpj) | np.iscomplex(N_xpj), 0, N_xpj)
        N_ypj = np.where(np.isnan(N_ypj) | np.isinf(N_ypj) | np.iscomplex(N_ypj), 0, N_ypj)
        Nxpj[j, i], Nypj[j, i] = N_xpj, N_ypj
        del N_xpj, N_ypj

    return Nxpj, Nypj
