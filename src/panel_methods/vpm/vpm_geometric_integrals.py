import pathlib
from itertools import product

import h5py
import numba as nb
import numpy as np
import tqdm

from src.code_collections import data_collections as dc
from ..utils import compute_repeating_terms, normal_geometric_integral_vortex, \
    tangential_geometric_integral_vortex, horizontal_geometric_integral_vortex, vertical_geometric_integral_vortex


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
