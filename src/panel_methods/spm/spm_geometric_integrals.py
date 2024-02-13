import pathlib
from itertools import product
import h5py
import numba as nb
import numpy as np
import tqdm
from ...code_collections import data_collections as dc
from ..utils import compute_repeating_terms, normal_geometric_integral_source, \
    tangential_geometric_integral_source, horizontal_geometric_integral_source, vertical_geometric_integral_source


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
