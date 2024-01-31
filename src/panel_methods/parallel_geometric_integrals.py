import numpy as np
import h5py
from itertools import product
import pathlib
import tqdm
from joblib import Parallel, delayed
from ..code_collections import data_collections as dc
from . import geometric_integrals as gi
import numba as nb
import time as t
import timeit
from functools import partial

__all__ = ['compute_grid_geometric_integrals_source_mp']


# def compute_grid_geometric_integrals_source_internal(panel_geometry: dc.PanelizedGeometry, grid_x: np.ndarray,
#                                                      grid_y: np.ndarray, core_idx: int):
#     """
#     An internal function to compute the geometric integrals for the source panels.
#     :param panel_geometry:
#     :param grid_x:
#     :param grid_y:
#     :param core_idx:
#     :return:
#     """
#     print(f'\tGrid Geometric integrals core {core_idx}')
#     pathlib.Path.unlink(f'Mxpj_{core_idx}.h5', missing_ok=True)
#     pathlib.Path.unlink(f'Mypj_{core_idx}.h5', missing_ok=True)
#     Mxpjf = h5py.File(f'Mxpj_{core_idx}.h5', 'w')
#     Mypjf = h5py.File(f'Mypj_{core_idx}.h5', 'w')
#     Mxpj = Mxpjf.create_dataset('Mxpj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
#     Mypj = Mypjf.create_dataset('Mypj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
#     X = panel_geometry.xC - panel_geometry.S / 2 * np.cos(panel_geometry.phi)
#     Y = panel_geometry.yC - panel_geometry.S / 2 * np.sin(panel_geometry.phi)
#     array_points = product(range(grid_x.size), range(grid_y.size))
#
#     for i, j in array_points:
#         A, B, E = gi.compute_repeating_terms(grid_x[i], grid_y[j], X,
#                                              Y, panel_geometry.phi)
#
#         M_xpj = gi.horizontal_geometric_integral_source(grid_x[i], X, panel_geometry.phi, panel_geometry.S,
#                                                         A, B, E)
#         M_ypj = gi.vertical_geometric_integral_source(grid_y[j], Y, panel_geometry.phi, panel_geometry.S,
#                                                       A, B, E)
#
#         del A, B, E
#         mask_M_xpj = np.where(np.isnan(M_xpj))
#         mask_M_ypj = np.where(np.isnan(M_ypj))
#         M_xpj[mask_M_xpj] = 0.
#         M_ypj[mask_M_ypj] = 0.
#         M_xpj = np.where(np.isnan(M_xpj) | np.isinf(M_xpj) | np.iscomplex(M_xpj), 0, M_xpj)
#         M_ypj = np.where(np.isnan(M_ypj) | np.isinf(M_ypj) | np.iscomplex(M_ypj), 0, M_ypj)
#         Mxpj[j, i], Mypj[j, i] = M_xpj, M_ypj
#
#     Mxpjf.close()
#     Mypjf.close()

def compute_grid_geometric_integrals_source_internal(panel_geometry: dc.PanelizedGeometry, grid_x: np.ndarray,
                                                     grid_y: np.ndarray, core_idx: int):
    """
    An internal function to compute the geometric integrals for the source panels.
    :param panel_geometry:
    :param grid_x:
    :param grid_y:
    :param core_idx:
    :return:
    """
    print(f'\tGrid Geometric integrals core {core_idx}')
    pathlib.Path.unlink(f'Mxpj_{core_idx}.h5', missing_ok=True)
    pathlib.Path.unlink(f'Mypj_{core_idx}.h5', missing_ok=True)
    Mxpjf = h5py.File(f'Mxpj_{core_idx}.h5', 'w')
    Mypjf = h5py.File(f'Mypj_{core_idx}.h5', 'w')
    Mxpj = Mxpjf.create_dataset('Mxpj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
    Mypj = Mypjf.create_dataset('Mypj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
    X = panel_geometry.xC - panel_geometry.S / 2 * np.cos(panel_geometry.phi)
    Y = panel_geometry.yC - panel_geometry.S / 2 * np.sin(panel_geometry.phi)

    for j in range(len(grid_y)):
        Mxpj[j, :], Mypj[j, :] = actual_computation_grid_geometric_integrals_source(grid_x, grid_y[j],
                                                                                    panel_geometry.S,
                                                                                    panel_geometry.phi, X, Y)

    Mxpjf.close()
    Mypjf.close()


@nb.njit(cache=True)
def actual_computation_grid_geometric_integrals_source(grid_x, grid_y, S, phi, X, Y):
    A, B, E = np.empty(len(S), dtype=float), np.empty(len(S), dtype=float), np.empty(len(S), dtype=float)
    M_xpj, M_ypj = np.empty((len(grid_x), len(S)), dtype=float), np.empty((len(grid_x), len(S)), dtype=float)
    for i in range(grid_x.size):
        A, B, E = gi.compute_repeating_terms(x_i=grid_x[i], y_i=grid_y, X_j=X, Y_j=Y, phi_j=phi)
        M_xpj[i, :] = gi.horizontal_geometric_integral_source(x_p=grid_x[i], X_j=X, phi_j=phi, S_j=S, A=A, B=B, E=E)
        M_ypj[i, :] = gi.vertical_geometric_integral_source(y_p=grid_y, Y_j=Y, phi_j=phi, S_j=S, A=A, B=B, E=E)

    M_xpj = np.where(np.isnan(M_xpj), 0., M_xpj)
    M_ypj = np.where(np.isnan(M_ypj), 0., M_ypj)

    M_xpj = np.where(np.isnan(M_xpj) | np.isinf(M_xpj) | np.iscomplex(M_xpj), 0, M_xpj)
    M_ypj = np.where(np.isnan(M_ypj) | np.isinf(M_ypj) | np.iscomplex(M_ypj), 0, M_ypj)
    return M_xpj, M_ypj


def compute_grid_geometric_integrals_vortex_internal(panel_geometry: dc.PanelizedGeometry, grid_x: np.ndarray,
                                                     grid_y: np.ndarray, core_idx: int):
    print(f'\tGrid Geometric integrals core {core_idx}')
    pathlib.Path.unlink(f'Nxpj_{core_idx}.h5', missing_ok=True)
    pathlib.Path.unlink(f'Nypj_{core_idx}.h5', missing_ok=True)
    Nxpjf = h5py.File(f'Nxpj_{core_idx}.h5', 'w')
    Nypjf = h5py.File(f'Nypj_{core_idx}.h5', 'w')
    Nxpj = Nxpjf.create_dataset('Nxpj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
    Nypj = Nypjf.create_dataset('Nypj', shape=(grid_y.size, grid_x.size, panel_geometry.S.size), dtype=np.float64)
    X = panel_geometry.xC - panel_geometry.S / 2 * np.cos(panel_geometry.phi)
    Y = panel_geometry.yC - panel_geometry.S / 2 * np.sin(panel_geometry.phi)
    array_points = product(range(grid_x.size), range(grid_y.size))
    for i, j in array_points:
        A, B, E = gi.compute_repeating_terms(grid_x[i], grid_y[j], X,
                                             Y, panel_geometry.phi)

        N_xpj = gi.horizontal_geometric_integral_vortex(grid_y[j], Y, panel_geometry.phi, panel_geometry.S,
                                                        A, B, E)
        N_ypj = gi.vertical_geometric_integral_vortex(grid_x[i], X, panel_geometry.phi, panel_geometry.S,
                                                      A, B, E)

        del A, B, E
        mask_N_xpj = np.where(np.isnan(N_xpj))
        mask_N_ypj = np.where(np.isnan(N_ypj))
        N_xpj[mask_N_xpj] = 0.
        N_ypj[mask_N_ypj] = 0.
        N_xpj = np.where(np.isnan(N_xpj) | np.isinf(N_xpj) | np.iscomplex(N_xpj), 0, N_xpj)
        N_ypj = np.where(np.isnan(N_ypj) | np.isinf(N_ypj) | np.iscomplex(N_ypj), 0, N_ypj)
        Nxpj[j, i], Nypj[j, i] = N_xpj, N_ypj

    Nxpjf.close()
    Nypjf.close()


def compute_grid_geometric_integrals_source_mp(panel_geometry: dc.PanelizedGeometry, grid_x: np.ndarray,
                                               grid_y: np.ndarray, num_cores: int = 1):
    """
    Compute the geometric integrals for the source panels.
    :param panel_geometry:
    :param grid_x:
    :param grid_y:
    :param num_cores:
    :return:
    """

    # Split up the panel geometry into smaller chunks
    yC_chunks = np.tile(panel_geometry.yC, (num_cores, 1))
    xC_chunks = np.tile(panel_geometry.xC, (num_cores, 1))
    S_chunks = np.tile(panel_geometry.S, (num_cores, 1))
    phi_chunks = np.tile(panel_geometry.phi, (num_cores, 1))
    delta_chunks = np.tile(panel_geometry.delta, (num_cores, 1))
    beta_chunks = np.tile(panel_geometry.beta, (num_cores, 1))

    # Create a list of new panel geometries
    panel_geometry_chunks = []
    for i in range(num_cores):
        panel_geometry_chunks.append(dc.PanelizedGeometry(S=S_chunks[i], phi=phi_chunks[i], delta=delta_chunks[i],
                                                          beta=beta_chunks[i], xC=xC_chunks[i], yC=yC_chunks[i]))

    # Create a list of new grid geometries
    grid_x_chunks = np.tile(grid_x, (num_cores, 1))
    grid_y_chunks = np.array_split(grid_y, num_cores)

    grid_geometry_chunks = []
    for i in range(num_cores):
        grid_geometry_chunks.append((grid_x_chunks[i], grid_y_chunks[i]))

    # for i, (chunked_panel_geometry, (chunked_grid_x, chucked_grid_y)) in enumerate(zip(panel_geometry_chunks, grid_geometry_chunks)):
    #
    #     compute_grid_geometric_integrals_source_internal(panel_geometry=chunked_panel_geometry, grid_x=chunked_grid_x,
    #                                                      grid_y=chucked_grid_y, core_idx=i)

    # Compute the geometric integrals for each chunk in parallel
    Parallel(n_jobs=num_cores)(
        delayed(compute_grid_geometric_integrals_source_internal)(panel_geometry=chunked_panel_geometry,
                                                                  grid_x=chunked_grid_x,
                                                                  grid_y=chucked_grid_y, core_idx=i) for
        i, (chunked_panel_geometry, (chunked_grid_x, chucked_grid_y)) in
        enumerate(zip(panel_geometry_chunks, grid_geometry_chunks)))


def compute_grid_geometric_integrals_vortex_mp(panel_geometry: dc.PanelizedGeometry, grid_x: np.ndarray,
                                               grid_y: np.ndarray, num_cores: int = 1):
    """
    Compute the geometric integrals for the source panels.
    :param panel_geometry:
    :param grid_x:
    :param grid_y:
    :param num_cores:
    :return:
    """

    # Split up the panel geometry into smaller chunks
    yC_chunks = np.tile(panel_geometry.yC, (num_cores, 1))
    xC_chunks = np.tile(panel_geometry.xC, (num_cores, 1))
    S_chunks = np.tile(panel_geometry.S, (num_cores, 1))
    phi_chunks = np.tile(panel_geometry.phi, (num_cores, 1))
    delta_chunks = np.tile(panel_geometry.delta, (num_cores, 1))
    beta_chunks = np.tile(panel_geometry.beta, (num_cores, 1))

    # Create a list of new panel geometries
    panel_geometry_chunks = []
    for i in range(num_cores):
        panel_geometry_chunks.append(dc.PanelizedGeometry(S=S_chunks[i], phi=phi_chunks[i], delta=delta_chunks[i],
                                                          beta=beta_chunks[i], xC=xC_chunks[i], yC=yC_chunks[i]))

    # Create a list of new grid geometries
    grid_x_chunks = np.tile(grid_x, (num_cores, 1))
    grid_y_chunks = np.array_split(grid_y, num_cores)

    grid_geometry_chunks = []
    for i in range(num_cores):
        grid_geometry_chunks.append((grid_x_chunks[i], grid_y_chunks[i]))

    # for i, (chunked_panel_geometry, (chunked_grid_x, chucked_grid_y)) in enumerate(zip(panel_geometry_chunks, grid_geometry_chunks)):
    #
    #     compute_grid_geometric_integrals_source_internal(panel_geometry=chunked_panel_geometry, grid_x=chunked_grid_x,
    #                                                      grid_y=chucked_grid_y, core_idx=i)

    # Compute the geometric integrals for each chunk in parallel
    Parallel(n_jobs=num_cores)(
        delayed(compute_grid_geometric_integrals_vortex_internal)(panel_geometry=chunked_panel_geometry,
                                                                  grid_x=chunked_grid_x,
                                                                  grid_y=chucked_grid_y, core_idx=i) for
        i, (chunked_panel_geometry, (chunked_grid_x, chucked_grid_y)) in
        enumerate(zip(panel_geometry_chunks, grid_geometry_chunks)))
