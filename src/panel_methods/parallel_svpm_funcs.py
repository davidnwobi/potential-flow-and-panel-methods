from . import parallel_geometric_integrals as pgi
from .spm_funcs import point_in_polygon
from ..code_collections import data_collections as dc
import numpy as np
import h5py

__all__ = ['compute_grid_velocity_source_vortex_mp']


def compute_grid_velocity_source_vortex_mp(panelized_geometry: dc.PanelizedGeometry, x: np.ndarray, y: np.ndarray,
                                           lam: np.ndarray,
                                           gamma: np.ndarray, free_stream_velocity: float = 1., AoA: float = 0.,
                                           num_cores: int = 1):
    pgi.compute_grid_geometric_integrals_source_mp(panel_geometry=panelized_geometry, grid_x=x, grid_y=y,
                                                   num_cores=num_cores)
    pgi.compute_grid_geometric_integrals_vortex_mp(panel_geometry=panelized_geometry, grid_x=x, grid_y=y,
                                                   num_cores=num_cores)

    # Split up the panel geometry into smaller chunks
    yC_chunks = np.tile(panelized_geometry.yC, (num_cores, 1))
    xC_chunks = np.tile(panelized_geometry.xC, (num_cores, 1))
    S_chunks = np.tile(panelized_geometry.S, (num_cores, 1))
    phi_chunks = np.tile(panelized_geometry.phi, (num_cores, 1))
    delta_chunks = np.tile(panelized_geometry.delta, (num_cores, 1))
    beta_chunks = np.tile(panelized_geometry.beta, (num_cores, 1))

    # Create a list of new panel geometries
    panel_geometry_chunks = []
    for i in range(num_cores):
        panel_geometry_chunks.append(dc.PanelizedGeometry(S=S_chunks[i], phi=phi_chunks[i], delta=delta_chunks[i],
                                                          beta=beta_chunks[i], xC=xC_chunks[i], yC=yC_chunks[i]))

    # Create a list of new grid geometries
    grid_x_chunks = np.tile(x, (num_cores, 1))
    grid_y_chunks = np.array_split(y, num_cores)

    grid_geometry_chunks = []
    for i in range(num_cores):
        grid_geometry_chunks.append((grid_x_chunks[i], grid_y_chunks[i]))

    u = []
    v = []

    for i, (chunked_panel_geometry, (chunked_grid_x, chucked_grid_y)) in enumerate(
            zip(panel_geometry_chunks, grid_geometry_chunks)):
        out = compute_grid_velocity_source_vortex(panelized_geometry=chunked_panel_geometry, x=chunked_grid_x,
                                                  y=chucked_grid_y,
                                                  lam=lam, gamma=gamma, free_stream_velocity=free_stream_velocity,
                                                  AoA=AoA,
                                                  core_idx=i)
        u.append(out[0])
        v.append(out[1])

    return np.vstack(u), np.vstack(v)


def compute_grid_velocity_source_vortex(panelized_geometry: dc.PanelizedGeometry, x: np.ndarray, y: np.ndarray,
                                        lam: np.ndarray, gamma: np.ndarray, free_stream_velocity: float = 1.,
                                        AoA: float = 0., core_idx: int = 0):
    Mxpj = h5py.File(f'Mxpj_{core_idx}.h5', 'r')['Mxpj']
    Mypj = h5py.File(f'Mypj_{core_idx}.h5', 'r')['Mypj']
    Nxpj = h5py.File(f'Nxpj_{core_idx}.h5', 'r')['Nxpj']
    Nypj = h5py.File(f'Nypj_{core_idx}.h5', 'r')['Nypj']

    X = panelized_geometry.xC - panelized_geometry.S / 2 * np.cos(panelized_geometry.phi)
    Y = panelized_geometry.yC - panelized_geometry.S / 2 * np.sin(panelized_geometry.phi)
    X = np.append(X, X[0])
    Y = np.append(Y, Y[0])
    shape = np.vstack((X, Y)).T
    u = np.zeros((len(x), len(y)))
    v = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            if point_in_polygon(float(x[i]), float(y[j]), shape):
                u[j, i] = 0
                v[j, i] = 0
            else:
                u[j, i] = -np.sum(gamma * Nxpj[j, i] / (2 * np.pi)) + np.sum(
                    lam * Mxpj[j, i] / (2 * np.pi)) + free_stream_velocity * np.cos(AoA * np.pi / 180)
                v[j, i] = -np.sum(gamma * Nypj[j, i] / (2 * np.pi)) + np.sum(
                    lam * Mypj[j, i] / (2 * np.pi)) + free_stream_velocity * np.sin(AoA * np.pi / 180)
    return u, v
