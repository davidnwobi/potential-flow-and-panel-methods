from . import parallel_geometric_integrals as pgi
from .spm_funcs import point_in_polygon
from ..code_collections import data_collections as dc
import numpy as np


def compute_grid_velocity_source_vortex_mp(panelized_geometry: dc.PanelizedGeometry, x: np.ndarray, y: np.ndarray,
                                           lam: np.ndarray,
                                           gamma: np.ndarray, free_stream_velocity: float = 1., AoA: float = 0.,
                                           num_cores: int = 1):
    Mxpj, Mypj = pgi.compute_grid_geometric_integrals_source_mp(panel_geometry=panelized_geometry, grid_x=x, grid_y=y)
    Nxpj, Nypj = pgi.compute_grid_geometric_integrals_vortex_mp(panel_geometry=panelized_geometry, grid_x=x, grid_y=y)

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
