from . import parallel_geometric_integrals as pgi
from ..utils import point_in_polygon
from ...code_collections import data_collections as dc
from ..multi_svpm.multi_element_svpm_funcs import compute_panel, compute_panel_velocities_source_vortex_nb
import numpy as np
import h5py
import numba as nb
import pathlib
from joblib import Parallel, delayed

__all__ = ['compute_grid_velocity_source_vortex_mp', 'run_source_vortex_panel_method_svpm_mp']


def compute_grid_velocity_source_vortex_mp(discrete_geometries, panelized_geometry: dc.PanelizedGeometry, x: np.ndarray,
                                           y: np.ndarray,
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

    from time import time
    time1 = time()

    shapes = [np.vstack((geometry.x, geometry.y)).T for geometry in discrete_geometries]

    out = Parallel(n_jobs=num_cores)(
        delayed(compute_grid_velocity_source_vortex)(i=i, shapes=shapes, panelized_geometry=chunked_panel_geometry,
                                                  x=chunked_grid_x,
                                                  y=chucked_grid_y,
                                                  lam=lam, gamma=gamma, free_stream_velocity=free_stream_velocity,
                                                  AoA=AoA,
                                                  core_idx=i) for
        i, (chunked_panel_geometry, (chunked_grid_x, chucked_grid_y)) in
        enumerate(zip(panel_geometry_chunks, grid_geometry_chunks)))
    out.sort(key=lambda x: x[0])
    # for i, (chunked_panel_geometry, (chunked_grid_x, chucked_grid_y)) in enumerate(
    #         zip(panel_geometry_chunks, grid_geometry_chunks)):
    #     out = compute_grid_velocity_source_vortex(i=i, shapes=shapes, panelized_geometry=chunked_panel_geometry,
    #                                               x=chunked_grid_x,
    #                                               y=chucked_grid_y,
    #                                               lam=lam, gamma=gamma, free_stream_velocity=free_stream_velocity,
    #                                               AoA=AoA,
    #                                               core_idx=i)

    for i in range(num_cores):
        u.append(out[i][1])
        v.append(out[i][2])

    time2 = time()
    print(f'Elapsed time: {time2 - time1}')
    return np.vstack(u), np.vstack(v)


def compute_grid_velocity_source_vortex(i, shapes, panelized_geometry: dc.PanelizedGeometry, x: np.ndarray, y: np.ndarray,
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
    u = np.zeros((len(y), len(x)))
    v = np.zeros((len(y), len(x)))

    for j in range(len(y)):
        u[j], v[j] = actual_grid_velocity_computation(shapes, x, y[j], lam, gamma, free_stream_velocity, AoA,
                                                      np.array(Mxpj[j]), np.array(Mypj[j]), np.array(Nxpj[j]),
                                                      np.array(Nypj)[j])
    del Mxpj, Mypj, Nxpj, Nypj
    pathlib.Path(f'Mxpj_{core_idx}.h5', missing_ok=True).unlink()
    pathlib.Path(f'Mypj_{core_idx}.h5', missing_ok=True).unlink()
    pathlib.Path(f'Nxpj_{core_idx}.h5', missing_ok=True).unlink()
    pathlib.Path(f'Nypj_{core_idx}.h5', missing_ok=True).unlink()

    return i, u, v


@nb.njit(cache=True)
def actual_grid_velocity_computation(shapes, x, y, lam, gamma, free_stream_velocity, AoA, Mxpj, Mypj, Nxpj, Nypj):
    u = np.empty(len(x))
    v = np.empty(len(x))
    for i in range(len(x)):
        for shape in shapes:
            if point_in_polygon(float(x[i]), float(y), shape):
                u[i] = 0
                v[i] = 0
                break
        else:
            u[i] = -np.sum(gamma * Nxpj[i] / (2 * np.pi)) + np.sum(
                lam * Mxpj[i] / (2 * np.pi)) + free_stream_velocity * np.cos(AoA * np.pi / 180)
            v[i] = -np.sum(gamma * Nypj[i] / (2 * np.pi)) + np.sum(
                lam * Mypj[i] / (2 * np.pi)) + free_stream_velocity * np.sin(AoA * np.pi / 180)

    return u, v


def run_source_vortex_panel_method_svpm_mp(discrete_geometries, panelized_geometry: dc.PanelizedGeometryNb,
                                           x: np.ndarray,
                                           y: np.ndarray, num_airfoil: int, num_points, V: float, AoA: float,
                                           calc_velocities=True,
                                           use_memmap=True, num_cores=1):
    try:
        I, J, K, L, lam, gamma = compute_panel(discrete_geometries, panelized_geometry, num_airfoil, num_points, V, AoA)
        print('Finished computing source vortex strengths')
        print('-' * 80)
        print('Beginning to compute panel velocities')
        V_normal, V_tangential = compute_panel_velocities_source_vortex_nb(panelized_geometry, lam, gamma, V, I, J, K,
                                                                           L)

        print('Finished computing panel velocities')
        print('-' * 80)
        from time import time
        time1 = time()
        if calc_velocities:
            print('Beginning to compute grid velocities')
            if use_memmap:

                u, v = compute_grid_velocity_source_vortex_mp(discrete_geometries,
                                                              dc.PanelizedGeometry.from_panelized_geometry_nb(
                                                                  panelized_geometry), x, y, lam,
                                                              gamma,
                                                              V,
                                                              AoA, num_cores=num_cores)
            else:
                u, v = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
            print('-' * 80)

        else:
            u, v = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))
        time2 = time()
        print(f'Elapsed time: {time2 - time1}')
        panel_results = dc.SourceVortexPanelMethodResults(V_normal=V_normal, V_tangential=V_tangential,
                                                          Source_Strengths=lam, Circulation=gamma, V_horizontal=u,
                                                          V_vertical=v)
        return panel_results
    except Exception as e:
        raise e
