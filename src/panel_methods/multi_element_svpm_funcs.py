import numpy as np
from .spm_funcs import point_in_polygon
from . import geometric_integrals as gi
from ..code_collections import data_collections as dc
from ..multi_element_airfoil.airfoil_setup import create_clean_panelized_geometry
import numba as nb
import tqdm
from itertools import product

__all__ = ['run_source_vortex_panel_method_svpm', 'compute_cl_svpm']


@nb.njit(cache=True)
def compute_multi_element_source_vortex_strengths_nb(panelized_geometry: dc.PanelizedGeometryNb, V: float, I: np.ndarray,
                                                     J: np.ndarray, K: np.ndarray, L: np.ndarray, k_start: np.ndarray,
                                                     k_end: np.ndarray, num_airfoils: int, num_points: int):
    np.fill_diagonal(I, np.pi)
    np.fill_diagonal(J, 0)
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)

    A = np.zeros((len(I[0]) + num_airfoils, len(I[0]) + num_airfoils))
    b = np.zeros(len(I[0]) + num_airfoils)

    A[:-num_airfoils, :-num_airfoils] = I

    for i in range(len(K)):
        k_for_A = np.zeros(num_airfoils)
        for j in range(num_airfoils):
            k_for_A[j] = -np.sum(K[i, k_start[j]:k_end[j] + 1])
        A[i, -num_airfoils:] = k_for_A

    b[:-num_airfoils] = -np.cos(panelized_geometry.beta) * V * 2 * np.pi

    # Kutta condition
    A_kutta_condition = np.zeros((num_airfoils, len(A[0])))

    for i in range(num_airfoils):
        j_kutta = J[k_start[i]] + J[k_end[i]]
        l_kutta = -L[k_start[i]] - L[k_end[i]]
        A_kutta_condition[i, :-num_airfoils] = j_kutta
        A_kutta_condition[i, -num_airfoils + i] = np.sum(l_kutta[k_start[i]:k_end[i] + 1]) + 2 * np.pi

    b_kutta = np.zeros(num_airfoils)
    for i in range(num_airfoils):
        b_kutta[+i] = -V * 2 * np.pi * (np.sin(panelized_geometry.beta[k_start[i]]) +
                                        np.sin(panelized_geometry.beta[k_end[i]]))

    A[-num_airfoils:, :] = A_kutta_condition
    b[-num_airfoils:] = b_kutta

    solution = np.linalg.solve(A, b)
    lam = solution[:-num_airfoils]
    gamma = solution[-num_airfoils:]
    gamma = np.repeat(gamma, num_points - 1)

    return lam, gamma


@nb.njit(cache=True)
def compute_multi_element_grid_velocity_svpm_nb(discrete_geometries, panelized_geometry, x, y, lam, gamma,
                                                free_stream_velocity=1, AoA=0):
    Mxpj, Mypj = gi.compute_grid_geometric_integrals_source_nb(panel_geometry=panelized_geometry, grid_x=x, grid_y=y)
    Nxpj, Nypj = gi.compute_grid_geometric_integrals_vortex_nb(panelized_geometry, x, y)

    shapes = [np.vstack((geometry.x, geometry.y)).T for geometry in discrete_geometries]
    u = np.zeros((len(x), len(y)))
    v = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            for shape in shapes:
                if point_in_polygon(float(x[i]), float(y[j]), shape):
                    u[j, i] = 0
                    v[j, i] = 0
                    break
            else:
                u[j, i] = -np.sum(gamma * Nxpj[j, i] / (2 * np.pi)) + np.sum(
                    lam * Mxpj[j, i] / (2 * np.pi)) + free_stream_velocity * np.cos(AoA * np.pi / 180)
                v[j, i] = -np.sum(gamma * Nypj[j, i] / (2 * np.pi)) + np.sum(
                    lam * Mypj[j, i] / (2 * np.pi)) + free_stream_velocity * np.sin(AoA * np.pi / 180)
    return u, v


@nb.njit(cache=True)
def compute_panel_velocities_source_vortex_nb(panelized_geometry, lam, gamma, V, I, J, K, L):
    V_normal = np.empty(len(panelized_geometry.xC))
    V_tangential = np.empty(len(panelized_geometry.xC))

    for i in range(len(panelized_geometry.xC)):
        V_normal[i] = np.sum(-gamma[i] * K[i] / (2 * np.pi)) + np.sum(lam * I[i] / (2 * np.pi)) + V * np.cos(
            panelized_geometry.beta[i])
        V_tangential[i] = np.sum(-gamma[i] * L[i] / (2 * np.pi)) + np.sum(lam * J[i] / (2 * np.pi)) + V * np.sin(
            panelized_geometry.beta[i]) + gamma[i] / 2
    return V_normal, V_tangential


def compute_multi_element_grid_velocity_svpm(discrete_geometries, panelized_geometry, x, y, lam, gamma,
                                             free_stream_velocity=1, AoA=0):
    print('\t' + '-' * 60)
    Mxpj, Mypj = gi.compute_grid_geometric_integrals_source(panel_geometry=panelized_geometry, grid_x=x, grid_y=y)
    print('\t' + '-' * 60)
    Nxpj, Nypj = gi.compute_grid_geometric_integrals_vortex(panelized_geometry, x, y)
    print('-' * 64)
    print('Grid velocities')
    shapes = [np.vstack((geometry.x, geometry.y)).T for geometry in discrete_geometries]
    u = np.zeros((len(x), len(y)))
    v = np.zeros((len(x), len(y)))

    grid = product(range(len(x)), range(len(y)))
    for i, j in tqdm.tqdm(grid, total=len(x) * len(y), position=0):
        for shape in shapes:
            if point_in_polygon(float(x[i]), float(y[j]), shape):
                u[j, i] = 0
                v[j, i] = 0
                break
        else:
            u[j, i] = -np.sum(gamma * Nxpj[j, i] / (2 * np.pi)) + np.sum(
                lam * Mxpj[j, i] / (2 * np.pi)) + free_stream_velocity * np.cos(AoA * np.pi / 180)
            v[j, i] = -np.sum(gamma * Nypj[j, i] / (2 * np.pi)) + np.sum(
                lam * Mypj[j, i] / (2 * np.pi)) + free_stream_velocity * np.sin(AoA * np.pi / 180)

    del Mxpj, Mypj, Nxpj, Nypj

    return u, v


def run_source_vortex_panel_method_svpm(discrete_geometries, panelized_geometry: dc.PanelizedGeometryNb, x: np.ndarray,
                                        y: np.ndarray, num_airfoil: int, num_points, V: float, AoA: float,
                                        calc_velocities=True,
                                        use_memmap=False):
    try:
        I, J, K, L, lam, gamma = compute_panel(discrete_geometries, panelized_geometry, num_airfoil, num_points, V, AoA)
        print('Finished computing source vortex strengths')
        print('-' * 80)
        print('Beginning to compute panel velocities')
        V_normal, V_tangential = compute_panel_velocities_source_vortex_nb(panelized_geometry, lam, gamma, V, I, J, K,
                                                                           L)

        print('Finished computing panel velocities')
        print('-' * 80)
        if calc_velocities:
            print('Beginning to compute grid velocities')
            if use_memmap:
                u, v = compute_multi_element_grid_velocity_svpm(discrete_geometries, panelized_geometry, x, y, lam,
                                                                gamma,
                                                                V,
                                                                AoA)
            else:
                u, v = compute_multi_element_grid_velocity_svpm_nb(discrete_geometries, panelized_geometry, x, y, lam,
                                                                   gamma, V,
                                                                   AoA)
            print('-' * 80)

        else:
            u, v = np.zeros((len(x), len(y))), np.zeros((len(x), len(y)))

        panel_results = dc.SourceVortexPanelMethodResults(V_normal=V_normal, V_tangential=V_tangential,
                                                          Source_Strengths=lam, Circulation=gamma, V_horizontal=u,
                                                          V_vertical=v)
        return panel_results
    except Exception as e:
        raise e


def compute_panel(discrete_geometries, panelized_geometry: dc.PanelizedGeometryNb, num_airfoil: int, num_points, V: float,
                  AoA: float):
    k_start = np.cumsum(num_points - 1) - (num_points - 1)
    k_end = np.cumsum(num_points - 1) - 1
    print('-' * 80)
    print('Beginning to compute geometric integrals')
    I, J = gi.compute_panel_geometric_integrals_source_nb(panelized_geometry)
    K, L = gi.compute_panel_geometric_integrals_vortex_nb(panelized_geometry)
    print('Finished computing geometric integrals')
    print('-' * 80)
    print('Beginning to compute source vortex strengths')
    lam, gamma = compute_multi_element_source_vortex_strengths_nb(panelized_geometry, V, I, J, K, L, k_start, k_end,
                                                                  num_airfoil, num_points)
    print('Finished computing source vortex strengths')
    print('-' * 80)

    return I, J, K, L, lam, gamma


def compute_cl_svpm(multi_element_airfoil, airfoil_directory_path,  V: float, AoA: np.ndarray):

    cl1 = np.zeros_like(AoA)
    cl2 = np.zeros_like(AoA)
    cl3 = np.zeros_like(AoA)

    for i, aoa in enumerate(AoA):
        geometries, total_geometry, total_panelized_geometry = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                               airfoil_directory_path,
                                                                                               aoa)
        min_x = np.inf
        max_x = -np.inf
        for geometry in geometries:
            min_x = min(min_x, np.min(geometry.x))
            max_x = max(max_x, np.max(geometry.x))
        chordv2 = max_x - min_x
        chordV3 = np.sum([np.max(geometry.x) - np.min(geometry.x) for geometry in geometries])


        num_airfoils = len(geometries)

        num_points = np.array([len(geometry.x) for geometry in geometries])


        _, _, _, _, lam, gamma = compute_panel(geometries, total_panelized_geometry, num_airfoils, num_points, V, aoa)
        sumGamma = np.sum(gamma * total_panelized_geometry.S)
        print(sumGamma)
        cl1[i] = 2 * sumGamma
        cl2[i] = 2 * sumGamma / chordv2
        cl3[i] = 2 * sumGamma / chordV3

    return cl1, cl2, cl3
