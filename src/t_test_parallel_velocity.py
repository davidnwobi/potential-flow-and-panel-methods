import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.code_collections import Geometry, MultiElementAirfoil, Ellipse, FlowFieldProperties, PanelizedGeometry
from src.util import compute_ellipse_and_circulation
from src.multi_element_airfoil import create_clean_panelized_geometry
from src.panel_methods.p_multi_svpm.parallel_multi_element_svpm_funcs import run_panel_method
from src.util import PanelGenerator
from matplotlib import colors, ticker
from matplotlib.gridspec import GridSpec
import h5py
import multiprocessing

airfoil_directory_path = Path('../../Airfoil_DAT_Selig')
xfoil_usable_airfoil_directory_path = Path('../../xfoil_usable')

airfoils = ['0012']
load_NACA = [True]
num_points = np.array([811])
AF_flip = np.array([[1, 1]])
AF_scale = np.array([1])
AF_angle = np.array([0])
AF_offset = np.array([[0, 0]])

V = 1
AoA = 8
num_grid = 1111  # Change this if it is too slow or you run out of memory

X_NEG_LIMIT = -2
X_POS_LIMIT = 2
Y_NEG_LIMIT = -2
Y_POS_LIMIT = 2

need_velocity = True  # Set to False if you only want to see the geometry

# %% PANEL GEOMETRY SETUP
multi_element_airfoil = MultiElementAirfoil(airfoils=airfoils, load_NACA=load_NACA, num_points=num_points,
                                            AF_flip=AF_flip, AF_scale=AF_scale, AF_angle=AF_angle,
                                            AF_offset=AF_offset)

import pickle

geometries, total_geometry, total_panelized_geometry_nb = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                          airfoil_directory_path, AoA)
total_panelized_geometry = PanelizedGeometry.from_panelized_geometry_nb(total_panelized_geometry_nb)
num_airfoils = len(airfoils)
x, y = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, num_grid), np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT,
                                                                    num_grid)  # Create grid
n_jobs = 24
#
# V_normal_bench, V_tangential_bench, lam_bench, gamma_bench, u_bench, v_bench = run_source_vortex_panel_method_svpm(
#     geometries,
#     total_panelized_geometry_nb,
#     x=x,
#     y=y,
#     num_airfoil=num_airfoils,
#     num_points=num_points,
#     V=V, AoA=AoA,
#     calc_velocities=True)

V_normal, V_tangential, lam, gamma, u, v = run_panel_method(geometries,
                                                            total_panelized_geometry_nb,
                                                            x=x,
                                                            y=y,
                                                            num_airfoil=num_airfoils,
                                                            num_points=num_points,
                                                            V=V, AoA=AoA,
                                                            calc_velocities=True, use_memmap=True,
                                                            num_cores=n_jobs)
# print('u_bench: ')
# print(u_bench)
# print('u: ')
# print(u)
#
#
# assert np.allclose(V_normal_bench, V_normal)
# assert np.allclose(V_tangential_bench, V_tangential)
# assert np.allclose(u_bench, u)
# assert np.allclose(v_bench, v) # DEBUG THIS
# import timeit
#
# print(timeit.timeit(lambda: compute_grid_velocity_source_vortex_mp(panelized_geometry=total_panelized_geometry, x=x,
#                                                              y=y,
#                                                              lam=lam, gamma=gamma, free_stream_velocity=V,
#                                                              AoA=AoA, num_cores=n_jobs), number=10)/10)

# assert np.allclose(u_bench, u)
# import time
#
# start_time = time.time()
#
# compute_grid_geometric_integrals_source_mp(total_panelized_geometry, x, y, n_jobs)
#
# end_time = time.time()
# execution_time = end_time - start_time
# print(f'Execution time: {execution_time}')


# start_time = time.time()
#
# compute_grid_geometric_integrals_source(total_panelized_geometry, x, y)
#
# end_time = time.time()
# execution_time = end_time - start_time
# print(f'Execution time: {execution_time}')
