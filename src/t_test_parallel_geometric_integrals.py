import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.code_collections import Geometry, MultiElementAirfoil, Ellipse, FlowFieldProperties, PanelizedGeometry
from src.useful import compute_ellipse_and_circulation
from src.multi_element_airfoil import create_clean_panelized_geometry
from src.useful import PanelGenerator
from src.panel_methods import compute_grid_geometric_integrals_source_mp, compute_grid_geometric_integrals_source
from matplotlib import colors, ticker
from matplotlib.gridspec import GridSpec
import h5py
import multiprocessing

airfoil_directory_path = Path('../../Airfoil_DAT_Selig')
xfoil_usable_airfoil_directory_path = Path('../../xfoil_usable')

airfoils = ['0012']
load_NACA = [True]
num_points = np.array([501])
AF_flip = np.array([[1, 1]])
AF_scale = np.array([1])
AF_angle = np.array([0])
AF_offset = np.array([[0, 0]])

V = 1
AoA = 8
num_grid = 2000  # Change this if it is too slow or you run out of memory

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
print(type(total_panelized_geometry))
num_airfoils = len(airfoils)
x, y = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, num_grid), np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT,
                                                                    num_grid)  # Create grid
n_jobs = 24

# compute_grid_geometric_integrals_source_mp(total_panelized_geometry, x, y, n_jobs)
# benchmark = compute_grid_geometric_integrals_source(total_panelized_geometry, x, y)
# benchmark_array = np.empty((len(y), len(x), len(total_panelized_geometry.S)), dtype=float)
# benchmark_array[:, :, :] = benchmark[0]
# del benchmark
# print((len(y), len(x), len(total_panelized_geometry.S)))
# source_grid_integrals = np.empty((len(y), len(x), len(total_panelized_geometry.S)), dtype=float)
# i = 0
# j = 0
# for core_idx in range(n_jobs):
#     with h5py.File(f'Mxpj_{core_idx}.h5', 'r') as f:
#         dset = f['Mxpj']
#         j += len(dset)
#         source_grid_integrals[i:j] = dset
#         i = j



# assert np.array_equal(benchmark_array, source_grid_integrals)

import time

start_time = time.time()

compute_grid_geometric_integrals_source_mp(total_panelized_geometry, x, y, n_jobs)

end_time = time.time()
execution_time = end_time - start_time
print(f'Execution time: {execution_time}')


# start_time = time.time()
#
# compute_grid_geometric_integrals_source(total_panelized_geometry, x, y)
#
# end_time = time.time()
# execution_time = end_time - start_time
# print(f'Execution time: {execution_time}')