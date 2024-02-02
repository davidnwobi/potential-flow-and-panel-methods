import pytest
import numpy as np
from pathlib import Path
from src.code_collections import Geometry, MultiElementAirfoil, PanelizedGeometry
from src.multi_element_airfoil import create_clean_panelized_geometry
from src.panel_methods import run_source_vortex_panel_method_svpm, compute_grid_velocity_source_vortex_mp
import h5py

@pytest.fixture
def default_data(request):
    airfoil_directory_path = Path('../../Airfoil_DAT_Selig')
    xfoil_usable_airfoil_directory_path = Path('../../xfoil_usable')

    airfoils = ['0012']
    load_NACA = [True]
    num_points = np.array([101])
    AF_flip = np.array([[1, 1]])
    AF_scale = np.array([1])
    AF_angle = np.array([0])
    AF_offset = np.array([[0, 0]])

    V = 1
    AoA = 8
    num_grid = 100  # Change this if it is too slow or you run out of memory

    X_NEG_LIMIT = -2
    X_POS_LIMIT = 2
    Y_NEG_LIMIT = -2
    Y_POS_LIMIT = 2

    need_velocity = True  # Set to False if you only want to see the geometry

    multi_element_airfoil = MultiElementAirfoil(airfoils=airfoils, load_NACA=load_NACA, num_points=num_points,
                                                AF_flip=AF_flip, AF_scale=AF_scale, AF_angle=AF_angle,
                                                AF_offset=AF_offset)

    import pickle

    geometries, total_geometry, total_panelized_geometry_nb = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                              airfoil_directory_path,
                                                                                              AoA)
    total_panelized_geometry = PanelizedGeometry.from_panelized_geometry_nb(total_panelized_geometry_nb)
    num_airfoils = len(airfoils)
    x, y = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, num_grid), np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT,
                                                                        num_grid)  # Create grid
    n_jobs = 24

    yield geometries, total_geometry, total_panelized_geometry_nb, total_panelized_geometry, num_points, num_airfoils, x, y, n_jobs


def test_parallel_velocity(default_data):
    V = 1
    AoA = 0

    geometries, total_geometry, total_panelized_geometry_nb, total_panelized_geometry, num_points, num_airfoils, x, y, n_jobs = default_data

    V_normal, V_tangential, lam, gamma, u_bench, v_bench = run_source_vortex_panel_method_svpm(geometries,
                                                                                               total_panelized_geometry_nb,
                                                                                               x=x,
                                                                                               y=y,
                                                                                               num_airfoil=num_airfoils,
                                                                                               num_points=num_points,
                                                                                               V=V, AoA=AoA)

    u, v = compute_grid_velocity_source_vortex_mp(panelized_geometry=total_panelized_geometry, x=x,
                                                  y=y,
                                                  lam=lam, gamma=gamma, free_stream_velocity=V,
                                                  AoA=AoA, num_cores=n_jobs)

    np.testing.assert_allclose(u, u_bench)
    np.testing.assert_allclose(v, v_bench)
