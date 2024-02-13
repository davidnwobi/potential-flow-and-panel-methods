import pytest
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.code_collections import Geometry, MultiElementAirfoil, Ellipse, FlowFieldProperties
from src.util import compute_ellipse_and_circulation
from src.multi_element_airfoil import create_clean_panelized_geometry
from src.util import PanelGenerator
from src.panel_methods.multi_svpm import run_panel_method
from aeropy import xfoil_module as xf
from matplotlib import colors, ticker
import warnings


def get_limits(geometries, focus_scale=1.4, aspect_ratio_scaling=12):
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    for geometry in geometries:
        min_x = min(min_x, np.min(geometry.x))
        max_x = max(max_x, np.max(geometry.x))
        min_y = min(min_y, np.min(geometry.y))
        max_y = max(max_y, np.max(geometry.y))

    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    focus_scale = 1.4
    aspect_ratio_scaling = 12
    X_NEG_LIMIT = center_x + (min_x - center_x) * focus_scale
    X_POS_LIMIT = center_x + (max_x - center_x) * focus_scale
    Y_NEG_LIMIT = (center_y + (min_y - center_y) * focus_scale) * aspect_ratio_scaling
    Y_POS_LIMIT = (center_y + (max_y - center_y) * focus_scale) * aspect_ratio_scaling

    return X_NEG_LIMIT, X_POS_LIMIT, Y_NEG_LIMIT, Y_POS_LIMIT

@pytest.fixture
def naca0012_test_zero_degrees():
    airfoils = ['0012']
    load_NACA = [True]
    num_points = np.array([201])
    AF_flip = np.array([[1, 1]])
    AF_scale = np.array([1])
    AF_angle = np.array([0])
    AF_offset = np.array([[0, 0]])
    AoA = 0
    multi_element_airfoil = MultiElementAirfoil(airfoils=airfoils, load_NACA=load_NACA, num_points=num_points,
                                                AF_flip=AF_flip, AF_scale=AF_scale, AF_angle=AF_angle,
                                                AF_offset=AF_offset)

    geometries, total_geometry, total_panelized_geometry = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                           "", AoA)

    focus_scale = 1.4
    aspect_ratio_scaling = 12
    X_NEG_LIMIT, X_POS_LIMIT, Y_NEG_LIMIT, Y_POS_LIMIT = get_limits(geometries, focus_scale, aspect_ratio_scaling)
    x = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, 300)
    y = np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT, 300)

    return geometries, total_geometry, total_panelized_geometry, x, y

@pytest.fixture
def naca0012_test_8_degrees():
    airfoils = ['0012']
    load_NACA = [True]
    num_points = np.array([201])
    AF_flip = np.array([[1, 1]])
    AF_scale = np.array([1])
    AF_angle = np.array([0])
    AF_offset = np.array([[0, 0]])
    AoA = 8
    multi_element_airfoil = MultiElementAirfoil(airfoils=airfoils, load_NACA=load_NACA, num_points=num_points,
                                                AF_flip=AF_flip, AF_scale=AF_scale, AF_angle=AF_angle,
                                                AF_offset=AF_offset)

    geometries, total_geometry, total_panelized_geometry = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                           "", AoA)

    focus_scale = 1.4
    aspect_ratio_scaling = 12
    X_NEG_LIMIT, X_POS_LIMIT, Y_NEG_LIMIT, Y_POS_LIMIT = get_limits(geometries, focus_scale, aspect_ratio_scaling)
    x = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, 300)
    y = np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT, 300)

    return geometries, total_geometry, total_panelized_geometry, x, y

@pytest.fixture
def naca2412_test_zero_degrees():
    airfoils = ['2412']
    load_NACA = [True]
    num_points = np.array([201])
    AF_flip = np.array([[1, 1]])
    AF_scale = np.array([1])
    AF_angle = np.array([0])
    AF_offset = np.array([[0, 0]])
    AoA = 0
    multi_element_airfoil = MultiElementAirfoil(airfoils=airfoils, load_NACA=load_NACA, num_points=num_points,
                                                AF_flip=AF_flip, AF_scale=AF_scale, AF_angle=AF_angle,
                                                AF_offset=AF_offset)

    geometries, total_geometry, total_panelized_geometry = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                           "", AoA)

    focus_scale = 1.4
    aspect_ratio_scaling = 12
    X_NEG_LIMIT, X_POS_LIMIT, Y_NEG_LIMIT, Y_POS_LIMIT = get_limits(geometries, focus_scale, aspect_ratio_scaling)
    x = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, 300)
    y = np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT, 300)

    return geometries, total_geometry, total_panelized_geometry, x, y

@pytest.fixture
def naca2412_test_8_degrees():
    airfoils = ['2412']
    load_NACA = [True]
    num_points = np.array([201])
    AF_flip = np.array([[1, 1]])
    AF_scale = np.array([1])
    AF_angle = np.array([0])
    AF_offset = np.array([[0, 0]])
    AoA = 8
    multi_element_airfoil = MultiElementAirfoil(airfoils=airfoils, load_NACA=load_NACA, num_points=num_points,
                                                AF_flip=AF_flip, AF_scale=AF_scale, AF_angle=AF_angle,
                                                AF_offset=AF_offset)

    geometries, total_geometry, total_panelized_geometry = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                           "", AoA)

    focus_scale = 1.4
    aspect_ratio_scaling = 12
    X_NEG_LIMIT, X_POS_LIMIT, Y_NEG_LIMIT, Y_POS_LIMIT = get_limits(geometries, focus_scale, aspect_ratio_scaling)
    x = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, 300)
    y = np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT, 300)

    return geometries, total_geometry, total_panelized_geometry, x, y

@pytest.fixture
def naca0012_2_test_zero_degrees():
    airfoils = ['0012', '0012']
    load_NACA = [True, True]
    num_points = np.array([101, 101])
    AF_flip = np.array([[1, 1],
                        [1, 1]])
    AF_scale = np.array([1, 1])
    AF_angle = np.array([0, 0])
    AF_offset = np.array([[0, 0],
                          [1.2, 0]])
    AoA = 0
    multi_element_airfoil = MultiElementAirfoil(airfoils=airfoils, load_NACA=load_NACA, num_points=num_points,
                                                AF_flip=AF_flip, AF_scale=AF_scale, AF_angle=AF_angle,
                                                AF_offset=AF_offset)

    geometries, total_geometry, total_panelized_geometry = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                           "", AoA)

    focus_scale = 1.4
    aspect_ratio_scaling = 12
    X_NEG_LIMIT, X_POS_LIMIT, Y_NEG_LIMIT, Y_POS_LIMIT = get_limits(geometries, focus_scale, aspect_ratio_scaling)
    x = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, 300)
    y = np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT, 300)

    return geometries, total_geometry, total_panelized_geometry, x, y


@pytest.fixture
def naca0012_3_test_zero_degrees():
    airfoils = ['0012', '0012', '0012']
    load_NACA = [True, True, True]
    num_points = np.array([251, 251, 251])
    AF_flip = np.array([[1, 1],
                        [1, 1],
                        [1, 1]])
    AF_scale = np.array([1, 1, 1])
    AF_angle = np.array([0, 0, 0])
    AF_offset = np.array([[0, 0],
                          [1.2, 0],
                          [2.4, 0]])

    AoA = 0
    multi_element_airfoil = MultiElementAirfoil(airfoils=airfoils, load_NACA=load_NACA, num_points=num_points,
                                                AF_flip=AF_flip, AF_scale=AF_scale, AF_angle=AF_angle,
                                                AF_offset=AF_offset)

    geometries, total_geometry, total_panelized_geometry = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                           "", AoA)

    focus_scale = 1.4
    aspect_ratio_scaling = 12
    X_NEG_LIMIT, X_POS_LIMIT, Y_NEG_LIMIT, Y_POS_LIMIT = get_limits(geometries, focus_scale, aspect_ratio_scaling)
    x = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, 300)
    y = np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT, 300)

    return geometries, total_geometry, total_panelized_geometry, x, y


def compute_panel_quantities(geometries, total_geometry, total_panelized_geometry, V, AoA, x, y, num_airfoils,
                             num_points):
    V_normal, V_tangential, lam, gamma, u, v = run_panel_method(geometries, total_panelized_geometry,
                                                                x=x,
                                                                y=y,
                                                                num_airfoil=num_airfoils,
                                                                num_points=num_points,
                                                                V=V, AoA=AoA)
    sumLambda = np.sum(lam * total_panelized_geometry.S)
    sumGamma = np.sum(gamma * total_panelized_geometry.S)
    print('Sum of gamma: ', np.sum(gamma * total_panelized_geometry.S))
    print('Sum of lam: ', np.sum(lam * total_panelized_geometry.S))
    Cp = 1 - (V_tangential / V) ** 2

    min_x = np.min(total_geometry.x)
    min_y = np.min(total_geometry.y)
    max_x = np.max(total_geometry.x)
    max_y = np.max(total_geometry.y)

    a = np.abs(max_x - min_x) / np.sqrt(2)
    b = np.abs(max_y - min_y) / np.sqrt(2)
    x0 = (max_x + min_x) / 2
    y0 = (max_y + min_y) / 2
    airfoil_ellipse = Ellipse(x0, y0, a * 1.1, b * 1.1)
    flowfieldproperties = FlowFieldProperties(x=x, y=y, u=u, v=v)
    circulation = compute_ellipse_and_circulation(flowfieldproperties, airfoil_ellipse, 5000)

    chordv2 = max_x - min_x
    chordV3 = np.sum([np.max(geometry.x) - np.min(geometry.x) for geometry in geometries])

    return V_normal, V_tangential, sumLambda, sumGamma, Cp, circulation, chordv2, chordV3


def test_naca0012_zero_degrees(naca0012_test_zero_degrees):
    geometries, total_geometry, total_panelized_geometry, x, y = naca0012_test_zero_degrees
    V = 1
    AoA = 0
    num_airfoils = len(geometries)
    num_points = np.array([len(geometry.x) for geometry in geometries])
    V_normal, V_tangential, sumLambda, sumGamma, Cp, circulation, chordv2, chordV3 = compute_panel_quantities(
        geometries, total_geometry, total_panelized_geometry, V, AoA, x, y, num_airfoils, num_points)
    assert np.isclose(sumLambda, 0, atol=1e-2)
    assert np.isclose(2 * sumGamma, 2 * circulation.circulation, atol=1e-2)
    assert np.isclose(2 * sumGamma / chordv2, 2 * circulation.circulation / chordv2, atol=1e-2)
    assert np.isclose(2 * sumGamma / chordV3, 2 * circulation.circulation / chordV3, atol=1e-2)
    assert np.isclose(2 * sumGamma, 2 * sumGamma / chordv2, atol=1e-2)
    assert np.isclose(2 * sumGamma, 2 * sumGamma / chordV3, atol=1e-2)
    assert np.isclose(2 * circulation.circulation, 2 * circulation.circulation / chordv2, atol
    =1e-2)

def test_naca0012_8_degrees(naca0012_test_8_degrees):
    geometries, total_geometry, total_panelized_geometry, x, y = naca0012_test_8_degrees
    V = 1
    AoA = 8
    num_airfoils = len(geometries)
    num_points = np.array([len(geometry.x) for geometry in geometries])
    V_normal, V_tangential, sumLambda, sumGamma, Cp, circulation, chordv2, chordV3 = compute_panel_quantities(
        geometries, total_geometry, total_panelized_geometry, V, AoA, x, y, num_airfoils, num_points)

    x_foil_cl = xf.find_coefficients(airfoil='naca0012', alpha=AoA, NACA=True, delete=True)['CL']
    assert np.isclose(sumLambda, 0, atol=1e-2)
    print(x_foil_cl, 2 * circulation.circulation)
    assert np.isclose(2 * circulation.circulation, x_foil_cl, rtol=0.05)
    assert np.isclose(2 * circulation.circulation / chordv2, x_foil_cl, rtol=0.05)
    assert np.isclose(2 * circulation.circulation / chordV3, x_foil_cl, rtol=0.05)

def test_naca2412_zero_degrees(naca2412_test_zero_degrees):
    geometries, total_geometry, total_panelized_geometry, x, y = naca2412_test_zero_degrees
    V = 1
    AoA = 0
    num_airfoils = len(geometries)
    num_points = np.array([len(geometry.x) for geometry in geometries])
    V_normal, V_tangential, sumLambda, sumGamma, Cp, circulation, chordv2, chordV3 = compute_panel_quantities(
        geometries, total_geometry, total_panelized_geometry, V, AoA, x, y, num_airfoils, num_points)
    x_foil_cl = xf.find_coefficients(airfoil='naca2412', alpha=AoA, NACA=True, delete=True)['CL']
    assert np.isclose(sumLambda, 0, atol=1e-2)
    assert np.isclose(2 * circulation.circulation, x_foil_cl, rtol=0.05)
    assert np.isclose(2 * circulation.circulation / chordv2, x_foil_cl, rtol=0.05)
    assert np.isclose(2 * circulation.circulation / chordV3, x_foil_cl, rtol=0.05)


def test_naca2412_8_degrees(naca2412_test_8_degrees):
    geometries, total_geometry, total_panelized_geometry, x, y = naca2412_test_8_degrees
    V = 1
    AoA = 8
    num_airfoils = len(geometries)
    num_points = np.array([len(geometry.x) for geometry in geometries])
    V_normal, V_tangential, sumLambda, sumGamma, Cp, circulation, chordv2, chordV3 = compute_panel_quantities(
        geometries, total_geometry, total_panelized_geometry, V, AoA, x, y, num_airfoils, num_points)

    x_foil_cl = xf.find_coefficients(airfoil='naca2412', alpha=AoA, NACA=True, delete=True)['CL']
    assert np.isclose(sumLambda, 0, atol=1e-2)
    assert np.isclose(2 * circulation.circulation, x_foil_cl, rtol=0.05)
    assert np.isclose(2 * circulation.circulation / chordv2, x_foil_cl, rtol=0.05)
    assert np.isclose(2 * circulation.circulation / chordV3, x_foil_cl, rtol=0.05)

def test_naca0012_2_zero_degrees(naca0012_2_test_zero_degrees):
    geometries, total_geometry, total_panelized_geometry, x, y = naca0012_2_test_zero_degrees
    V = 1
    AoA = 0
    num_airfoils = len(geometries)
    num_points = np.array([len(geometry.x) for geometry in geometries])
    V_normal, V_tangential, sumLambda, sumGamma, Cp, circulation, chordv2, chordV3 = compute_panel_quantities(
        geometries, total_geometry, total_panelized_geometry, V, AoA, x, y, num_airfoils, num_points)
    assert np.isclose(sumLambda, 0, atol=1e-2)
    assert np.isclose(2 * sumGamma, 2 * circulation.circulation, atol=1e-2)
    assert np.isclose(2 * sumGamma / chordv2, 2 * circulation.circulation / chordv2, atol=1e-2)
    assert np.isclose(2 * sumGamma / chordV3, 2 * circulation.circulation / chordV3, atol=1e-2)
    assert np.isclose(2 * sumGamma, 2 * sumGamma / chordv2, atol=1e-2)
    assert np.isclose(2 * sumGamma, 2 * sumGamma / chordV3, atol=1e-2)
    assert np.isclose(2 * circulation.circulation, 2 * circulation.circulation / chordv2, atol
    =1e-2)


def test_naca0012_3_zero_degrees(naca0012_3_test_zero_degrees):
    geometries, total_geometry, total_panelized_geometry, x, y = naca0012_3_test_zero_degrees
    V = 1
    AoA = 0
    num_airfoils = len(geometries)
    num_points = np.array([len(geometry.x) for geometry in geometries])
    V_normal, V_tangential, sumLambda, sumGamma, Cp, circulation, chordv2, chordV3 = compute_panel_quantities(
        geometries, total_geometry, total_panelized_geometry, V, AoA, x, y, num_airfoils, num_points)
    assert np.isclose(sumLambda, 0, atol=1e-2)
    assert np.isclose(2 * sumGamma, 2 * circulation.circulation, atol=1e-2)
    assert np.isclose(2 * sumGamma / chordv2, 2 * circulation.circulation / chordv2, atol=1e-2)
    assert np.isclose(2 * sumGamma / chordV3, 2 * circulation.circulation / chordV3, atol=1e-2)
    assert np.isclose(2 * sumGamma, 2 * sumGamma / chordv2, atol=1e-2)
    assert np.isclose(2 * sumGamma, 2 * sumGamma / chordV3, atol=1e-2)
    assert np.isclose(2 * circulation.circulation, 2 * circulation.circulation / chordv2, atol
    =1e-2)