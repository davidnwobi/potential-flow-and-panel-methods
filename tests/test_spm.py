from src.util import PanelGenerator
from src.util import compute_ellipse_and_circulation
from src.panel_methods.spm import run_panel_method
from src.code_collections import data_collections as dc
from src.util import generate_four_digit_NACA
import numpy as np
import pytest


@pytest.fixture
def naca0012_test_zero_degrees():
    airfoil = '0012'
    AoA = 0
    V = 1
    XB, YB = generate_four_digit_NACA(num_NACA=airfoil, num_points=171, chord_length=1, b=2)
    geometry = dc.Geometry(x=XB, y=YB, AoA=AoA)
    panelized_geometry = PanelGenerator.compute_geometric_quantities(geometry=geometry)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)

    return panelized_geometry, V, AoA, x, y


@pytest.fixture
def naca2412_test_zero_degrees():
    airfoil = '0012'
    AoA = 0
    V = 1
    XB, YB = generate_four_digit_NACA(num_NACA=airfoil, num_points=300, chord_length=1, b=2)
    geometry = dc.Geometry(x=XB, y=YB, AoA=AoA)
    panelized_geometry = PanelGenerator.compute_geometric_quantities(geometry=geometry)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)

    return panelized_geometry, V, AoA, x, y

def compute_panel_quantities(panelized_geometry, V, AoA, x, y):
    V_normal, V_tangential, lam, u, v = run_panel_method(panelized_geometry=panelized_geometry, V=V, AoA=AoA,
                                                         x=x, y=y)
    sumLambda = np.sum(lam * panelized_geometry.S)
    panel_velocities = V_tangential ** 2
    Cp = 1 - panel_velocities / V ** 2
    CN = -Cp * np.sin(panelized_geometry.beta) * panelized_geometry.S  # Normal coefficient
    CA = -Cp * np.cos(panelized_geometry.beta) * panelized_geometry.S  # Axial coefficient

    CL = np.sum(CN * np.cos(AoA * np.pi / 180)) - np.sum(CA * np.sin(AoA * np.pi / 180))  # Lift coefficient
    airfoil_ellipse = dc.Ellipse(0.5, 0, 0.6, 0.2)
    flowfieldproperties = dc.FlowFieldProperties(x=x, y=y, u=u, v=v)
    circulation = compute_ellipse_and_circulation(flowfieldproperties, airfoil_ellipse, 1000)
    cl_circulation = 2 * circulation.circulation

    return V_normal, V_tangential, sumLambda, panel_velocities, CL, cl_circulation


def test_naca0012_zero_degrees(naca0012_test_zero_degrees):
    panelized_geometry, V, AoA, x, y = naca0012_test_zero_degrees
    V_normal, V_tangential, sumLambda, panel_velocities, CL, cl_circulation = compute_panel_quantities(
        panelized_geometry, V, AoA, x, y)
    assert np.isclose(sumLambda, 0, atol=1e-2)
    assert np.isclose(CL, cl_circulation, atol=1e-2)


