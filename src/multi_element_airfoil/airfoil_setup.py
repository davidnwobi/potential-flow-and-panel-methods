from ..util import interpolator
from ..util import generate_four_digit_NACA
from ..code_collections import Geometry, MultiElementAirfoil
from ..util import PanelGenerator
import numpy as np
from pathlib import Path
import typing as tp

__all__ = ['create_clean_panelized_geometry']

def load_airfoil(airfoil: str, num_points: int, load_NACA: bool,
                 airfoil_directory_path: tp.Union[str, Path] = Path('Airfoil_DAT_Selig')) -> np.ndarray:
    if load_NACA:
        airfoil_points = np.vstack(generate_four_digit_NACA(airfoil, num_points=num_points, chord_length=1)).T
    elif num_points == -1:
        airfoil_points = np.loadtxt(airfoil_directory_path / (airfoil + '.dat'), skiprows=1)
    else:
        airfoil_points = np.vstack(
            interpolator(airfoil_directory_path / (airfoil + '.dat'), num_points))
    return airfoil_points


def transform_airfoil(airfoil_points: np.ndarray, AF_scale: float, AF_angle: float, AF_offset: np.ndarray,
                      AF_flip: np.ndarray) -> np.ndarray:
    AF_angle = np.deg2rad(AF_angle)
    airfoil_points = airfoil_points * AF_scale

    rot_matrix = np.matrix([np.array([np.cos(AF_angle), -np.sin(AF_angle)]),
                            np.array([np.sin(AF_angle), np.cos(AF_angle)])])
    airfoil_points = np.array(rot_matrix * airfoil_points.T).T
    airfoil_points = airfoil_points * AF_flip
    airfoil_points = airfoil_points + AF_offset
    return airfoil_points


def setup_multi_element_airfoil(multi_element_airfoil: MultiElementAirfoil,
                                airfoil_directory_path: tp.Union[str, Path] = Path('Airfoil_DAT_Selig'),
                                AoA: float = 0) -> tp.List[Geometry]:
    airfoil_geometries = []

    for i, airfoil in enumerate(multi_element_airfoil.airfoils):
        airfoil_points = load_airfoil(airfoil, multi_element_airfoil.num_points[i], multi_element_airfoil.load_NACA[i],
                                      airfoil_directory_path)
        airfoil_points = transform_airfoil(airfoil_points, multi_element_airfoil.AF_scale[i],
                                           multi_element_airfoil.AF_angle[i], multi_element_airfoil.AF_offset[i],
                                           multi_element_airfoil.AF_flip[i])
        airfoil_geometries.append(Geometry(x=airfoil_points[:, 0], y=airfoil_points[:, 1], AoA=AoA))

    return airfoil_geometries


def clean_up_geometry(total_panelized_geometry, num_points):
    false_panels = np.cumsum(num_points)[:-1] - 1
    total_panelized_geometry.S = np.delete(total_panelized_geometry.S, false_panels, axis=0)
    total_panelized_geometry.phi = np.delete(total_panelized_geometry.phi, false_panels, axis=0)
    total_panelized_geometry.delta = np.delete(total_panelized_geometry.delta, false_panels, axis=0)
    total_panelized_geometry.beta = np.delete(total_panelized_geometry.beta, false_panels, axis=0)
    total_panelized_geometry.xC = np.delete(total_panelized_geometry.xC, false_panels, axis=0)
    total_panelized_geometry.yC = np.delete(total_panelized_geometry.yC, false_panels, axis=0)

    return total_panelized_geometry


def create_clean_panelized_geometry(multi_element_airfoil, airfoil_directory_path, AoA):
    """
    Create a clean panelized geometry for a multi-element airfoil

    Note: The geometries are presented in a clockwise manner hence for the discrete geometries, the order is reversed.

    :param multi_element_airfoil:
    :param airfoil_directory_path:
    :param AoA:
    :return:
    """
    geometries = setup_multi_element_airfoil(multi_element_airfoil, airfoil_directory_path, AoA=AoA)
    total_geometry = Geometry(np.concatenate([geometry.x for geometry in geometries]), np.concatenate(
        [geometry.y for geometry in geometries]), AoA)
    total_panelized_geometry = PanelGenerator.compute_geometric_quantities(total_geometry)
    total_panelized_geometry = clean_up_geometry(total_panelized_geometry, multi_element_airfoil.num_points)
    geometries = geometries[::-1]
    return geometries, total_geometry, total_panelized_geometry
