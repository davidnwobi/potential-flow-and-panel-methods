import numpy as np
import matplotlib.pyplot as plt
from src.code_collections import Geometry
from src.util import PanelGenerator
from src.util import plot_panelized_geometry
from src.util import generate_four_digit_NACA

numB = 9  # Number of boundary points
tO = 22.5  # Angle offset [deg]
load = 'Circle'  # Load circle or airfoil
geometry = None
panelized_geometry = None
numPan = None
if load == 'Circle':
    theta = np.linspace(0, 360, numB)  # Angles to compute boundary points [deg]
    theta = theta + tO  # Add angle offset [deg]
    theta = theta * (np.pi / 180)  # Convert angle to radians [rad]

    # Boundary points
    XB = np.cos(theta)  # X value of boundary points
    YB = np.sin(theta)  # Y value of boundary points

    # Number of panels
    numPan = len(XB) - 1

    geometry = Geometry(XB, YB, 0)
    panelized_geometry = PanelGenerator.compute_geometric_quantities(geometry)

else:
    # Airfoil geometry
    data = generate_four_digit_NACA(num_NACA='2412', num_points=100, chord_length=1)
    XB = data[:, 0]
    YB = data[:, 1]

    # Number of panels
    numPan = len(XB) - 1
    geometry = Geometry(XB, YB, AoA=0)
    panelized_geometry = PanelGenerator.compute_geometric_quantities(geometry)

# %% PLOTTING

# Dashed circle defined
T = np.linspace(0, 2 * np.pi, 1000)  # Angle array to compute circle
x = np.cos(T)  # Circle X points
y = np.sin(T)  # Circle Y points

fig = plot_panelized_geometry(geometry, panelized_geometry)
if load == 'Circle':  # If circle is selected
    plt.plot(x, y, 'k--')  # Plot actual circle outline
plt.show()