import matplotlib.pyplot as plt
from src.useful import PanelGenerator
from src.panel_methods import run_source_panel_method
import src.code_collections.data_collections as dc
import numpy as np
import pandas as pd
import time

time_start = time.time()
X_NEG_LIMIT = -2
X_POS_LIMIT = 2
Y_NEG_LIMIT = -2
Y_POS_LIMIT = 2
numB = 200
num_grid = 100
AoA = 0
V = 1
x = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, num=num_grid)
y = np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT, num=num_grid)

# %% Define Circle Geometry

radius = 1
theta = np.linspace(0, 360, numB)  # Angles to compute boundary points [deg]
theta = theta * (np.pi / 180)  # Convert angle to radians [rad]

# Boundary points of circle
XB = np.cos(theta) * radius  # X value of boundary points
YB = np.sin(theta) * radius  # Y value of boundary points

# %% Preprocess
geometry = dc.Geometry(x=XB, y=YB, AoA=AoA)
panelized_geometry = PanelGenerator.compute_geometric_quantities(geometry=geometry)
# %% Actual Computation
panel_methods_result = run_source_panel_method(panelized_geometry=panelized_geometry, V=V, AoA=AoA, x=x, y=y)
V_normal, V_tangential, lam, u, v = panel_methods_result

# %% Manipulate Results

# Sanity Check
sumLambda = sum(lam * panelized_geometry.S)
print("Sum of Lamdba: ", sumLambda)  # Should be a very small number

# For later use when plotting
X = geometry.x
Y = geometry.y
panel_normal_vector_X = panelized_geometry.xC + panelized_geometry.S / 2 * np.cos(panelized_geometry.delta)
panel_normal_vector_Y = panelized_geometry.yC + panelized_geometry.S / 2 * np.sin(panelized_geometry.delta)

# Velocity everywhere in the grid
local_v = np.sqrt(u ** 2 + v ** 2)

# Calculate Cp
cp = 1 - (local_v / V) ** 2
panel_velocities = V_tangential ** 2
theoretical_Cp = lambda angle: 1 - 4 * (np.sin(angle)) ** 2
theta = np.linspace(0, 2 * np.pi, 100)
Cp = 1 - panel_velocities / V ** 2

# Calculate Lift and Drag
CN = -Cp * np.sin(panelized_geometry.beta) * panelized_geometry.S  # Normal coefficient
CA = -Cp * np.cos(panelized_geometry.beta) * panelized_geometry.S  # Axial coefficient

CL = np.sum(CN * np.cos(AoA * np.pi / 180)) - np.sum(CA * np.sin(AoA * np.pi / 180))  # Lift coefficient
CD = np.sum(CN * np.sin(AoA * np.pi / 180)) + np.sum(CA * np.cos(AoA * np.pi / 180))  # Drag coefficient

time_end = time.time()
print("Time: ", time_end - time_start)

# %% Plotting
fig, axs = plt.subplots(2, 3, figsize=(30, 20), dpi=200, layout="constrained")
fig.delaxes(axs[1, 2])
# Panel Geometry
axs[0, 0].set_title('Panel Geometry')
for i in range(len(panelized_geometry.S)):
    axs[0, 0].plot([X[i], X[i + 1]], [Y[i], Y[i + 1]], 'k')
    axs[0, 0].plot([X[i]], [Y[i]], 'ro')
    axs[0, 0].plot([panelized_geometry.xC[i]], [panelized_geometry.yC[i]], 'bo')

    axs[0, 0].plot([panelized_geometry.xC[i], panel_normal_vector_X[i]],
                   [panelized_geometry.yC[i], panel_normal_vector_Y[i]],
                   'k')
    axs[0, 0].plot([panel_normal_vector_X[i]], [panel_normal_vector_Y[i]], 'go')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
axs[0, 0].set_aspect('equal', adjustable='box')

# Streamlines
axs[0, 1].set_title('Streamlines')
axs[0, 1].streamplot(x, y, u, v, density=2, color='b')
axs[0, 1].set_xlim(X_NEG_LIMIT, X_POS_LIMIT)
axs[0, 1].set_ylim(Y_NEG_LIMIT, Y_POS_LIMIT)
axs[0, 1].fill(geometry.x, geometry.y, 'k')  # Plot polygon (circle or airfoil)
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
axs[0, 1].set_aspect('equal', adjustable='box')

# Pressure Contours
axs[0, 2].set_title('Pressure Contours')
cp_plot = axs[0, 2].contourf(x, y, cp, 100, cmap='jet')
plt.colorbar(cp_plot, ax=axs[0, 2], label="Pressure Coefficient")
axs[0, 2].fill(geometry.x, geometry.y, 'k')
axs[0, 2].set_xlabel('X')
axs[0, 2].set_ylabel('Y')
axs[0, 2].set_aspect('equal', adjustable='box')

# Cp Distribution
axs[1, 0].set_title('Cp Distribution')
axs[1, 0].plot(theta, theoretical_Cp(theta), 'k', label='Theoretical')
axs[1, 0].plot(panelized_geometry.beta, Cp, 'ro', label='Panel Method')
axs[1, 0].set_xlabel('Theta [rad]')
axs[1, 0].set_ylabel('Cp')
axs[1, 0].legend()

# Results Table
axs[1, 1].axis('off')
table_data = pd.DataFrame({
    "CL": [round(CL, 6)],
    "CD": [round(CD, 6)],
    "Sum of Source Strengths": [round(sumLambda, 6)]
})
table_data = table_data.round(6)
table_data = table_data.T
row_labels = table_data.index.values
table = axs[1, 1].table(cellText=table_data.values,
                        rowLabels=row_labels,
                        loc='center',
                        cellLoc='center',
                        bbox=[0.4, 0.6, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0])  # Adjust column width
axs[1, 1].set_title('Results', fontsize=12, pad=20)  # Use 'pad' to adjust the distance from the top
axs[1, 1].grid(False)
for key, cell in table.get_celld().items():
    cell.set_linewidth(0.5)

plt.show()


