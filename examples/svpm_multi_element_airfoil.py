import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.code_collections import Geometry, MultiElementAirfoil, Ellipse, FlowFieldProperties
from src.useful import compute_ellipse_and_circulation
from src.multi_element_airfoil import create_clean_panelized_geometry
from src.useful import PanelGenerator
from src.panel_methods import run_source_vortex_panel_method_svpm
from matplotlib import colors, ticker
import warnings

airfoil_directory_path = Path('../Airfoil_DAT_Selig')
xfoil_usable_airfoil_directory_path = Path('../xfoil_usable')
#%% BELOW ARE VARIOUS CONFIGURATIONS FOR THE MULTI-ELEMENT AIRFOIL YOU CAN PLAY WITH


# airfoils = ['2412']
# load_NACA = [True]
# num_points = np.array([171])
# AF_flip = np.array([[1, 1]])
# AF_scale = np.array([1])
# AF_angle = np.array([-6])
# AF_offset = np.array([[0, 0]])

# airfoils = ['0012', '0012']
# load_NACA = [True, True]
# num_points = np.array([101, 101])
# AF_flip = np.array([[1, 1],
#                     [1, 1]])
# AF_scale = np.array([1, 1])
# AF_angle = np.array([0, 0])
# AF_offset = np.array([[0, 0],
#                       [1.2, 0]])

# airfoils = ['0012', '0012', '0012']
# load_NACA = [True, True, True]
# num_points = np.array([151, 151, 151])
# AF_flip = np.array([[1, 1],
#                     [1, 1],
#                     [1, 1]])
# AF_scale = np.array([1, 1, 1])
# AF_angle = np.array([0, 0, 0])
# AF_offset = np.array([[0, 0],
#                       [2, 0],
#                       [4, 0]])



# airfoils = ['9410', '9509', '9412']
# load_NACA = [True, True, True]
# num_points = np.array([101, 101, 101])
# AF_flip = np.array([[1, 1],
#                     [1, 1],
#                     [1, 1]])
# AF_scale = np.array([0.167, 1, 0.33])
# AF_angle = np.array([32, 0, -7])
# AF_offset = np.array([[0.26, -0.01],
#                       [0.3, 0],
#                       [1.05, -0.015]])

# airfoils = ['9410', '9509', '9412']
# load_NACA = [True, True, True]
# num_points = np.array([101, 101, 101])
# AF_flip = np.array([[1, 1],
#                     [1, 1],
#                     [1, 1]])
# AF_scale = np.array([0.167, 1, 0.33])
# AF_angle = np.array([40, 0, -20])
# AF_offset = np.array([[0.18, -0.06],
#                       [0.3, 0],
#                       [1.21, -0.03]])

# airfoils = ['nlr7301', '2412']
# load_NACA = [False, True]
# num_points = np.array([101, 101])
# AF_flip = np.array([[1, 1],
#                     [1, 1]])
# AF_scale = np.array([1, 1])
# AF_angle = np.array([0, 0])
# AF_offset = np.array([[0, 0],
#                       [1.2, 0]])

# airfoils = ['8412', '5412']
# load_NACA = [True, True]
# num_points = np.array([101, 101])
# AF_flip = np.array([[1, -1],
#                     [1, -1]])
# AF_scale = np.array([1, 0.35])
# AF_angle = np.array([-15, -40])
# AF_offset = np.array([[0, 0],
#                       [0.9, 0.3]])
#
# airfoils = ['8412', '5412']
# load_NACA = [True, True]
# num_points = np.array([101, 101])
# AF_flip = np.array([[1, -1],
#                     [1, -1]])
# AF_scale = np.array([1, 0.35])
# AF_angle = np.array([-15, -15])
# AF_offset = np.array([[0, 0],
#                       [0.85, 0.43]])

# airfoils = ['6412', '3412', '2412']
# load_NACA = [True, True, True]
# num_points = np.array([101, 101, 101])
# AF_flip = np.array([[1, 1],
#                     [1, 1],
#                     [1, 1]])
# AF_scale = np.array([1, 0.3, 0.2])
# AF_angle = np.array([0, -25, -45])
# AF_offset = np.array([[0, 0],
#                       [1.05, -0.05],
#                       [1.35, -0.25]])
#
# airfoils = ['nlr7301', '5412']
# load_NACA = [False, True]
# num_points = np.array([101, 101])
# AF_flip = np.array([[1, 1],
#                     [1, 1]])
# AF_scale = np.array([1, 0.32])
# AF_angle = np.array([0, -20])
# AF_offset = np.array([[0, 0],
#                       [1-0.053, -0.026-0.0096]])

# airfoils = ['S1210']
# load_NACA = [False]
# num_points = np.array([171])
# AF_flip = np.array([[1, -1]])
# AF_scale = np.array([1])
# AF_angle = np.array([0])
# AF_offset = np.array([[0, 0]])

# These are definitely be the coolest ones
# The airfoil was extracted from the following paper:
"""Khormi, H., & Alfifi, S. (2022). Multi-element airfoil analysis for NACA 0012 using computational fluid dynamics. 
In AIAA SciTech Forum (pp. 1-15). American Institute of Aeronautics and Astronautics. doi:10.2514/6.2022-1528"""

# airfoils = ['NACA0012M0', 'NACA0012M1', 'NACA0012M2', 'NACA0012M3', 'NACA0012M4']
# load_NACA = [False, False, False, False, False]
# num_points = np.array([100, 100, 100, 100, 100])
# AF_flip = np.array([[1, 1],
#                     [1, 1],
#                     [1, 1],
#                     [1, 1],
#                     [1, 1]])
# AF_scale = np.array([1, 1, 1, 1, 1])
# AF_angle = np.array([0, 0, 0, 0, 0])
# AF_offset = np.loadtxt('Airfoil_DAT_Selig/offsetsNACA0012M.dat', skiprows=1) + np.array([[0, 0],
#                                                                                             [0, 0],
#                                                                                             [0, 0],
#                                                                                             [0, 0],
#                                                                                             [0, 0]])


airfoils = ['NACA0012M0', 'NACA0012M1', 'NACA0012M2', 'NACA0012M3', 'NACA0012M4']
load_NACA = [False, False, False, False, False]
num_points = np.array([100, 100, 100, 100, 100])
AF_flip = np.array([[1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1]])
AF_scale = np.array([1, 1, 1, 1, 1])
AF_angle = np.array([28, 14, 0, -14, -28])
AF_offset = np.loadtxt('../Airfoil_DAT_Selig/offsetsNACA0012MRotated.dat', skiprows=1) + np.array([[0, 0],
                                                                                                   [0, 0],
                                                                                                   [0, 0],
                                                                                                   [0, 0],
                                                                                                   [0, 0]])


V = 1
AoA = 1
num_grid = 100  # This code is really slow, so keep this number low. Max: 400
X_NEG_LIMIT = -1
X_POS_LIMIT = 3
Y_NEG_LIMIT = -0.5
Y_POS_LIMIT = 1

need_velocity = True  # Set to False if you only want to see the geometry

# %% PANEL GEOMETRY SETUP
multi_element_airfoil = MultiElementAirfoil(airfoils=airfoils, load_NACA=load_NACA, num_points=num_points,
                                            AF_flip=AF_flip, AF_scale=AF_scale, AF_angle=AF_angle,
                                            AF_offset=AF_offset)

geometries, total_geometry, total_panelized_geometry = create_clean_panelized_geometry(multi_element_airfoil,
                                                                                       airfoil_directory_path, AoA)

num_airfoils = len(airfoils)
x, y = np.linspace(X_NEG_LIMIT, X_POS_LIMIT, num_grid), np.linspace(Y_NEG_LIMIT, Y_POS_LIMIT,
                                                                    num_grid)  # Create grid
V_normal, V_tangential, lam, gamma, u, v = 0, 0, 0, 0, 0, 0
num_points = np.array([len(geometry.x) for geometry in geometries])

if need_velocity:
    V_normal, V_tangential, lam, gamma, u, v = run_source_vortex_panel_method_svpm(geometries, total_panelized_geometry,
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
    print('Chord = Total X Length: ', chordv2)
    print('Chord =  Sum of Chords ', chordV3)
    print('CL (Kutta-Joukowski) Unit Chord:  ', 2 * circulation.circulation)
    print('CL (Kutta-Joukowski) Chord = Total X Length: ', 2 * circulation.circulation / chordv2)
    print('CL (Kutta-Joukowski) Chord =  Sum of Chords ', 2 * circulation.circulation / chordV3)
    print('Circulation (Evaluated from grid velocities): ', circulation.circulation)
    print('CL (Calculated from G): ', 2 * sumGamma)

local_velocity = u ** 2 + v ** 2
cp = 1 - local_velocity / V ** 2

# %% PLOTTING
fig = plt.figure()

gs1 = fig.add_gridspec(3, 1)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
ax3 = fig.add_subplot(gs1[2])
gs1.tight_layout(fig, rect=[None, None, 0.45, None])

gs2 = fig.add_gridspec(2, 1)
ax4 = fig.add_subplot(gs2[0])
ax5 = fig.add_subplot(gs2[1])
with warnings.catch_warnings():
    # gs2.tight_layout cannot handle the subplots from the first gridspec
    # (gs1), so it will raise a warning. We are going to match the gridspecs
    # manually so we can filter the warning away.
    warnings.simplefilter("ignore", UserWarning)
    gs2.tight_layout(fig, rect=[0.45, None, None, None])

# now match the top and bottom of two gridspecs.
top = min(gs1.top, gs2.top)
bottom = max(gs1.bottom, gs2.bottom)

gs1.update(top=top, bottom=bottom)
gs2.update(top=top, bottom=bottom)

# fig.suptitle(f'Source Vortex Panel Method Results for NACA {airfoil} at {AoA} degrees AoA', fontsize=16)
# Panel Geometry
ax1.set_title('Panel Geometry')
for airfoil in geometries:
    X = airfoil.x
    Y = airfoil.y

    panelized_geometry_loc = PanelGenerator.compute_geometric_quantities(airfoil)
    panel_normal_vector_X = panelized_geometry_loc.xC + panelized_geometry_loc.S / 2 * np.cos(
        panelized_geometry_loc.delta)
    panel_normal_vector_Y = panelized_geometry_loc.yC + panelized_geometry_loc.S / 2 * np.sin(
        panelized_geometry_loc.delta)
    for i in range(len(panelized_geometry_loc.S)):
        ax1.plot([X[i], X[i + 1]], [Y[i], Y[i + 1]], 'k')
        ax1.plot([X[i]], [Y[i]], 'ro', markersize=0.5)
        ax1.plot([panelized_geometry_loc.xC[i]], [panelized_geometry_loc.yC[i]], 'bo', markersize=0.5)
        if i == 0:
            ax1.plot([panelized_geometry_loc.xC[i], panel_normal_vector_X[i]],
                     [panelized_geometry_loc.yC[i], panel_normal_vector_Y[i]],
                     'k', label='First Panel')
        if i == 1:
            ax1.plot([panelized_geometry_loc.xC[i], panel_normal_vector_X[i]],
                     [panelized_geometry_loc.yC[i], panel_normal_vector_Y[i]],
                     'k', label='Second Panel')
        else:
            ax1.plot([panelized_geometry_loc.xC[i], panel_normal_vector_X[i]],
                     [panelized_geometry_loc.yC[i], panel_normal_vector_Y[i]],
                     'k')
        ax1.plot([panel_normal_vector_X[i]], [panel_normal_vector_Y[i]], 'go', markersize=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal', adjustable='box')

if need_velocity:
    # Streamlines
    ax4.set_title('Streamlines')
    # ax4.quiver(x, y, u, v, color='b', scale=50)
    ax4.streamplot(x, y, u, v, color='b', density=10, linewidth=0.5, arrowsize=0.5)
    for geometry in geometries:
        ax4.fill(geometry.x, geometry.y, 'k')  # Plot polygon (circle or airfoil)
    plt.plot(circulation.x_cor, circulation.y_cor, 'r', markersize=2)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')

    # Pressure Contours
    ax5.set_title('Pressure Contours')
    x_m, y_m = np.meshgrid(x, y)
    cp_plot = ax5.pcolor(x_m, y_m, cp, cmap='jet', norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                                          vmin=cp.min(), vmax=cp.max()))
    ticks = -np.flip(np.geomspace(0.05, np.abs(cp.min()), 7))
    ticks = np.append(ticks, np.geomspace(0.01, cp.max(), 3))
    cbar = fig.colorbar(cp_plot, ax=ax5, label="Pressure Coefficient", ticks=ticks)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

    for geometry in geometries:
        ax5.fill(geometry.x, geometry.y, 'k')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')

    # Cp Distribution

    k_start = np.cumsum(num_points - 1) - (num_points - 1)
    k_end = np.cumsum(num_points - 1) - 1
    ax2.set_title('Cp Distribution')
    ax2.invert_yaxis()

    for i in range(num_airfoils):
        if i == 0:
            ax2.plot(total_panelized_geometry.xC[k_start[i]:k_end[i] + 1], Cp[k_start[i]:k_end[i] + 1], marker='o')
        else:
            ax2.plot(total_panelized_geometry.xC[k_start[i]:k_end[i] + 1], Cp[k_start[i]:k_end[i] + 1], marker='o')
    ax2.legend()

    ax2.set_xlabel('x/c')
    ax2.set_ylabel('Cp')
    ax2.legend()

    # Pressure Vectors
    ax3.set_title('Pressure Vectors')
    for j, airfoil in enumerate(geometries):
        X = airfoil.x
        Y = airfoil.y
        ax3.fill(X, Y, 'k')  # Plot polygon (circle or airfoil)
        panelized_geometry_loc = PanelGenerator.compute_geometric_quantities(airfoil)
        panel_normal_vector_X = panelized_geometry_loc.xC + panelized_geometry_loc.S / 2 * np.cos(
            panelized_geometry_loc.delta)
        panel_normal_vector_Y = panelized_geometry_loc.yC + panelized_geometry_loc.S / 2 * np.sin(
            panelized_geometry_loc.delta)
        Cp_loc = Cp[k_start[j]:k_end[j] + 1]
        dCpx = -Cp_loc * np.cos(panelized_geometry_loc.beta)
        dCpy = -Cp_loc * np.sin(panelized_geometry_loc.beta)
        for i in range(len(panelized_geometry_loc.S)):

            if Cp[k_start[j]:k_end[j] + 1][i] < 0:
                ax3.quiver(panelized_geometry_loc.xC[i], panelized_geometry_loc.yC[i], dCpx[i], dCpy[i], color='b',
                           scale=25, width=0.0025, headwidth=2, headlength=4)
            else:
                ax3.quiver(panelized_geometry_loc.xC[i], panelized_geometry_loc.yC[i], dCpx[i], dCpy[i], color='r',
                           scale=25, width=0.0025, headwidth=2, headlength=4, pivot='tip')

    ax3.set_xlim(X_NEG_LIMIT, X_POS_LIMIT)
    ax3.set_ylim(Y_NEG_LIMIT, Y_POS_LIMIT)
    ax3.set_xlabel('x/c')
    ax3.set_ylabel('Cp')
    ax3.set_aspect('equal', adjustable='box')
plt.show()
exit()

# Results Table
axs[1, 2].axis('off')
table_data = pd.DataFrame({
    'Sum of Source Strengths': [round(sumLambda, 6)],
    "Sum of Vortex Circulation": [round(sumGamma, 6)],
    "cl (Calculated from CP)": [round(CL, 6)],
    'Circulation (Evaluated from grid velocities)': [round(circulation.circulation, 6)],
    'cl (Kutta-Joukowski)': [round(2 * circulation.circulation, 6)],
    'cl (XFOIL)': [round(x_foil_cl, 6)],

})
table_data = table_data.round(6)
table_data = table_data.T
row_labels = table_data.index.values
# Customize the table appearance
table = axs[1, 2].table(cellText=table_data.values,
                        rowLabels=row_labels,
                        loc='center',
                        cellLoc='center',
                        bbox=[0.5, 0.6, 0.18, 0.2])

# Adjust font size and style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0])  # Adjust column width

# Add a title to the table
axs[1, 2].set_title('Results', fontsize=12, pad=20)  # Use 'pad' to adjust the distance from the top

# Optionally add grid lines
axs[1, 2].grid(False)

# Optionally add a border around the table
for key, cell in table.get_celld().items():
    cell.set_linewidth(0.5)

plt.show()
