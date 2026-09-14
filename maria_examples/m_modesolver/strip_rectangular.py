
# 4 MODES TARDA PROU
#importem tot lo necessari: cross_sections, materials, solvers, ferramentes de plot
import matplotlib.pyplot as plt
from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle, Trapezoid
from upvfab_design_tools.core.materials import (
    AIR,
    BCB,
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.modesolver import FemwellModeSolver
from upvfab_design_tools.modesolver import plot_modes_grid
from upvfab_design_tools.core.visualization import (
    plot_cross_section,
    save_figure,
)
from upvfab_design_tools.core.waveguides import strip_waveguide

#waveguide parameters
wvg_width = 1.0
wvg_thickness = 0.3
bcb_thickness = 0.5

#modesolver parameters
window_width = 10.0
bottom_margin = 3.0
top_margin = 3.0

bcb_region = Rectangle(
    x_min = -0.5*window_width, #per a tindre el centre en 0
    x_max = 0.5*window_width, 
    z_min = 0.0, 
    z_max = bcb_thickness, 
    material = BCB, 
    name = "BCB",
)

air_region = Rectangle(
    x_min = -0.5*window_width, 
    x_max = 0.5*window_width, 
    z_min = bcb_thickness, 
    z_max = bcb_thickness + bottom_margin, 
    material = AIR, 
    name = "Air",
)

sin_core = Rectangle(
    x_min = -0.5*wvg_width,
    x_max = 0.5*wvg_width,
    z_min = 0, 
    z_max = wvg_thickness, #aixina tens bcb de cladding
    material = SILICON_NITRIDE, 
    name = "Core",
)

xs = CrossSection(
    name = "sin_cross_section", 
    background_material = THERMAL_SILICON_DIOXIDE, 
    x_min = -0.5*window_width, 
    x_max = 0.5*window_width, 
    z_min = -bottom_margin, 
    z_max = wvg_thickness + top_margin,
    structures = (
        bcb_region, 
        air_region,
        sin_core, 
    ),
)

fig, ax = plot_cross_section(
    xs,
    wavelength_um = 1.55,
    show_names = True,
    show_indices = True,
)
plt.show() #aixina revises que la xs está bé

# solver configuration

solver = FemwellModeSolver(
    default_resolution=0.3, #default mesh resolution in um
    min_resolution=0.01, #minimum automatically generated resolution in um
    resolution_factor=10.0, #automatic resolution is approximately min_feature / resolution_factor
    filter_guided=True,#if True, only keeps modes with n_eff above the reference material
    reference_material=THERMAL_SILICON_DIOXIDE, #if None, xs background material is used
    guided_tolerance=1e-2, #minimum index difference for guided mode filtering
    enable_plots=False, #if True, plot mesh, domains and epsilon
)

result = solver.solve(
    cross_section = xs,
    wavelength_um = 1.55,
    num_modes = 4, 
)

# comentaris camilo:
# Need to adjust because in a grid I might need to switch from one field component to another. TE/TM
# Also need to add the option to plot "abs" or "real"

fig, axs = plot_modes_grid(
    result,
    field_component = "auto",
    field_part = "mag", 
    max_modes = 4,
    xlim = (-2,2),
    zlim = (-1,1),
)

# fig_neff, ax_neff = plot_effective_indices(result)

try:
    plt.show()
except Exception:
    save_figure(fig, "modes_ex.png")
    # save_figure(fig_neff, "effective_indices.png")
    pass

for mode in result.modes:
    print(
        mode.index,
        mode.neff, 
        mode.polarization, 
        mode.te_fraction, 
        mode.tm_fraction,
    )
    