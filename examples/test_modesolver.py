from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle, Trapezoid
from upvfab_design_tools.core.materials import (
    AIR,
    BCB,
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.modesolver import FemwellModeSolver
from upvfab_design_tools.modesolver import plot_effective_indices,plot_modes_grid,plot_mode_ex_ey, save_figure
import matplotlib.pyplot as plt
from upvfab_design_tools.core.visualization import (
    plot_cross_section,
    save_figure,
)
from upvfab_design_tools.core.waveguides import strip_waveguide

# Waveguide parameters 
wvgd_width = 1.0
wvgd_thickness = 0.3
bcb_thickness = 0.5

# Modesolver parameters
window_width = 10.0 
bottom_margin = 3.0
top_margin = 3.0

bcb_region = Rectangle(
    x_min=-0.5*window_width,
    x_max=0.5*window_width,
    z_min=0.0,
    z_max=bcb_thickness,
    material=BCB,
    name="BCB",
)

air_region = Rectangle(
    x_min=-0.5*window_width,
    x_max=0.5*window_width,
    z_min=bcb_thickness,
    z_max=bcb_thickness + bottom_margin,
    material=AIR,
    name="Air",
)

# sin_core = Trapezoid.from_sidewall_angle(
#     x_center=0.0,
#     z_min=0.0,
#     height=0.3,
#     width=1.0,
#     width_reference="top",
#     sidewall_angle_deg=6.0,
#     material=SILICON_NITRIDE,
#     name="rib ridge",
# )

# xs = CrossSection(
#     name="complex_sin_cross_section",
#     background_material=THERMAL_SILICON_DIOXIDE,
#     x_min=-4.0,
#     x_max=4.0,
#     z_min=-2.0,
#     z_max=2.0,
#     structures=(
#         bcb_region,
#         air_region,
#         sin_core,
#     ),
# )

xs = strip_waveguide(
    width = wvgd_width,
    height=  wvgd_thickness,
    core_material = SILICON_NITRIDE,
    background_material = THERMAL_SILICON_DIOXIDE,
    x_center = 0.0,
    z_min = 0.0,
    sidewall_angle_deg = 0.0,
    width_reference = "bottom",
    x_span = window_width,
    bottom_margin = bottom_margin,
    top_margin = top_margin,
    surrounding_regions = (
        bcb_region,
        air_region,
    ),
)

fig, ax = plot_cross_section(
    xs,
    wavelength_um=1.55,
    show_names=False,
    show_indices=True,
)
plt.show()

solver = FemwellModeSolver(
    default_resolution=0.3,
    min_resolution=0.01,
    resolution_factor=10.0,
    filter_guided=True,
    reference_material=THERMAL_SILICON_DIOXIDE,
    guided_tolerance=1e-2,
    enable_plots=False,
)

result = solver.solve(
    cross_section=xs,
    wavelength_um=1.55,
    num_modes=8,
)

# Need to adjust because in a grid I might need to switch from one field component to another. TE/TM
# Also need to add the option to plot "abs" or "real"
fig, axs = plot_modes_grid(
    result,
    field_component="auto",
    field_part ="mag",
    max_modes=4,
    xlim=(-2, 2),
    zlim=(-1, 1),
)
# fig_neff, ax_neff = plot_effective_indices(result)

try:
    plt.show()
except:
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
