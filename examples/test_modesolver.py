from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle
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

bcb_region = Rectangle(
    x_min=-4.0,
    x_max=4.0,
    z_min=0.3,
    z_max=0.35,
    material=BCB,
    name="BCB",
)

air_region = Rectangle(
    x_min=-4.0,
    x_max=4.0,
    z_min=0.35,
    z_max=2.0,
    material=AIR,
    name="Air",
)

sin_core = Rectangle(
    x_min=-0.5,
    x_max=0.5,
    z_min=0.0,
    z_max=0.3,
    material=SILICON_NITRIDE,
    name="core",
)

xs = CrossSection(
    name="complex_sin_cross_section",
    background_material=THERMAL_SILICON_DIOXIDE,
    x_min=-4.0,
    x_max=4.0,
    z_min=-2.0,
    z_max=2.0,
    structures=(
        bcb_region,
        air_region,
        sin_core,
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
    default_resolution=0.4,
    min_resolution=0.02,
    resolution_factor=5.0,
    filter_guided=True,
    reference_material=THERMAL_SILICON_DIOXIDE,
    guided_tolerance=1e-2,
    enable_plots=False,
)

result = solver.solve(
    cross_section=xs,
    wavelength_um=1.55,
    num_modes=4,
)

# Need to adjust because in a grid I might need to switch from one field component to another. TE/TM
fig, axs = plot_modes_grid(
    result,
    field_component="Ex",
    max_modes=4,
    xlim=(-2, 2),
    zlim=(-1, 1),
)
plt.show()
save_figure(fig, "modes_ex.png")

for mode in result.modes:
    print(
        mode.index,
        mode.neff,
        mode.polarization,
        mode.te_fraction,
        mode.tm_fraction,
    )