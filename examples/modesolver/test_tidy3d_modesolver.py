"""Minimal Tidy3D mode-solver example using UPVfab core geometry."""

import numpy as np
from matplotlib import pyplot as plt
from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle, Trapezoid
from upvfab_design_tools.core.materials import (
    AIR,
    BCB,
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.modesolver import (
    Tidy3DModeSolver,
    plot_modes_grid,
    sample_mode_profiles,
)


WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.3
CORE_WIDTH_UM = 1.0
FIELD_CUT_Z_UM = 0.5 * CORE_HEIGHT_UM


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

sin_core = Trapezoid.from_sidewall_angle(
    x_center=0.0,
    z_min=0.0,
    height=wvgd_thickness,
    width=wvgd_width,
    width_reference="bottom",
    sidewall_angle_deg=6.0,
    material=SILICON_NITRIDE,
    name="Core",
)

# xs = CrossSection(
#     name="sin_cross_section",
#     background_material=THERMAL_SILICON_DIOXIDE,
#     x_min=-0.5*window_width,
#     x_max=0.5*window_width,
#     z_min=-bottom_margin,
#     z_max=wvgd_thickness + top_margin,
#     structures=(
#         # bcb_region,
#         # air_region,
#         sin_core,
#     ),
# )

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


# cross_section = strip_waveguide(
#     width=CORE_WIDTH_UM,
#     height=CORE_HEIGHT_UM,
#     core_material=SILICON_NITRIDE,
#     background_material=THERMAL_SILICON_DIOXIDE,
#     x_center=0.0,
#     z_min=0.0,
#     x_span=6.0,
#     bottom_margin=2.0,
#     top_margin=2.0,
#     name="tidy3d_1um_strip",
# )

solver = Tidy3DModeSolver(
    min_steps_per_wvl=30,
    reference_material=THERMAL_SILICON_DIOXIDE,
    filter_guided=True,
    guided_tolerance=1e-2,
)
result = solver.solve(
    xs,
    wavelength_um=WAVELENGTH_UM,
    num_modes=4,
)

x_um = np.linspace(-2.5, 2.5, 301)
profiles = sample_mode_profiles(
    result,
    x_um=x_um,
    z_um=FIELD_CUT_Z_UM,
    field_component="Ex",
)

# fig_neff, ax_neff = plot_effective_indices(result)
fig_grid, axs_grid = plot_modes_grid(
    result,
    field_component="auto",
    field_part="mag",
    max_modes=None,
    xlim=(-2.5, 2.5),
    zlim=(-1.0, 1.0),
)
fig_profiles, ax_profiles = plt.subplots(figsize=(6, 3.5))

for mode, profile in zip(result.modes, profiles, strict=True):
    ax_profiles.plot(
        x_um,
        np.abs(profile),
        label=f"mode {mode.index} {mode.polarization}",
    )

ax_profiles.set_xlabel("x [µm]")
ax_profiles.set_ylabel("|Ex|")
ax_profiles.set_title("Tidy3D sampled Ex modal profiles")
ax_profiles.grid(True)
ax_profiles.legend()

try:
    plt.show()
except Exception:
    # fig_neff.savefig("tidy3d_effective_indices.png", dpi=300, bbox_inches="tight")
    fig_grid.savefig("tidy3d_modes_grid.png", dpi=300, bbox_inches="tight")
    fig_profiles.savefig("tidy3d_mode_profiles.png", dpi=300, bbox_inches="tight")

for mode in result.modes:
    print(
        mode.index,
        mode.neff,
        mode.polarization,
        mode.te_fraction,
        mode.tm_fraction,
    )

print(f"Sampled profiles shape: {profiles.shape}")
