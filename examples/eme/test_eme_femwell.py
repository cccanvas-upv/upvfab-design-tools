"""FEMWELL + EME example for a centered 1 um to 10 um strip transition."""

import numpy as np
from matplotlib import pyplot as plt
from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.waveguides import strip_waveguide
from upvfab_design_tools.eme import (
    modal_excitation_coefficients,
    normalize_profiles,
    plot_propagation,
    propagate_modes,
    shift_profile,
)
from upvfab_design_tools.modesolver import (
    FemwellModeSolver,
    ModeSolverResult,
    sample_mode_profile,
    sample_mode_profiles,
)


WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.3
INPUT_WIDTH_UM = 1.0
PROPAGATION_WIDTH_UM = 10.0
PROPAGATION_LENGTH_UM = 200.0
DZ_UM = 0.2
DOMAIN_X_SPAN_UM = 14.0
FIELD_CUT_Z_UM = 0.5 * CORE_HEIGHT_UM


input_cross_section = strip_waveguide(
    width=INPUT_WIDTH_UM,
    height=CORE_HEIGHT_UM,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    x_center=0.0,
    z_min=0.0,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=1.5,
    top_margin=1.5,
    name="centered_1um_input_strip",
)

propagation_cross_section = strip_waveguide(
    width=PROPAGATION_WIDTH_UM,
    height=CORE_HEIGHT_UM,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    x_center=0.0,
    z_min=0.0,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=1.5,
    top_margin=1.5,
    name="centered_10um_propagation_strip",
)

solver = FemwellModeSolver(
    resolution_overrides={
        "core": {"resolution": 0.06, "distance": 0.3},
        "background": {"resolution": 0.4, "distance": 1.0},
    },
    reference_material=THERMAL_SILICON_DIOXIDE,
    filter_guided=True,
    guided_tolerance=1e-2,
)

print("Solving modes of the centered 1 um input strip...")
input_result = solver.solve(
    input_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=4,
)

if not input_result.te_modes:
    raise RuntimeError("The input waveguide solve returned no guided TE mode.")

input_mode = input_result.te_modes[0]

print("Solving modes of the 10 um propagation strip...")
wide_result = solver.solve(
    propagation_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=30,
)

if not wide_result.te_modes:
    raise RuntimeError("The propagation-section solve returned no guided TE modes.")

propagation_mode_result = ModeSolverResult(
    modes=wide_result.te_modes,
    wavelength_um=wide_result.wavelength_um,
    backend=wide_result.backend,
    cross_section=wide_result.cross_section,
    metadata=wide_result.metadata,
)

x_um = np.linspace(-6.9, 6.9, 1024)

INPUT_X_POSITION_UM = 0.0

centered_input_profile = sample_mode_profile(
    input_mode,
    x_um=x_um,
    z_um=FIELD_CUT_Z_UM,
    field_component="Ex",
)
input_profile = shift_profile(
    centered_input_profile,
    x_um,
    shift_um=INPUT_X_POSITION_UM,
)
propagation_profiles = sample_mode_profiles(
    propagation_mode_result,
    x_um=x_um,
    z_um=FIELD_CUT_Z_UM,
    field_component="Ex",
)
propagation_profiles = normalize_profiles(propagation_profiles, x_um)

initial_amplitudes = modal_excitation_coefficients(
    input_profile,
    propagation_profiles,
    x_um,
)
propagation = propagate_modes(
    propagation_mode_result,
    initial_amplitudes=initial_amplitudes,
    length_um=PROPAGATION_LENGTH_UM,
    dz_um=DZ_UM,
)

fig, ax = plot_propagation(
    propagation,
    x_um=x_um,
    field_profiles=propagation_profiles,
    xlim=(-6.0, 6.0),
    zlim=(0.0, PROPAGATION_LENGTH_UM),
)
ax.axhline(
    -0.5 * PROPAGATION_WIDTH_UM,
    color="white",
    linewidth=0.8,
    alpha=0.7,
)
ax.axhline(
    0.5 * PROPAGATION_WIDTH_UM,
    color="white",
    linewidth=0.8,
    alpha=0.7,
)
ax.axhline(
    INPUT_X_POSITION_UM,
    color="cyan",
    linestyle="--",
    linewidth=0.8,
    alpha=0.8,
)

# output_filename = "eme_strip_1um_to_10um.png"
# fig.savefig(output_filename, dpi=300, bbox_inches="tight")
plt.show()

coupled_power = float(np.sum(np.abs(initial_amplitudes) ** 2))
print(f"Input TE mode: n_eff = {input_mode.neff.real:.6f}")
print(f"Input x position: {INPUT_X_POSITION_UM:.3f} um")
print(f"Propagation TE modes used: {len(propagation_mode_result)}")
print(f"Scalar power represented by these modes: {coupled_power:.6f}")
# print(f"Saved propagation plot to {output_filename}")
