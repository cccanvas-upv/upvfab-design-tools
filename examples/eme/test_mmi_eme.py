"""Uniform-section EME example for a 2x2 paired-interference MMI.

The multimode section is modeled as one uniform 12 um-wide waveguide. A 2 um
input access mode is launched at x = -2 um, the field is propagated over a
length estimated from the multimode-section beat length Lpi, and the final
field is projected onto two 2 um output access modes centered at x = +/-2 um.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.waveguides import strip_waveguide
from upvfab_design_tools.eme import (
    modal_excitation_coefficients,
    normalize_profile,
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
ACCESS_WIDTH_UM = 2.0
MMI_WIDTH_UM = 12.0
ACCESS_X_POSITIONS_UM = (-2.0, 2.0)
INPUT_X_POSITION_UM = ACCESS_X_POSITIONS_UM[0]
PAIRED_INTERFERENCE_LENGTH_FACTOR = 0.5
DZ_UM = 0.25
DOMAIN_X_SPAN_UM = 18.0
FIELD_CUT_Z_UM = 0.5 * CORE_HEIGHT_UM
NUM_ACCESS_MODES = 8
NUM_MMI_MODES = 24


def te_mode_result(result: ModeSolverResult) -> ModeSolverResult:
    """Return a result containing only TE-like modes."""

    if not result.te_modes:
        raise RuntimeError("The mode solve returned no guided TE-like modes.")

    return ModeSolverResult(
        modes=result.te_modes,
        wavelength_um=result.wavelength_um,
        backend=result.backend,
        cross_section=result.cross_section,
        metadata=result.metadata,
    )


def scalar_projection(
    field_profile: np.ndarray,
    mode_profile: np.ndarray,
    x_um: np.ndarray,
) -> complex:
    """Project an arbitrary field profile onto one normalized mode profile."""

    mode_profile = normalize_profile(mode_profile, x_um)
    return complex(np.trapezoid(np.conjugate(mode_profile) * field_profile, x_um))


def lpi_from_first_two_modes(result: ModeSolverResult) -> float:
    """Estimate MMI beat length from the first two modes."""

    if len(result) < 2:
        raise RuntimeError("At least two TE modes are required to estimate Lpi.")

    beta0 = complex(result[0].beta)
    beta1 = complex(result[1].beta)
    delta_beta = abs(float(np.real(beta0 - beta1)))

    if delta_beta <= 0:
        raise RuntimeError("Could not estimate Lpi because beta0 and beta1 match.")

    return float(np.pi / delta_beta)


access_cross_section = strip_waveguide(
    width=ACCESS_WIDTH_UM,
    height=CORE_HEIGHT_UM,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    x_center=0.0,
    z_min=0.0,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=1.5,
    top_margin=1.5,
    name="centered_2um_access_strip",
)

mmi_cross_section = strip_waveguide(
    width=MMI_WIDTH_UM,
    height=CORE_HEIGHT_UM,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    x_center=0.0,
    z_min=0.0,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=1.5,
    top_margin=1.5,
    name="uniform_12um_mmi_section",
)

solver = FemwellModeSolver(
    resolution_overrides={
        "core": {"resolution": 0.08, "distance": 0.3},
        "background": {"resolution": 0.6, "distance": 1.0},
    },
    reference_material=THERMAL_SILICON_DIOXIDE,
    filter_guided=True,
    guided_tolerance=1e-2,
)

print("Solving modes of the centered 2 um access strip...", flush=True)
access_result = solver.solve(
    access_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=NUM_ACCESS_MODES,
)

if not access_result.te_modes:
    raise RuntimeError("The access waveguide solve returned no guided TE mode.")

access_mode = access_result.te_modes[0]

print("Solving modes of the 12 um multimode section...", flush=True)
mmi_result = solver.solve(
    mmi_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=NUM_MMI_MODES,
)
mmi_result = te_mode_result(mmi_result)

lpi_um = lpi_from_first_two_modes(mmi_result)
mmi_length_um = PAIRED_INTERFERENCE_LENGTH_FACTOR * lpi_um

x_um = np.linspace(-0.5 * DOMAIN_X_SPAN_UM, 0.5 * DOMAIN_X_SPAN_UM, 1024)

centered_access_profile = sample_mode_profile(
    access_mode,
    x_um=x_um,
    z_um=FIELD_CUT_Z_UM,
    field_component="Ex",
)
input_profile = shift_profile(
    centered_access_profile,
    x_um,
    shift_um=INPUT_X_POSITION_UM,
)
output_profiles = [
    shift_profile(centered_access_profile, x_um, shift_um=x_position_um)
    for x_position_um in ACCESS_X_POSITIONS_UM
]

mmi_profiles = sample_mode_profiles(
    mmi_result,
    x_um=x_um,
    z_um=FIELD_CUT_Z_UM,
    field_component="Ex",
)
mmi_profiles = normalize_profiles(mmi_profiles, x_um)

initial_amplitudes = modal_excitation_coefficients(
    input_profile,
    mmi_profiles,
    x_um,
)
propagation = propagate_modes(
    mmi_result,
    initial_amplitudes=initial_amplitudes,
    length_um=mmi_length_um,
    dz_um=DZ_UM,
)
final_field = propagation.reconstruct_field(mmi_profiles)[-1]

output_amplitudes = np.array(
    [
        scalar_projection(final_field, output_profile, x_um)
        for output_profile in output_profiles
    ],
    dtype=np.complex128,
)
output_powers = np.abs(output_amplitudes) ** 2

represented_input_power = float(np.sum(np.abs(initial_amplitudes) ** 2))
transmissions = output_powers / represented_input_power

fig, ax = plot_propagation(
    propagation,
    x_um=x_um,
    field_profiles=mmi_profiles,
    xlim=(-0.5 * DOMAIN_X_SPAN_UM, 0.5 * DOMAIN_X_SPAN_UM),
    zlim=(0.0, mmi_length_um),
)

ax.axhline(-0.5 * MMI_WIDTH_UM, color="white", linewidth=0.8, alpha=0.75)
ax.axhline(0.5 * MMI_WIDTH_UM, color="white", linewidth=0.8, alpha=0.75)

for x_position_um in ACCESS_X_POSITIONS_UM:
    ax.axhline(
        x_position_um,
        color="cyan",
        linestyle="--",
        linewidth=0.8,
        alpha=0.85,
    )

try:
    plt.show()
except Exception:
    fig.savefig("eme_mmi_2x2_paired_interference.png", dpi=300, bbox_inches="tight")

print(f"MMI width: {MMI_WIDTH_UM:.3f} um")
print(f"Access width: {ACCESS_WIDTH_UM:.3f} um")
print(f"Access positions: {ACCESS_X_POSITIONS_UM} um")
print(f"Input position: {INPUT_X_POSITION_UM:.3f} um")
print(f"TE modes used in MMI: {len(mmi_result)}")
print(f"Lpi from first two TE modes: {lpi_um:.6f} um")
print(
    "MMI length "
    f"({PAIRED_INTERFERENCE_LENGTH_FACTOR:g} * Lpi): {mmi_length_um:.6f} um"
)
print(f"Represented input scalar power: {represented_input_power:.6f}")

for x_position_um, power, transmission in zip(
    ACCESS_X_POSITIONS_UM,
    output_powers,
    transmissions,
    strict=True,
):
    print(
        f"Output at x = {x_position_um:+.3f} um: "
        f"power = {power:.6f}, transmission = {transmission:.6f}"
    )

print(f"Collected output transmission: {float(np.sum(transmissions)):.6f}")
