"""Stepwise EME taper example using the Tidy3D mode solver backend.

This mirrors ``examples/eme/test_taper_eme.py`` as closely as possible, but
uses ``Tidy3DModeSolver`` instead of ``FemwellModeSolver``. The purpose is to
check that EME consumes the same generic ``ModeSolverResult`` and sampled mode
profiles independently of the mode-solver backend.
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
    modal_overlap_matrix,
    normalize_profiles,
    propagate_modes,
)
from upvfab_design_tools.modesolver import (
    ModeSolverResult,
    Tidy3DModeSolver,
    sample_mode_profiles,
)


WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.3
WIDTH_START_UM = 1.0
WIDTH_END_UM = 3.0
TAPER_LENGTH_UM = 50.0
NUM_SECTIONS = 20
NUM_MODES = 7
DZ_UM = 0.5
DOMAIN_X_SPAN_UM = 8.0
FIELD_CUT_Z_UM = 0.5 * CORE_HEIGHT_UM


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


widths_um = np.linspace(WIDTH_START_UM, WIDTH_END_UM, NUM_SECTIONS)
section_lengths_um = np.full(NUM_SECTIONS, TAPER_LENGTH_UM / NUM_SECTIONS)
x_um = np.linspace(-0.5 * DOMAIN_X_SPAN_UM, 0.5 * DOMAIN_X_SPAN_UM, 768)

solver = Tidy3DModeSolver(
    min_steps_per_wvl=20,
    reference_material=THERMAL_SILICON_DIOXIDE,
    filter_guided=True,
    guided_tolerance=1e-2,
)

section_results: list[ModeSolverResult] = []
section_profiles: list[np.ndarray] = []

for section_index, width_um in enumerate(widths_um):
    print(
        f"Solving Tidy3D taper section {section_index + 1}/{NUM_SECTIONS}: "
        f"width = {width_um:.3f} um",
        flush=True,
    )
    cross_section = strip_waveguide(
        width=float(width_um),
        height=CORE_HEIGHT_UM,
        core_material=SILICON_NITRIDE,
        background_material=THERMAL_SILICON_DIOXIDE,
        x_center=0.0,
        z_min=0.0,
        x_span=DOMAIN_X_SPAN_UM,
        bottom_margin=1.5,
        top_margin=1.5,
        name=f"tidy3d_taper_section_{section_index:02d}",
    )
    result = solver.solve(
        cross_section,
        wavelength_um=WAVELENGTH_UM,
        num_modes=NUM_MODES,
    )
    result = te_mode_result(result)
    profiles = sample_mode_profiles(
        result,
        x_um=x_um,
        z_um=FIELD_CUT_Z_UM,
        field_component="Ex",
    )

    section_results.append(result)
    section_profiles.append(normalize_profiles(profiles, x_um))

input_profile = section_profiles[0][0]
current_amplitudes = modal_excitation_coefficients(
    input_profile,
    section_profiles[0],
    x_um,
)
input_power = float(np.sum(np.abs(current_amplitudes) ** 2))
print(f"Power input in TE modes: {input_power:.6f}")

global_z_um: list[np.ndarray] = []
global_field: list[np.ndarray] = []
z_offset_um = 0.0

for section_index, (result, profiles, length_um) in enumerate(
    zip(section_results, section_profiles, section_lengths_um, strict=True)
):
    propagation = propagate_modes(
        result,
        initial_amplitudes=current_amplitudes,
        length_um=float(length_um),
        dz_um=DZ_UM,
    )

    section_z_um = z_offset_um + propagation.z_um
    section_field = propagation.reconstruct_field(profiles)

    if section_index > 0:
        section_z_um = section_z_um[1:]
        section_field = section_field[1:]

    global_z_um.append(section_z_um)
    global_field.append(section_field)

    current_amplitudes = propagation.final_amplitudes
    z_offset_um += float(length_um)

    if section_index < NUM_SECTIONS - 1:
        coupling = modal_overlap_matrix(
            source_profiles=profiles,
            target_profiles=section_profiles[section_index + 1],
            x_um=x_um,
        )
        current_amplitudes = coupling @ current_amplitudes

z_plot_um = np.concatenate(global_z_um)
field_plot = np.vstack(global_field)
intensity = np.abs(field_plot) ** 2

fig, ax = plt.subplots(figsize=(8, 4))
image = ax.pcolormesh(
    z_plot_um,
    x_um,
    intensity.T,
    shading="auto",
    cmap="inferno",
)

ax.plot(
    [0.0, TAPER_LENGTH_UM],
    [-0.5 * WIDTH_START_UM, -0.5 * WIDTH_END_UM],
    color="0.15",
    linewidth=0.9,
    alpha=0.8,
)
ax.plot(
    [0.0, TAPER_LENGTH_UM],
    [0.5 * WIDTH_START_UM, 0.5 * WIDTH_END_UM],
    color="0.15",
    linewidth=0.9,
    alpha=0.8,
)
ax.set_xlabel("z [µm]")
ax.set_ylabel("x [µm]")
ax.set_xlim(0.0, TAPER_LENGTH_UM)
ax.set_ylim(-0.5 * DOMAIN_X_SPAN_UM, 0.5 * DOMAIN_X_SPAN_UM)
ax.set_title("Stepwise EME taper propagation with Tidy3D |E(x, z)|²")
fig.colorbar(image, ax=ax, label="Field intensity |E|²")

try:
    plt.show()
except Exception:
    fig.savefig("eme_taper_1um_to_1p2um_tidy3d.png", dpi=300, bbox_inches="tight")


final_fundamental_power = float(np.abs(current_amplitudes[0]) ** 2)
final_represented_power = float(np.sum(np.abs(current_amplitudes) ** 2))

print(f"Taper: {WIDTH_START_UM:.3f} um -> {WIDTH_END_UM:.3f} um")
print(f"Taper length: {TAPER_LENGTH_UM:.3f} um")
print(f"Sections used: {NUM_SECTIONS}")
print(f"Backend: {section_results[-1].backend}")
print(f"Final TE modes represented: {len(section_results[-1])}")
print(f"Power in final fundamental TE mode: {final_fundamental_power:.6f}")
print(f"Total represented final scalar power: {final_represented_power:.6f}")
