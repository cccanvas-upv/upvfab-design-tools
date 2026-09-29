from __future__ import annotations

from typing import Literal

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

# CONFIGURACION

MODE_SOLVER_BACKEND: Literal["femwell", "tidy"] = "tidy"

WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.3

MMI_WIDTH_UM = 8

# Anchura nominal + incremento
ACCESS_WIDTH_BASE_UM = 1
ACCESS_WIDTH_INCREMENT_UM = 0.6
ACCESS_WIDTH_UM = ACCESS_WIDTH_BASE_UM + ACCESS_WIDTH_INCREMENT_UM

# Incremento respecto a 3*Lpi/8
MMI_LENGTH_INCREMENT_UM = -0.6

#ACCESS_OFFSET_UM = MMI_WIDTH_UM / 6.0

# Variación de la posición de los accesos
DY_in = 0

DY_out = 0

INPUT_X_POSITION_UM = DY_in

ACCESS_OFFSET_UM = MMI_WIDTH_UM / 3.85


OUTPUT_X_POSITIONS_UM = (
    -ACCESS_OFFSET_UM - DY_out,
    +ACCESS_OFFSET_UM + DY_out,
)

DZ_UM = 0.25

DOMAIN_X_SPAN_UM = MMI_WIDTH_UM + 4
FIELD_CUT_Z_UM = 0.5 * CORE_HEIGHT_UM

NUM_ACCESS_MODES = 8
NUM_MMI_MODES = 24

NUM_X_POINTS = 1024


# MODE SOLVER

def make_mode_solver():
    """Create the selected mode-solver backend."""

    if MODE_SOLVER_BACKEND == "femwell":

        return FemwellModeSolver(
                default_resolution = 0.3, 
                min_resolution = 0.03, 
                resolution_factor = 6,
                filter_guided = True, 
                reference_material = THERMAL_SILICON_DIOXIDE, 
                guided_tolerance = 1e-2, 
                enable_plots = False, 
        )


    if MODE_SOLVER_BACKEND == "tidy":

        from upvfab_design_tools.modesolver import Tidy3DModeSolver

        return Tidy3DModeSolver(
            min_steps_per_wvl=45,
            reference_material=THERMAL_SILICON_DIOXIDE,
            filter_guided=True,
            guided_tolerance=1e-2,
        )

    raise ValueError(
        "MODE_SOLVER_BACKEND must be 'femwell' or 'tidy'."
    )


# FUNCIONES AUXILIARES

def te_mode_result(
    result: ModeSolverResult,
) -> ModeSolverResult:
    """Return only guided TE-like modes."""

    if not result.te_modes:
        raise RuntimeError(
            "The mode solve returned no guided TE-like modes."
        )

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
    """Project a field onto one normalized modal profile."""

    mode_profile = normalize_profile(
        mode_profile,
        x_um,
    )

    return complex(
        np.trapezoid(
            np.conjugate(mode_profile) * field_profile,
            x_um,
        )
    )


def lpi_from_first_two_modes(
    result: ModeSolverResult,
) -> float:
    """Diagnostic beat length of the first two TE-like modes."""

    if len(result) < 2:
        raise RuntimeError(
            "At least two TE modes are required to estimate Lpi."
        )

    beta0 = complex(result[0].beta)
    beta1 = complex(result[1].beta)

    delta_beta = abs(
        float(np.real(beta0 - beta1))
    )

    if delta_beta <= 0:
        raise RuntimeError(
            "Could not estimate Lpi because beta0 and beta1 match."
        )

    return float(np.pi / delta_beta)


# GEOMETRIAS

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
    name="centered_access_strip",
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
    name="uniform_mmi_section",
)


# RESOLVER MODOS

solver = make_mode_solver()

print(
    f"Mode solver backend: {MODE_SOLVER_BACKEND}",
    flush=True,
)

print(
    f"Solving modes of the centered "
    f"{ACCESS_WIDTH_UM:.3f} um access strip...",
    flush=True,
)

access_result = solver.solve(
    access_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=NUM_ACCESS_MODES,
)

if not access_result.te_modes:
    raise RuntimeError(
        "The access waveguide solve returned no guided TE mode."
    )

access_mode = access_result.te_modes[0]


print(
    f"Solving modes of the "
    f"{MMI_WIDTH_UM:.3f} um multimode section...",
    flush=True,
)

mmi_result = solver.solve(
    mmi_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=NUM_MMI_MODES,
)

mmi_result = te_mode_result(mmi_result)


# DIAGNOSTICO Lpi

lpi_um = lpi_from_first_two_modes(mmi_result)

MMI_LENGTH_BASE_UM = 3*lpi_um /(4*2)
MMI_LENGTH_UM = MMI_LENGTH_BASE_UM + MMI_LENGTH_INCREMENT_UM


# PERFILES TRANSVERSALES

x_um = np.linspace(
    -0.5 * DOMAIN_X_SPAN_UM,
    +0.5 * DOMAIN_X_SPAN_UM,
    NUM_X_POINTS,
)


centered_access_profile = sample_mode_profile(
    access_mode,
    x_um=x_um,
    z_um=FIELD_CUT_Z_UM,
    field_component="Ex",
)


# Entrada
input_profile = shift_profile(
    centered_access_profile,
    x_um,
    shift_um=INPUT_X_POSITION_UM,
)


# Dos perfiles de acceso de salida
output_profiles = [
    shift_profile(
        centered_access_profile,
        x_um,
        shift_um=x_position_um,
    )
    for x_position_um in OUTPUT_X_POSITIONS_UM
]


# MODOS DEL MMI

mmi_profiles = sample_mode_profiles(
    mmi_result,
    x_um=x_um,
    z_um=FIELD_CUT_Z_UM,
    field_component="Ex",
)

mmi_profiles = normalize_profiles(
    mmi_profiles,
    x_um,
)


# EXCITACION DEL MMI

initial_amplitudes = modal_excitation_coefficients(
    input_profile,
    mmi_profiles,
    x_um,
)


represented_input_power = float(
    np.sum(
        np.abs(initial_amplitudes) ** 2
    )
)


# PROPAGACION

propagation = propagate_modes(
    mmi_result,
    initial_amplitudes=initial_amplitudes,
    length_um=MMI_LENGTH_UM,
    dz_um=DZ_UM,
)


final_field = propagation.reconstruct_field(
    mmi_profiles
)[-1]


# PROYECCION SOBRE LOS DOS PUERTOS

output_amplitudes = np.array(
    [
        scalar_projection(
            final_field,
            output_profile,
            x_um,
        )
        for output_profile in output_profiles
    ],
    dtype=np.complex128,
)


output_powers = np.abs(
    output_amplitudes
) ** 2


transmissions = (
    output_powers
    / represented_input_power
)


collected_output_transmission = float(
    np.sum(transmissions)
)


if collected_output_transmission > 0:

    split_fractions = (
        transmissions
        / collected_output_transmission
    )

else:

    split_fractions = np.zeros_like(
        transmissions
    )


# METRICAS EQUIVALENTES A LA TABLA

total_power_in_coupled = represented_input_power

power_over_outputs = output_powers.copy()

total_out_power = float(
    np.sum(power_over_outputs)
)


if total_out_power > 0:

    ratio_over_outputs = (
        power_over_outputs
        / total_out_power
    )

    excess_loss_db = (
        -10.0 * np.log10(total_out_power)
    )

else:

    ratio_over_outputs = np.zeros_like(
        power_over_outputs
    )

    excess_loss_db = np.inf


# REPRESENTACION

fig, ax = plot_propagation(
    propagation,
    x_um=x_um,
    field_profiles=mmi_profiles,
    xlim=(
        -0.5 * DOMAIN_X_SPAN_UM,
        +0.5 * DOMAIN_X_SPAN_UM,
    ),
    zlim=(
        0.0,
        MMI_LENGTH_UM,
    ),
)


# Bordes del MMI
ax.axhline(
    -0.5 * MMI_WIDTH_UM,
    color="white",
    linewidth=0.8,
    alpha=0.75,
)

ax.axhline(
    +0.5 * MMI_WIDTH_UM,
    color="white",
    linewidth=0.8,
    alpha=0.75,
)


# Posiciones de los accesos
for x_position_um in OUTPUT_X_POSITIONS_UM:

    ax.axhline(
        x_position_um,
        color="cyan",
        linestyle="--",
        linewidth=0.8,
        alpha=0.85,
    )


# RESULTADOS

print("\n" + "=" * 55)
print("1x2 MMI EME SUMMARY")
print("=" * 55)

print(
    f"Mode solver backend: "
    f"{MODE_SOLVER_BACKEND}"
)

print(
    f"Wavelength: "
    f"{WAVELENGTH_UM:.4f} um"
)

print(
    f"MMI width: "
    f"{MMI_WIDTH_UM:.4f} um"
)

print(
    f"MMI length base (3*Lpi / 4*2): "
    f"{MMI_LENGTH_BASE_UM:.4f} um"
)

print(
    f"MMI length increment: "
    f"{MMI_LENGTH_INCREMENT_UM:+.4f} um"
)

print(
    f"MMI length: "
    f"{MMI_LENGTH_UM:.4f} um"
)

print(
    f"Access width base: "
    f"{ACCESS_WIDTH_BASE_UM:.4f} um"
)

print(
    f"Access width increment: "
    f"{ACCESS_WIDTH_INCREMENT_UM:+.4f} um"
)

print(
    f"Access width: "
    f"{ACCESS_WIDTH_UM:.4f} um"
)

print(
    f"dy_in: "
    f"{DY_in:+.4f} um"
)

print(
    f"dy_out: "
    f"{DY_out:+.4f} um"
)



print(
    "Output positions: "
    f"{OUTPUT_X_POSITIONS_UM} um"
)

print(
    f"TE modes used in MMI: "
    f"{len(mmi_result)}"
)

print(
    f"Lpi from first two TE modes: "
    f"{lpi_um:.6f} um"
)

print(
    f"Lmmi / Lpi: "
    f"{MMI_LENGTH_UM / lpi_um:.6f}"
)


print("\n--- PERFORMANCE ---")

print(
    f"Total power IN coupled: "
    f"{total_power_in_coupled:.4f}"
)

print(
    f"Total OUT power: "
    f"{total_out_power:.4f}"
)

print(
    f"Excess loss [dB]: "
    f"{excess_loss_db:.4f}"
)

print(
    "Power over outputs: "
    f"{[f'{power:.4f}' for power in power_over_outputs]}"
)

print(
    "Ratio over outputs: "
    f"{[f'{ratio:.4f}' for ratio in ratio_over_outputs]}"
)


print("\n--- OUTPUT DETAILS ---")

for (
    x_position_um,
    power,
    transmission,
    split_fraction,
) in zip(
    OUTPUT_X_POSITIONS_UM,
    output_powers,
    transmissions,
    split_fractions,
    strict=True,
):

    print(
        f"x = {x_position_um:+.4f} um: "
        f"power = {power:.6f}, "
        f"T = {transmission:.6f}, "
        f"split = {100 * split_fraction:.2f} %"
    )

print(
    "\nCollected output transmission: "
    f"{collected_output_transmission:.6f}"
)

print("=" * 55)

plt.show()