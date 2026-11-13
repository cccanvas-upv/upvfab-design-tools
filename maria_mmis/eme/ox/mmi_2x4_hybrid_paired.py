from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

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


MODE_SOLVER_BACKEND: Literal["femwell","tidy",] = "tidy"


INPUT_CONFIGURATION: Literal["input_0", "input_1", "hybrid"] = "input_1"

# Fase del input 1 respecto al input 0
DELTA_PHI_IN_DEG = 0.0

WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.3
MMI_WIDTH_UM = 16.0

ACCESS_WIDTH_BASE_UM = 1.0
ACCESS_WIDTH_INCREMENT_UM = 2.0

ACCESS_WIDTH_UM = ACCESS_WIDTH_BASE_UM + ACCESS_WIDTH_INCREMENT_UM

# Incremento respecto a: Lmmi = 3*Lpi/4
MMI_LENGTH_INCREMENT_UM = -0.5

N_INPUTS = 2
N_OUTPUTS = 4

DY = 0.0

INPUT_X_POSITIONS_UM = np.array([
    +MMI_WIDTH_UM / 6.0,
    -MMI_WIDTH_UM / 6.0,
])

OUTPUT_X_POSITIONS_UM = np.array([
    +MMI_WIDTH_UM / 3.0,
    +MMI_WIDTH_UM / 6.0,
    -MMI_WIDTH_UM / 6.0,
    -MMI_WIDTH_UM / 3.0,
])

DZ_UM = 0.25

DOMAIN_X_SPAN_UM = MMI_WIDTH_UM + 4.0

FIELD_CUT_Z_UM = 0.5* CORE_HEIGHT_UM

NUM_ACCESS_MODES = 8
NUM_MMI_MODES = 24

NUM_X_POINTS = 1024

def make_mode_solver():
    """Create the selected mode-solver backend."""

    if MODE_SOLVER_BACKEND == "femwell":

        return FemwellModeSolver(
            default_resolution=0.3,
            min_resolution=0.03,
            resolution_factor=6,
            filter_guided=True,
            reference_material=THERMAL_SILICON_DIOXIDE,
            guided_tolerance=1e-2,
            enable_plots=False,
        )

    if MODE_SOLVER_BACKEND == "tidy":

        from upvfab_design_tools.modesolver import Tidy3DModeSolver
        

        return Tidy3DModeSolver(
            min_steps_per_wvl=45,
            reference_material=THERMAL_SILICON_DIOXIDE,
            filter_guided=True,
            guided_tolerance=1e-2,
        )

    raise ValueError("MODE_SOLVER_BACKEND must be 'femwell' or 'tidy'.")

# FUNCIONES AUXILIARES

def te_mode_result(result: ModeSolverResult,) -> ModeSolverResult:
    """Return only guided TE-like modes."""

    if not result.te_modes:

        raise RuntimeError(
            "The mode solve returned no guided TE-like modes.")

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

    mode_profile = normalize_profile(mode_profile, x_um)

    return complex(np.trapezoid(np.conjugate(mode_profile) * field_profile, x_um))


def lpi_from_first_two_modes(result: ModeSolverResult) -> float:
    """Diagnostic beat length of the first two TE-like modes."""

    if len(result) < 2:
        raise RuntimeError("At least two TE modes are required to estimate Lpi.")

    beta0 = complex(result[0].beta)
    beta1 = complex(result[1].beta)

    delta_beta = abs(float(np.real(beta0 - beta1)))

    if delta_beta <= 0:
        raise RuntimeError("Could not estimate Lpi because beta0 and beta1 match.")
    return float(np.pi / delta_beta)


def wrapped_phase_deg(value: np.ndarray | complex) -> np.ndarray:
    """Return phase in the interval [-180, 180] degrees."""

    return np.degrees(np.angle(value))

print("\n--- 2x4 PORT GEOMETRY ---")

print("Input positions [um]: "
    f"{INPUT_X_POSITIONS_UM}")

print("Output positions [um]: "
    f"{OUTPUT_X_POSITIONS_UM}")

#GEOMETRIAS

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
    name="access_waveguide",
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
    name="mmi_2x4_section",
)

# RESOLVER MODOS

solver = make_mode_solver()

print("\nSolving access waveguide modes...",
    flush=True)

access_result = solver.solve(access_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=NUM_ACCESS_MODES,
)


if not access_result.te_modes:

    raise RuntimeError(
        "The access waveguide solve returned no guided TE mode.")


access_mode = (access_result.te_modes[0])


print(f"Solving modes of the "
    f"{MMI_WIDTH_UM:.3f} um "
    f"multimode section...",
    flush=True)

mmi_result = solver.solve(mmi_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=NUM_MMI_MODES)

mmi_result = te_mode_result(mmi_result)


######################################################################################### Lpi Y LONGITUD DEL MMI ################################################################################

lpi_um = lpi_from_first_two_modes(mmi_result)

MMI_LENGTH_BASE_UM = lpi_um/ 4.0
MMI_LENGTH_UM = (MMI_LENGTH_BASE_UM+ MMI_LENGTH_INCREMENT_UM)

################################################################################################################################################################################################

x_um = np.linspace(-0.5 * DOMAIN_X_SPAN_UM, +0.5 * DOMAIN_X_SPAN_UM, NUM_X_POINTS)


centered_access_profile = (
    sample_mode_profile(access_mode, x_um=x_um, z_um=FIELD_CUT_Z_UM, field_component="Ex"))

centered_access_profile = (
    normalize_profile(centered_access_profile, x_um))


input_profiles = [shift_profile(centered_access_profile, x_um, shift_um=x_position_um)
    for x_position_um in INPUT_X_POSITIONS_UM]


output_profiles = [shift_profile(centered_access_profile,x_um, shift_um=x_position_um)
    for x_position_um in OUTPUT_X_POSITIONS_UM]



mmi_profiles = sample_mode_profiles(
    mmi_result,
    x_um=x_um,
    z_um=FIELD_CUT_Z_UM,
    field_component="Ex",
)

mmi_profiles = normalize_profiles(mmi_profiles, x_um)


input_modal_amplitudes = np.array(
    [
        modal_excitation_coefficients(
            input_profile,
            mmi_profiles,
            x_um,
        )

        for input_profile
        in input_profiles
    ],
    dtype=np.complex128,
)


represented_input_powers = np.sum(
    np.abs(input_modal_amplitudes) ** 2,
    axis=1,
)


normalized_input_modal_amplitudes = ( input_modal_amplitudes/ np.sqrt( represented_input_powers)[:, None])

transfer_matrix = np.zeros(
    (
        N_OUTPUTS,
        N_INPUTS,
    ),
    dtype=np.complex128,
)


single_input_propagations = []


for input_index in range(N_INPUTS):

    propagation_single = propagate_modes(
        mmi_result,
        initial_amplitudes=(normalized_input_modal_amplitudes[input_index]),
        length_um=MMI_LENGTH_UM,
        dz_um=DZ_UM,
    )

    single_input_propagations.append(propagation_single)
    final_field_single = (propagation_single.reconstruct_field(mmi_profiles)[-1])
    for output_index, output_profile in enumerate(output_profiles):

        transfer_matrix[output_index, input_index] = scalar_projection(final_field_single,output_profile, x_um)



single_input_powers = (np.abs(transfer_matrix) ** 2)


single_input_total_power = (np.sum(single_input_powers, axis=0))


single_input_splitting = np.divide(
    single_input_powers,
    single_input_total_power[None,:],
    out=np.zeros_like(single_input_powers),
    where=(single_input_total_power[None,:]> 0.0),
)


single_input_excess_loss_db = (
    -10.0* np.log10(np.maximum(single_input_total_power,1e-20)))

# FASES DE LA MATRIZ DE TRANSFERENCIA

transfer_phase_deg = np.degrees(np.angle(transfer_matrix))


# Fase del camino procedente del input 1
# respecto al camino procedente del input 0
# para cada salida.

relative_transfer_phase_deg = np.degrees(
    np.angle(
        transfer_matrix[:, 0]
        * np.conj(
            transfer_matrix[:, 1]
        )
    )
)


if INPUT_CONFIGURATION == "input_0":
    initial_amplitudes = (normalized_input_modal_amplitudes[0])


elif INPUT_CONFIGURATION == "input_1":
    initial_amplitudes = (normalized_input_modal_amplitudes[1])


elif INPUT_CONFIGURATION == "hybrid":

    delta_phi_rad = np.deg2rad( DELTA_PHI_IN_DEG)

    initial_amplitudes = (
        normalized_input_modal_amplitudes[0] + np.exp(1j* delta_phi_rad) * normalized_input_modal_amplitudes[1]) / np.sqrt(2.0)


else:

    raise ValueError(
        "INPUT_CONFIGURATION must be 'input_0', 'input_1' or 'hybrid'.")

represented_input_power = float(np.sum(np.abs(initial_amplitudes) ** 2))



propagation = propagate_modes(
    mmi_result,
    initial_amplitudes=initial_amplitudes,
    length_um=MMI_LENGTH_UM,
    dz_um=DZ_UM,
)


field = propagation.reconstruct_field(mmi_profiles)

final_field = field[-1]

output_amplitudes = np.array(
    [
        scalar_projection(
            final_field,
            output_profile,
            x_um,
        )

        for output_profile
        in output_profiles
    ],
    dtype=np.complex128,
)


output_powers = (np.abs(output_amplitudes) ** 2)

transmissions = (output_powers/ represented_input_power)

collected_output_transmission = float(np.sum( transmissions))

if collected_output_transmission > 0.0:
    split_fractions = (transmissions/ collected_output_transmission)

else:
    split_fractions = np.zeros_like(transmissions)


# EXCESS LOSS

if collected_output_transmission > 0.0:
    excess_loss_db = (-10.0* np.log10(collected_output_transmission))

else:
    excess_loss_db = np.inf


def wrap_phase_deg(
    phase_deg: float | np.ndarray,
):
    """Wrap phase to [-180, 180) degrees."""

    return (
        phase_deg + 180.0
    ) % 360.0 - 180.0


# ============================================================
# RELACION DE FASE ENTRE LAS DOS ENTRADAS
# ============================================================

# Nuestro orden de inputs:
#
# input 0 = -3W/8 -> I2 del paper
# input 1 = +W/8  -> I1 del paper
#
# Por tanto, para cada output calculamos:
#
# phi_k = arg(H_k,I1) - arg(H_k,I2)
#
#       = arg(H_k,I1 * conj(H_k,I2))
#
# Esta es la fase que determina el desplazamiento de las
# franjas de interferencia cuando se excitan ambos inputs.


relative_transfer_phase_deg = np.degrees(
    np.angle(
        transfer_matrix[:, 0]
        * np.conj(
            transfer_matrix[:, 1]
        )
    )
)


# ============================================================
# MAPPING DE NUESTROS OUTPUTS A LOS DEL PAPER
# ============================================================

# Nuestro orden:
#
# output 0 = -3W/8 -> P4
# output 1 = -W/8  -> P3
# output 2 = +W/8  -> P2
# output 3 = +3W/8 -> P1
#
# Por tanto:

phi_p4_deg = relative_transfer_phase_deg[0]
phi_p3_deg = relative_transfer_phase_deg[1]
phi_p2_deg = relative_transfer_phase_deg[2]
phi_p1_deg = relative_transfer_phase_deg[3]


# ============================================================
# RELACIONES DE FASE ENTRE OUTPUTS
# ============================================================

# Yu et al. utiliza P3 como referencia.
#
# Idealmente:
#
# phi1 - phi3 = 90 deg
# phi2 - phi3 = 180 deg
# phi3 - phi4 = 90 deg


phase_13_deg = wrap_phase_deg(
    phi_p1_deg
    - phi_p3_deg
)

phase_23_deg = wrap_phase_deg(
    phi_p2_deg
    - phi_p3_deg
)

phase_34_deg = wrap_phase_deg(
    phi_p3_deg
    - phi_p4_deg
)


# ============================================================
# PHASE DEVIATIONS - YU ET AL.
# ============================================================

# Definiciones del paper:
#
# Delta_phi_13 = phi1 - phi3 - 90 deg
# Delta_phi_23 = phi2 - phi3 - 180 deg
# Delta_phi_34 = phi3 - phi4 - 90 deg
#
# Para un híbrido ideal:
#
# Delta_phi_13 = 0 deg
# Delta_phi_23 = 0 deg
# Delta_phi_34 = 0 deg


delta_phi_13_deg = wrap_phase_deg(
    phase_13_deg
    - 90.0
)

delta_phi_23_deg = wrap_phase_deg(
    phase_23_deg
    - 180.0
)

delta_phi_34_deg = wrap_phase_deg(
    phase_34_deg
    - 90.0
)


# ============================================================
# FASES EQUIVALENTES RESPECTO A P3
# ============================================================

# Esto sirve para visualizar el conocido:
#
# P4, P3, P2, P1
#
# 270, 0, 180, 90 deg
#
# Se representa en [0, 360) para que sea más intuitivo.


phase_relative_p3_deg = np.array(
    [
        phi_p4_deg - phi_p3_deg,
        0.0,
        phi_p2_deg - phi_p3_deg,
        phi_p1_deg - phi_p3_deg,
    ],
    dtype=float,
)

phase_relative_p3_360_deg = (
    phase_relative_p3_deg
    % 360.0
)


# ============================================================
# RESULTADOS
# ============================================================

print(
    "\n--- TRANSFER PHASE FOR EACH OUTPUT ---"
)

print(
    "Phase = arg(H_k,I1 * conj(H_k,I2))"
)

print(
    f"P1: {phi_p1_deg:+.4f} deg"
)

print(
    f"P2: {phi_p2_deg:+.4f} deg"
)

print(
    f"P3: {phi_p3_deg:+.4f} deg"
)

print(
    f"P4: {phi_p4_deg:+.4f} deg"
)


print(
    "\n--- PHASES RELATIVE TO P3 ---"
)

print(
    f"P4 relative to P3 = "
    f"{phase_relative_p3_360_deg[0]:.4f} deg"
)

print(
    f"P3 relative to P3 = "
    f"{phase_relative_p3_360_deg[1]:.4f} deg"
)

print(
    f"P2 relative to P3 = "
    f"{phase_relative_p3_360_deg[2]:.4f} deg"
)

print(
    f"P1 relative to P3 = "
    f"{phase_relative_p3_360_deg[3]:.4f} deg"
)


print(
    "\nIdeal P4, P3, P2, P1:"
)

print(
    "270, 0, 180, 90 deg"
)


print(
    "\n--- PHASE RELATIONS ---"
)

print(
    f"phi1 - phi3 = "
    f"{phase_13_deg:+.4f} deg "
    f"(ideal: +90 deg)"
)

print(
    f"phi2 - phi3 = "
    f"{phase_23_deg:+.4f} deg "
    f"(ideal: 180 deg)"
)

print(
    f"phi3 - phi4 = "
    f"{phase_34_deg:+.4f} deg "
    f"(ideal: +90 deg)"
)


print(
    "\n--- PHASE DEVIATIONS ---"
)

print(
    f"Delta phi13 = "
    f"{delta_phi_13_deg:+.4f} deg"
)

print(
    f"Delta phi23 = "
    f"{delta_phi_23_deg:+.4f} deg"
)

print(
    f"Delta phi34 = "
    f"{delta_phi_34_deg:+.4f} deg"
)

# REPRESENTACION DE LA PROPAGACION

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


# Inputs

for input_index, x_position_um in enumerate(
    INPUT_X_POSITIONS_UM):

    ax.axhline(
        x_position_um,
        color="magenta",
        linestyle="--",
        linewidth=0.8,
        alpha=0.85,
        label=("Inputs"if input_index == 0 else None
        ),
    )


# Outputs

for output_index, x_position_um in enumerate(
    OUTPUT_X_POSITIONS_UM
):

    ax.axhline(
        x_position_um,
        color="cyan",
        linestyle=":",
        linewidth=0.8,
        alpha=0.85,
        label=(
            "Outputs"
            if output_index == 0 else None
        ),
    )


ax.set_title(
    f"2x4 MMI 90-degree hybrid - "
    f"{INPUT_CONFIGURATION}"
)

ax.legend(
    loc="best"
)



# MATRIZ DE POTENCIA PARA LOS DOS INPUTS


output_indices = np.arange(N_OUTPUTS)

bar_width = 0.35

fig_power, ax_power = plt.subplots(figsize=(8, 5))

ax_power.bar(
    output_indices- 0.5 * bar_width,
    single_input_splitting[:, 0],
    width=bar_width,
    label="Input 0")

ax_power.bar(
    output_indices+ 0.5 * bar_width,
    single_input_splitting[:, 1],
    width=bar_width,
    label="Input 1",
)

ax_power.axhline(
    0.25,
    linestyle="--",
    linewidth=1,
    label="Ideal 25%",
)

ax_power.set_xticks(output_indices)

ax_power.set_xticklabels(
    [
        "O0",
        "O1",
        "O2",
        "O3",
    ]
)

ax_power.set_xlabel("Output port")
ax_power.set_ylabel("Splitting ratio")
ax_power.set_title("Single-input output splitting")

ax_power.grid(
    True,
    axis="y",
    alpha=0.3,
)

ax_power.legend()
fig_power.tight_layout()

fig_phase, ax_phase = plt.subplots(figsize=(8, 5))

ax_phase.plot(
    output_indices,
    relative_transfer_phase_deg,
    marker="o",
    linewidth=2,
)

ax_phase.set_xticks(output_indices)

ax_phase.set_xticklabels(
    [
        "O0",
        "O1",
        "O2",
        "O3",
    ]
)

ax_phase.set_xlabel("Output port")

ax_phase.set_ylabel( r"$\angle(H_{k1}H_{k0}^*)$ [deg]")

ax_phase.set_title("Relative transfer phase between inputs")

ax_phase.grid(True, alpha=0.3,)
fig_phase.tight_layout()


# ============================================================
# RESULTADOS
# ============================================================

print("\n"
    + "=" * 65
)

print("2x4 MMI 90-DEGREE HYBRID EME SUMMARY")

print("=" * 65)


print(f"Mode solver backend: "
    f"{MODE_SOLVER_BACKEND}")

print(f"Wavelength: "
    f"{WAVELENGTH_UM:.4f} um")

print(f"MMI width: "
    f"{MMI_WIDTH_UM:.4f} um")

print(
    f"Lpi: "
    f"{lpi_um:.6f} um"
)

print(
    f"MMI length base (3*Lpi/4): "
    f"{MMI_LENGTH_BASE_UM:.6f} um"
)

print(
    f"MMI length increment: "
    f"{MMI_LENGTH_INCREMENT_UM:+.6f} um"
)

print(
    f"MMI length: "
    f"{MMI_LENGTH_UM:.6f} um"
)

print(
    f"Lmmi / Lpi: "
    f"{MMI_LENGTH_UM / lpi_um:.6f}"
)

print(
    f"Access width: "
    f"{ACCESS_WIDTH_UM:.4f} um"
)

print(
    f"dy: "
    f"{DY:+.4f} um"
)

print(
    f"TE modes used in MMI: "
    f"{len(mmi_result)}"
)


# ============================================================
# RESPUESTA INPUT 0
# ============================================================

print(
    "\n--- INPUT 0 ---"
)

print(
    f"Position: "
    f"{INPUT_X_POSITIONS_UM[0]:+.4f} um"
)

print(
    f"Represented input power: "
    f"{represented_input_powers[0]:.6f}"
)

print(
    f"Collected output power: "
    f"{single_input_total_power[0]:.6f}"
)

print(
    f"Excess loss: "
    f"{single_input_excess_loss_db[0]:.4f} dB"
)

for output_index in range(
    N_OUTPUTS
):

    print(
        f"O{output_index}: "
        f"power="
        f"{single_input_powers[output_index, 0]:.6f}, "
        f"split="
        f"{100 * single_input_splitting[output_index, 0]:.2f} %, "
        f"phase="
        f"{transfer_phase_deg[output_index, 0]:+.2f} deg"
    )


# ============================================================
# RESPUESTA INPUT 1
# ============================================================

print(
    "\n--- INPUT 1 ---"
)

print(
    f"Position: "
    f"{INPUT_X_POSITIONS_UM[1]:+.4f} um"
)

print(
    f"Represented input power: "
    f"{represented_input_powers[1]:.6f}"
)

print(
    f"Collected output power: "
    f"{single_input_total_power[1]:.6f}"
)

print(
    f"Excess loss: "
    f"{single_input_excess_loss_db[1]:.4f} dB"
)

for output_index in range(
    N_OUTPUTS
):

    print(
        f"O{output_index}: "
        f"power="
        f"{single_input_powers[output_index, 1]:.6f}, "
        f"split="
        f"{100 * single_input_splitting[output_index, 1]:.2f} %, "
        f"phase="
        f"{transfer_phase_deg[output_index, 1]:+.2f} deg"
    )


# ============================================================
# FASE RELATIVA ENTRE LOS DOS CAMINOS
# ============================================================

print(
    "\n--- RELATIVE TRANSFER PHASE ---"
)

print(
    "phase = angle(H_k1 * conj(H_k0))"
)

for output_index, phase_deg in enumerate(
    relative_transfer_phase_deg
):

    print(
        f"O{output_index}: "
        f"{phase_deg:+.2f} deg"
    )


# ============================================================
# CONFIGURACION ACTIVA
# ============================================================

print(
    "\n--- ACTIVE CONFIGURATION ---"
)

print(
    f"Input configuration: "
    f"{INPUT_CONFIGURATION}"
)

if INPUT_CONFIGURATION == "hybrid":

    print(
        f"Input phase difference: "
        f"{DELTA_PHI_IN_DEG:+.2f} deg"
    )

print(
    f"Represented input power: "
    f"{represented_input_power:.6f}"
)

print(
    f"Collected output transmission: "
    f"{collected_output_transmission:.6f}"
)

print(
    f"Excess loss: "
    f"{excess_loss_db:.4f} dB"
)


print(
    "\n--- OUTPUT DETAILS ---"
)

for (
    output_index,
    x_position_um,
    power,
    transmission,
    split_fraction,
) in zip(
    range(N_OUTPUTS),
    OUTPUT_X_POSITIONS_UM,
    output_powers,
    transmissions,
    split_fractions,
    strict=True,
):

    print(
        f"O{output_index}: "
        f"x={x_position_um:+.4f} um, "
        f"power={power:.6f}, "
        f"T={transmission:.6f}, "
        f"split={100 * split_fraction:.2f} %"
    )

print(
    "=" * 65
)


plt.show()