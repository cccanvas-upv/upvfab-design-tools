from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib import pyplot as plt

from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
    BCB, 
    AIR,
)

from upvfab_design_tools.core.waveguides import (
    strip_waveguide_bcb,
)

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


# ============================================================
# CONFIGURACION GENERAL
# ============================================================

MODE_SOLVER_BACKEND: Literal[
    "femwell",
    "tidy",
] = "tidy"


WAVELENGTH_UM = 1.55

CORE_HEIGHT_UM = 0.30
BCB_HEIGHT_UM = 0.4


# ============================================================
# MMI 3x3 - GENERAL INTERFERENCE
# ============================================================

N_PORTS = 3

MMI_WIDTH_UM = 10.0


# Correccion respecto a la longitud teorica:
#
# L_MMI = 3 * Lpi / N
#
# Para N = 3:
#
# L_MMI = Lpi

MMI_LENGTH_INCREMENT_UM = -3


# ============================================================
# GUIAS / TAPERS
# ============================================================

# Guia externa

ACCESS_WIDTH_UM = 1


# Anchura del taper en la interfaz con el MMI.
#
# El EME simplificado no propaga longitudinalmente
# por el taper, por lo que utilizamos esta anchura
# para generar el perfil modal que entra en el MMI.

TAPER_WIDTH_UM = 1.4

TAPER_LENGTH_UM = 15.0


# ============================================================
# POSICIONES DE LOS PUERTOS
# ============================================================

# Punto de partida para GI 3x3:
#
# y = -W/3, 0, +W/3
#
# PORT_POSITION_OFFSET_UM permite optimizar ligeramente
# la posicion de los dos puertos exteriores.

PORT_POSITION_BASE_UM = (
    MMI_WIDTH_UM / 3.0
)


PORT_POSITION_OFFSET_UM = 0.06


PORT_POSITION_UM = (
    PORT_POSITION_BASE_UM
    + PORT_POSITION_OFFSET_UM
)


PORT_POSITIONS_UM = np.array(
    [
        -PORT_POSITION_UM,
        0.0,
        +PORT_POSITION_UM,
    ],
    dtype=float,
)


PORT_NAMES = (
    "bottom",
    "center",
    "top",
)


# Gap geometrico resultante.

IMPLIED_GAP_UM = (
    PORT_POSITION_UM
    - TAPER_WIDTH_UM
)


# ============================================================
# ENTRADAS DEL HYBRID
# ============================================================

# Se excitan simultaneamente:
#
#     TOP
#     BOTTOM
#
# El puerto central de entrada queda sin excitar.
#
# Convencion:
#
# E_in = A_top * E_top
#      + A_bottom * exp(i * DeltaPhi) * E_bottom


TOP_INPUT_AMPLITUDE = 1.0

BOTTOM_INPUT_AMPLITUDE = 1.0


# Fase de BOTTOM respecto a TOP

DELTA_PHI_IN_DEG = 0.0


DELTA_PHI_IN_RAD = np.deg2rad(
    DELTA_PHI_IN_DEG
)


# ============================================================
# CONTROLES NUMERICOS
# ============================================================

DZ_UM = 0.25


DOMAIN_X_SPAN_UM = (
    MMI_WIDTH_UM
    + 4.0
)


FIELD_CUT_Z_UM = (
    0.5
    * CORE_HEIGHT_UM
)


NUM_PORT_MODES = 8

NUM_MMI_MODES = 24

NUM_X_POINTS = 1024


# ============================================================
# MODE SOLVER
# ============================================================

def make_mode_solver():

    if MODE_SOLVER_BACKEND == "femwell":

        return FemwellModeSolver(
            default_resolution=0.3,
            min_resolution=0.03,
            resolution_factor=6,
            filter_guided=True,
            reference_material=(
                THERMAL_SILICON_DIOXIDE
            ),
            guided_tolerance=1e-2,
            enable_plots=False,
        )


    if MODE_SOLVER_BACKEND == "tidy":

        from upvfab_design_tools.modesolver import (
            Tidy3DModeSolver,
        )

        return Tidy3DModeSolver(
            min_steps_per_wvl=45,
            reference_material=(
                THERMAL_SILICON_DIOXIDE
            ),
            filter_guided=True,
            guided_tolerance=1e-2,
        )


    raise ValueError(
        "MODE_SOLVER_BACKEND must be "
        "'femwell' or 'tidy'."
    )


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def te_mode_result(
    result: ModeSolverResult,
) -> ModeSolverResult:

    if not result.te_modes:

        raise RuntimeError(
            "The mode solve returned "
            "no guided TE-like modes."
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

    mode_profile = normalize_profile(
        mode_profile,
        x_um,
    )


    return complex(
        np.trapezoid(
            np.conjugate(
                mode_profile
            )
            * field_profile,
            x_um,
        )
    )


def lpi_from_first_two_modes(
    result: ModeSolverResult,
) -> float:

    if len(result) < 2:

        raise RuntimeError(
            "At least two TE modes are "
            "required to estimate Lpi."
        )


    beta0 = complex(
        result[0].beta
    )

    beta1 = complex(
        result[1].beta
    )


    delta_beta = abs(
        float(
            np.real(
                beta0
                - beta1
            )
        )
    )


    if delta_beta <= 0:

        raise RuntimeError(
            "Could not estimate Lpi because "
            "beta0 and beta1 match."
        )


    return float(
        np.pi
        / delta_beta
    )


def wrap_phase_deg(
    phase_deg: np.ndarray,
) -> np.ndarray:

    return (
        phase_deg
        + 180.0
    ) % 360.0 - 180.0


# ============================================================
# CROSS SECTION DEL PUERTO EN LA INTERFAZ DEL MMI
# ============================================================

port_cross_section = strip_waveguide_bcb(
    width=ACCESS_WIDTH_UM,
    height=CORE_HEIGHT_UM,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    top_material= BCB, 
    air = AIR,
    top_material_thickness=BCB_HEIGHT_UM,
    x_center=0.0,
    z_min=0.0,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=1.5,
    top_margin=1.5,
    name="centered_access_strip",
)


mmi_cross_section = strip_waveguide_bcb(
    width=MMI_WIDTH_UM,
    height=CORE_HEIGHT_UM,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    top_material= BCB, 
    air = AIR,
    top_material_thickness=BCB_HEIGHT_UM, 
    x_center=0.0,
    z_min=0.0,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=1.5,
    top_margin=1.5,
    name="uniform_mmi_section",
)



# ============================================================
# RESOLVER MODOS
# ============================================================

solver = make_mode_solver()


print(
    f"Mode solver backend: "
    f"{MODE_SOLVER_BACKEND}",
    flush=True,
)


# ============================================================
# MODO DEL PUERTO
# ============================================================

print(
    f"Solving fundamental mode of "
    f"{TAPER_WIDTH_UM:.3f} um port...",
    flush=True,
)


port_result = solver.solve(
    port_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=NUM_PORT_MODES,
)


if not port_result.te_modes:

    raise RuntimeError(
        "The port solve returned "
        "no guided TE mode."
    )


port_mode = (
    port_result.te_modes[0]
)


# ============================================================
# MODOS DEL MMI
# ============================================================

print(
    f"Solving modes of the "
    f"{MMI_WIDTH_UM:.3f} um "
    f"multimode section...",
    flush=True,
)


mmi_result = solver.solve(
    mmi_cross_section,
    wavelength_um=WAVELENGTH_UM,
    num_modes=NUM_MMI_MODES,
)


mmi_result = te_mode_result(
    mmi_result
)


# ============================================================
# Lpi
# ============================================================

lpi_um = lpi_from_first_two_modes(
    mmi_result
)


# ============================================================
# LONGITUD GENERAL NxN
# ============================================================

MMI_LENGTH_BASE_UM = (
    3.0
    * lpi_um
    / N_PORTS
)


MMI_LENGTH_UM = (
    MMI_LENGTH_BASE_UM
    + MMI_LENGTH_INCREMENT_UM
)


# ============================================================
# EJE TRANSVERSAL
# ============================================================

x_um = np.linspace(
    -0.5 * DOMAIN_X_SPAN_UM,
    +0.5 * DOMAIN_X_SPAN_UM,
    NUM_X_POINTS,
)


# ============================================================
# PERFIL FUNDAMENTAL CENTRADO
# ============================================================

centered_port_profile = sample_mode_profile(
    port_mode,
    x_um=x_um,
    z_um=FIELD_CUT_Z_UM,
    field_component="Ex",
)


centered_port_profile = normalize_profile(
    centered_port_profile,
    x_um,
)


# ============================================================
# PERFILES DE LOS TRES PUERTOS
# ============================================================

port_profiles = [

    shift_profile(
        centered_port_profile,
        x_um,
        shift_um=position_um,
    )

    for position_um
    in PORT_POSITIONS_UM
]


bottom_profile = (
    port_profiles[0]
)

center_profile = (
    port_profiles[1]
)

top_profile = (
    port_profiles[2]
)


# ============================================================
# MODOS DEL MMI
# ============================================================

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


# ============================================================
# FUNCION PARA PROPAGAR UNA EXCITACION
# ============================================================

def propagate_input(
    input_profile: np.ndarray,
):

    initial_amplitudes = (
        modal_excitation_coefficients(
            input_profile,
            mmi_profiles,
            x_um,
        )
    )


    represented_input_power = float(
        np.sum(
            np.abs(
                initial_amplitudes
            ) ** 2
        )
    )


    propagation = propagate_modes(
        mmi_result,
        initial_amplitudes=(
            initial_amplitudes
        ),
        length_um=MMI_LENGTH_UM,
        dz_um=DZ_UM,
    )


    final_field = (
        propagation
        .reconstruct_field(
            mmi_profiles
        )[-1]
    )


    output_amplitudes = np.array(
        [
            scalar_projection(
                final_field,
                output_profile,
                x_um,
            )

            for output_profile
            in port_profiles
        ],
        dtype=np.complex128,
    )


    output_powers = (
        np.abs(
            output_amplitudes
        ) ** 2
    )


    if represented_input_power > 0:

        transmissions = (
            output_powers
            / represented_input_power
        )

    else:

        transmissions = np.zeros(
            N_PORTS
        )


    collected_transmission = float(
        np.sum(
            transmissions
        )
    )


    if collected_transmission > 0:

        split_fractions = (
            transmissions
            / collected_transmission
        )

        excess_loss_db = (
            -10.0
            * np.log10(
                collected_transmission
            )
        )

    else:

        split_fractions = np.zeros(
            N_PORTS
        )

        excess_loss_db = np.inf


    return {
        "initial_amplitudes": (
            initial_amplitudes
        ),

        "represented_input_power": (
            represented_input_power
        ),

        "propagation": (
            propagation
        ),

        "final_field": (
            final_field
        ),

        "output_amplitudes": (
            output_amplitudes
        ),

        "output_powers": (
            output_powers
        ),

        "transmissions": (
            transmissions
        ),

        "split_fractions": (
            split_fractions
        ),

        "collected_transmission": (
            collected_transmission
        ),

        "excess_loss_db": (
            excess_loss_db
        ),
    }


# ============================================================
# RESPUESTAS BASE DEL HYBRID
# ============================================================

# Estas dos propagaciones NO representan tres excitaciones
# fisicas simultaneas.
#
# Se calculan por separado solo para obtener las dos columnas
# de transferencia del hybrid y poder medir el desfase 120 deg.


top_basis = propagate_input(
    top_profile
)


bottom_basis = propagate_input(
    bottom_profile
)


# ============================================================
# COLUMNAS DE TRANSFERENCIA NORMALIZADAS
# ============================================================

h_top = (
    top_basis["output_amplitudes"]
    / np.sqrt(
        top_basis[
            "represented_input_power"
        ]
    )
)


h_bottom = (
    bottom_basis["output_amplitudes"]
    / np.sqrt(
        bottom_basis[
            "represented_input_power"
        ]
    )
)


# ============================================================
# DIAGNOSTICO 33:33:33 PARA CADA INPUT
# ============================================================

top_split = (
    np.abs(
        h_top
    ) ** 2
)


top_split = (
    top_split
    / np.sum(
        top_split
    )
)


bottom_split = (
    np.abs(
        h_bottom
    ) ** 2
)


bottom_split = (
    bottom_split
    / np.sum(
        bottom_split
    )
)


# ============================================================
# FASES DEL HYBRID
# ============================================================

# Para cada output:
#
# theta_k = arg(
#     H_top,k * conj(H_bottom,k)
# )
#
# En un hybrid ideal estas tres fases estan separadas
# aproximadamente 120 grados.

hybrid_phase_offsets_deg = np.degrees(
    np.angle(
        h_top
        * np.conj(
            h_bottom
        )
    )
)


hybrid_phase_offsets_deg = wrap_phase_deg(
    hybrid_phase_offsets_deg
)


# ============================================================
# SEPARACION CIRCULAR ENTRE LAS TRES FASES
# ============================================================

phase_mod_360 = (
    hybrid_phase_offsets_deg
    % 360.0
)


phase_sorted = np.sort(
    phase_mod_360
)


phase_spacings_deg = np.array(
    [
        phase_sorted[1]
        - phase_sorted[0],

        phase_sorted[2]
        - phase_sorted[1],

        phase_sorted[0]
        + 360.0
        - phase_sorted[2],
    ]
)


# ============================================================
# EXCITACION FISICA:
# TOP + BOTTOM
# ============================================================

input_norm = np.sqrt(
    abs(
        TOP_INPUT_AMPLITUDE
    ) ** 2
    +
    abs(
        BOTTOM_INPUT_AMPLITUDE
    ) ** 2
)


input_profile = (
    TOP_INPUT_AMPLITUDE
    * top_profile

    +

    BOTTOM_INPUT_AMPLITUDE
    * np.exp(
        1j
        * DELTA_PHI_IN_RAD
    )
    * bottom_profile
)


input_profile = (
    input_profile
    / input_norm
)


# ============================================================
# PROPAGACION DE LA EXCITACION REAL DEL HYBRID
# ============================================================

hybrid_result = propagate_input(
    input_profile
)


# ============================================================
# FASES DE LOS OUTPUTS PARA ESA DELTA PHI
# ============================================================

hybrid_output_amplitudes = (
    hybrid_result[
        "output_amplitudes"
    ]
)


hybrid_output_phases_deg = np.degrees(
    np.angle(
        hybrid_output_amplitudes
    )
)


# Fase relativa respecto al output central

hybrid_relative_phases_deg = np.degrees(
    np.angle(
        hybrid_output_amplitudes
        * np.conj(
            hybrid_output_amplitudes[1]
        )
    )
)


hybrid_relative_phases_deg = wrap_phase_deg(
    hybrid_relative_phases_deg
)


# ============================================================
# REPRESENTACION DE LA PROPAGACION
# ============================================================

fig, ax = plot_propagation(
    hybrid_result["propagation"],
    x_um=x_um,
    field_profiles=mmi_profiles,
    xlim=(
        -0.5
        * DOMAIN_X_SPAN_UM,

        +0.5
        * DOMAIN_X_SPAN_UM,
    ),
    zlim=(
        0.0,
        MMI_LENGTH_UM,
    ),
)


# Bordes del MMI

ax.axhline(
    -0.5
    * MMI_WIDTH_UM,
    color="white",
    linewidth=0.8,
    alpha=0.75,
)


ax.axhline(
    +0.5
    * MMI_WIDTH_UM,
    color="white",
    linewidth=0.8,
    alpha=0.75,
)


# Posiciones de puertos

for position_um in PORT_POSITIONS_UM:

    ax.axhline(
        position_um,
        color="cyan",
        linestyle="--",
        linewidth=0.8,
        alpha=0.85,
    )


ax.set_title(
    "3x3 120-degree hybrid | "
    f"Delta phi in = "
    f"{DELTA_PHI_IN_DEG:+.1f} deg"
)


# ============================================================
# BARRIDO DE FASE ENTRE LAS DOS ENTRADAS
# ============================================================

PHASE_SWEEP_DEG = np.linspace(
    -180.0,
    +180.0,
    361,
)


PHASE_SWEEP_RAD = np.deg2rad(
    PHASE_SWEEP_DEG
)


# Por linealidad:
#
# Eout(phi) =
#
# (
#   H_top
#   +
#   exp(i*phi) H_bottom
# ) / sqrt(2)
#
# para amplitudes iguales.

phase_sweep_outputs = (
    h_top[
        np.newaxis,
        :
    ]

    +

    np.exp(
        1j
        * PHASE_SWEEP_RAD[
            :,
            np.newaxis,
        ]
    )

    * h_bottom[
        np.newaxis,
        :
    ]
) / np.sqrt(2.0)


phase_sweep_powers = (
    np.abs(
        phase_sweep_outputs
    ) ** 2
)


# ============================================================
# PLOT DEL HYBRID 120 DEG
# ============================================================

fig_phase, ax_phase = plt.subplots(
    figsize=(
        8,
        5,
    )
)


for output_index, output_name in enumerate(
    PORT_NAMES
):

    ax_phase.plot(
        PHASE_SWEEP_DEG,
        phase_sweep_powers[
            :,
            output_index,
        ],
        label=output_name,
    )


ax_phase.set_xlabel(
    "Input phase difference "
    r"$\Delta\phi$ (deg)"
)


ax_phase.set_ylabel(
    "Output power"
)


ax_phase.set_title(
    "3x3 hybrid response vs input phase"
)


ax_phase.grid(
    True,
    alpha=0.3,
)


ax_phase.legend()


fig_phase.tight_layout()


# ============================================================
# RESULTADOS
# ============================================================

print(
    "\n"
    + "=" * 70
)


print(
    "3x3 MMI - 120 DEGREE OPTICAL HYBRID"
)


print(
    "=" * 70
)


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
    f"Lpi: "
    f"{lpi_um:.6f} um"
)


print(
    f"General-interference base length "
    f"(3*Lpi/N): "
    f"{MMI_LENGTH_BASE_UM:.6f} um"
)


print(
    f"Length increment: "
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
    "\n--- PORT GEOMETRY ---"
)


print(
    f"Taper width at MMI: "
    f"{TAPER_WIDTH_UM:.4f} um"
)


print(
    f"Base port position W/3: "
    f"{PORT_POSITION_BASE_UM:.4f} um"
)


print(
    f"Port position offset: "
    f"{PORT_POSITION_OFFSET_UM:+.4f} um"
)


print(
    f"Outer port position: "
    f"+/- {PORT_POSITION_UM:.4f} um"
)


print(
    f"Port positions: "
    f"{PORT_POSITIONS_UM}"
)


print(
    f"Implied geometric gap: "
    f"{IMPLIED_GAP_UM:.4f} um"
)


# ============================================================
# DIAGNOSTICO DE CADA COLUMNA
# ============================================================

print(
    "\n--- SINGLE-INPUT COLUMN CHECK ---"
)


print(
    "TOP input splitting [%]:"
)


for name, value in zip(
    PORT_NAMES,
    top_split,
    strict=True,
):

    print(
        f"  {name:>6s}: "
        f"{100 * value:.3f} %"
    )


print(
    "\nBOTTOM input splitting [%]:"
)


for name, value in zip(
    PORT_NAMES,
    bottom_split,
    strict=True,
):

    print(
        f"  {name:>6s}: "
        f"{100 * value:.3f} %"
    )


# ============================================================
# DIAGNOSTICO DE FASE 120 DEG
# ============================================================

print(
    "\n--- HYBRID PHASE OFFSETS ---"
)


for name, phase in zip(
    PORT_NAMES,
    hybrid_phase_offsets_deg,
    strict=True,
):

    print(
        f"  {name:>6s}: "
        f"{phase:+.3f} deg"
    )


print(
    "\nCircular phase spacings:"
)


for spacing in phase_spacings_deg:

    print(
        f"  {spacing:.3f} deg"
    )


print(
    "Ideal hybrid spacing: "
    "120 deg / 120 deg / 120 deg"
)


# ============================================================
# EXCITACION SIMULTANEA
# ============================================================

print(
    "\n--- SIMULTANEOUS TOP + BOTTOM INPUT ---"
)


print(
    f"TOP amplitude: "
    f"{TOP_INPUT_AMPLITUDE:.4f}"
)


print(
    f"BOTTOM amplitude: "
    f"{BOTTOM_INPUT_AMPLITUDE:.4f}"
)


print(
    f"Input phase difference "
    f"(BOTTOM relative to TOP): "
    f"{DELTA_PHI_IN_DEG:+.3f} deg"
)


print(
    f"Represented input power: "
    f"{hybrid_result['represented_input_power']:.6f}"
)


print(
    f"Collected output transmission: "
    f"{hybrid_result['collected_transmission']:.6f}"
)


print(
    f"Excess loss: "
    f"{hybrid_result['excess_loss_db']:.4f} dB"
)


print(
    "\nOutputs:"
)


for (
    name,
    position_um,
    power,
    transmission,
    split,
    phase,
    relative_phase,
) in zip(
    PORT_NAMES,
    PORT_POSITIONS_UM,
    hybrid_result[
        "output_powers"
    ],
    hybrid_result[
        "transmissions"
    ],
    hybrid_result[
        "split_fractions"
    ],
    hybrid_output_phases_deg,
    hybrid_relative_phases_deg,
    strict=True,
):

    print(
        f"{name:>6s} | "
        f"y={position_um:+.4f} um | "
        f"P={power:.6f} | "
        f"T={transmission:.6f} | "
        f"split={100 * split:.2f}% | "
        f"phase={phase:+.2f} deg | "
        f"relative={relative_phase:+.2f} deg"
    )


print(
    "\n"
    + "=" * 70
)


plt.show()