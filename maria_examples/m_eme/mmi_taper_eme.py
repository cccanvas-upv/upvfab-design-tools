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
    modal_overlap_matrix,

)
from upvfab_design_tools.modesolver import (
    FemwellModeSolver,
    ModeSolverResult,
    sample_mode_profile,
    sample_mode_profiles,
)

WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.3
MMI_WIDTH_UM = 12.0
ACCESS_X_POSITIONS_UM = (-2.0, 2.0)
INPUT_X_POSITION_UM = ACCESS_X_POSITIONS_UM[0]
PAIRED_INTERFERENCE_LENGTH_FACTOR  = 0.5
DOMAIN_X_SPAN_UM = 18.0     
FIELD_CUT_Z_UM = 0.5 * CORE_HEIGHT_UM
NUM_MMI_MODES = 24
INPUT_WIDTH_START_UM = 1.0
INPUT_WIDTH_END_UM = 1.6
OUTPUT_WIDTH_START_UM = 1.6
OUTPUT_WIDTH_END_UM = 1.0
TAPER_LENGTH_UM = 10.0
NUM_SECTIONS = 8
NUM_MODES = 12
DZ_UM = 0.5

def te_mode_result(result: ModeSolverResult) -> ModeSolverResult:
    """Return a result containing only TE-like modes."""

    if not result.te_modes:
        raise RuntimeError("The mode solve returned no guided TE-like modes.")

    return ModeSolverResult(
        modes = result.te_modes, 
        wavelength_um=result.wavelength_um,
        backend=result.backend,
        cross_section= result.cross_section, 
        metadata = result.metadata,
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


#TAPER ENTRADA

input_widths_um = np.linspace(INPUT_WIDTH_START_UM, INPUT_WIDTH_END_UM, NUM_SECTIONS)
section_lengths_um = np.full(NUM_SECTIONS, TAPER_LENGTH_UM / NUM_SECTIONS)
x_um = np.linspace(-0.5 * DOMAIN_X_SPAN_UM, 0.5 * DOMAIN_X_SPAN_UM, 1024)

taper_solver = FemwellModeSolver(
    resolution_overrides = {
        "core": {"resolution": 0.05, "distance": 0.25}, 
        "background": {"resolution": 0.35, "distance": 1.0},
    },
    reference_material = THERMAL_SILICON_DIOXIDE, 
    filter_guided = True, 
    guided_tolerance=1e-2,
)

input_section_results: list[ModeSolverResult] = []
input_section_profiles: list[np.ndarray] = []

for section_index, width_um in enumerate(input_widths_um):
    print(
        f"Solving taper section {section_index + 1}/{NUM_SECTIONS}: "
        f"width = {width_um:.3f} um"
    )

    cross_section = strip_waveguide(
        width = float(width_um), 
        height=CORE_HEIGHT_UM,
        core_material=SILICON_NITRIDE,
        background_material=THERMAL_SILICON_DIOXIDE, 
        x_center=0.0,
        z_min=0.0,
        x_span=DOMAIN_X_SPAN_UM,
        bottom_margin=1.5,
        top_margin=1.5,
        name=f"taper_section_{section_index:02d}",
    )
    result = taper_solver.solve(
        cross_section,
        wavelength_um=WAVELENGTH_UM, 
        num_modes=NUM_MODES
    )
    result = te_mode_result(result)
    profiles = sample_mode_profiles(
        result, 
        x_um=x_um, 
        z_um = FIELD_CUT_Z_UM,
        field_component= "Ex",
    )

    input_section_results.append(result)
    input_section_profiles.append(normalize_profiles(profiles, x_um))

input_profile = input_section_profiles[0][0]
current_amplitudes = modal_excitation_coefficients(
    input_profile, 
    input_section_profiles[0], 
    x_um,
)

input_global_z_um: list[np.ndarray] = []
input_global_field: list[np.ndarray] = []
input_z_offset_um = 0.0

for section_index, (result, profiles, length_um) in enumerate(
    zip(input_section_results, input_section_profiles, section_lengths_um, strict=True)
): 
    propagation = propagate_modes(
        result,
        initial_amplitudes = current_amplitudes, 
        length_um = float(length_um), 
        dz_um=DZ_UM,
    )

    section_z_um = input_z_offset_um + propagation.z_um
    section_field = propagation.reconstruct_field(profiles)

    if section_index > 0:
        section_z_um = section_z_um[1:]
        section_field = section_field[1:]

    input_global_z_um.append(section_z_um)
    input_global_field.append(section_field)

    current_amplitudes = propagation.final_amplitudes
    input_z_offset_um += float(length_um)

    if section_index < NUM_SECTIONS - 1:
        coupling = modal_overlap_matrix(
            source_profiles=profiles,
            target_profiles=input_section_profiles[section_index + 1],
            x_um=x_um,
        )
        current_amplitudes = coupling @ current_amplitudes

input_z_plot_um = np.concatenate(input_global_z_um)
input_field_plot = np.vstack(input_global_field)

output_profile_input_taper = current_amplitudes @ input_section_profiles[-1]   #tmb podria haver fet: =field_plot[-1]

# Campo del taper en x = -2 um
input_field_plot_shifted = np.array(
    [
        shift_profile(
            field_profile,
            x_um,
            shift_um=INPUT_X_POSITION_UM,
        )
        for field_profile in input_field_plot
    ],
    dtype=np.complex128,
)

######## MMI

mmi_cross_section = strip_waveguide(
    width = MMI_WIDTH_UM, 
    height=CORE_HEIGHT_UM, 
    core_material=SILICON_NITRIDE, 
    background_material=THERMAL_SILICON_DIOXIDE, 
    x_center= 0.0, 
    z_min=0.0, 
    x_span= DOMAIN_X_SPAN_UM, 
    bottom_margin=1.5, 
    top_margin=1.5, 
    name = "uniform_12um_mmi_section"
)

mmi_solver = FemwellModeSolver(
    resolution_overrides={
        "core": {"resolution": 0.08, "distance": 0.3}, 
        "background": {"resolution": 0.6, "distance": 1}, 
    },
    reference_material= THERMAL_SILICON_DIOXIDE, 
    filter_guided= True, 
    guided_tolerance=1e-2,
)

mmi_result = mmi_solver.solve(
    mmi_cross_section, 
    wavelength_um = WAVELENGTH_UM, 
    num_modes= NUM_MMI_MODES,
)
mmi_result = te_mode_result(mmi_result)
lpi_um = lpi_from_first_two_modes(mmi_result)
mmi_length_um = PAIRED_INTERFERENCE_LENGTH_FACTOR * lpi_um


input_profile = shift_profile(
    output_profile_input_taper, 
    x_um, 
    shift_um= INPUT_X_POSITION_UM,
)

mmi_profiles = sample_mode_profiles(
    result = mmi_result, 
    x_um = x_um, 
    z_um = FIELD_CUT_Z_UM, 
    field_component = "Ex", 
)
mmi_profiles = normalize_profiles(mmi_profiles, x_um)

initial_amplitudes = modal_excitation_coefficients(
    input_profile= input_profile,
    mode_profiles= mmi_profiles, 
    x_um = x_um,
)

mmi_propagation = propagate_modes(
    mmi_result, 
    initial_amplitudes = initial_amplitudes, 
    length_um = mmi_length_um, 
    dz_um = DZ_UM,
)

mmi_field = mmi_propagation.reconstruct_field(mmi_profiles)
mmi_final_field = mmi_propagation.reconstruct_field(mmi_profiles)[-1]

## resolver modos del taper inverso 

output_widths_um = np.linspace(OUTPUT_WIDTH_START_UM, OUTPUT_WIDTH_END_UM, NUM_SECTIONS)
output_section_lengths_um = np.full(NUM_SECTIONS, TAPER_LENGTH_UM / NUM_SECTIONS)



output_section_results: list[ModeSolverResult] = []
output_section_profiles: list[np.ndarray] = []

for section_index, width_um in enumerate(output_widths_um):
    print(
        f"Solving taper section {section_index + 1}/{NUM_SECTIONS}: "
        f"width = {width_um:.3f} um"
    )

    cross_section = strip_waveguide(
        width = float(width_um), 
        height=CORE_HEIGHT_UM,
        core_material=SILICON_NITRIDE,
        background_material=THERMAL_SILICON_DIOXIDE, 
        x_center=0.0,
        z_min=0.0,
        x_span=DOMAIN_X_SPAN_UM,
        bottom_margin=1.5,
        top_margin=1.5,
        name=f"taper_section_{section_index:02d}",
    )
    result = taper_solver.solve(
        cross_section,
        wavelength_um=WAVELENGTH_UM, 
        num_modes=NUM_MODES
    )
    result = te_mode_result(result)
    profiles = sample_mode_profiles(
        result, 
        x_um=x_um, 
        z_um = FIELD_CUT_Z_UM,
        field_component= "Ex",
    )

    output_section_results.append(result)
    output_section_profiles.append(normalize_profiles(profiles, x_um))

DOWN_X_POSITION_UM = ACCESS_X_POSITIONS_UM[0]  
UP_X_POSITION_UM = ACCESS_X_POSITIONS_UM[1]    

#shift de todos los perfiles para taper arriba y abajo
down_section_profiles = [
    np.array(
        [
            shift_profile(profile, x_um, shift_um=DOWN_X_POSITION_UM)
            for profile in profiles
        ],
        dtype=np.complex128,
    )
    for profiles in output_section_profiles
]

up_section_profiles = [
    np.array(
        [
            shift_profile(profile,x_um, shift_um=UP_X_POSITION_UM)
            for profile in profiles
        ],
        dtype=np.complex128,
    )
    for profiles in output_section_profiles
]

down_initial_amplitudes = np.array(
    [
        scalar_projection(mmi_final_field,mode_profile, x_um)
        for mode_profile in down_section_profiles[0]
    ],
    dtype=np.complex128,
)

up_initial_amplitudes = np.array(
    [
        scalar_projection(mmi_final_field, mode_profile, x_um)
        for mode_profile in up_section_profiles[0]
    ],
    dtype=np.complex128,
)

#para no repetir el mismo código 

def propagate_output_taper(
    initial_amplitudes: np.ndarray,
    shifted_section_profiles: list[np.ndarray],
):
    current_amplitudes = initial_amplitudes.copy()

    global_z_um: list[np.ndarray] = []
    global_field: list[np.ndarray] = []
    z_offset_um = 0.0

    for section_index, (result, profiles, length_um) in enumerate(
        zip(output_section_results, shifted_section_profiles, output_section_lengths_um, strict=True)
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
                target_profiles=shifted_section_profiles[section_index + 1],
                x_um=x_um,
            )

            current_amplitudes = coupling @ current_amplitudes

    return (
        np.concatenate(global_z_um),
        np.vstack(global_field),
        current_amplitudes,
    )

down_z_um, down_field, down_final_amplitudes = propagate_output_taper(down_initial_amplitudes, down_section_profiles)

up_z_um, up_field, up_final_amplitudes = propagate_output_taper( up_initial_amplitudes, up_section_profiles)

output_field = up_field + down_field

#coordenadas globales
z_input_end = TAPER_LENGTH_UM
z_mmi_end = z_input_end + mmi_length_um
z_device_end = z_mmi_end + TAPER_LENGTH_UM
mmi_z_global = z_input_end + mmi_propagation.z_um
output_z_global = z_mmi_end + up_z_um

#quitar los puntos que coinciden 
device_z_um = np.concatenate(
    [
        input_z_plot_um,
        mmi_z_global[1:],
        output_z_global[1:],
    ]
)

device_field = np.vstack(
    [
        input_field_plot_shifted,
        mmi_field[1:],
        output_field[1:],
    ]
)

device_intensity = np.abs(device_field) ** 2

fig, ax = plt.subplots(figsize=(12, 5))

image = ax.pcolormesh(
    device_z_um,
    x_um,
    device_intensity.T,
    shading="auto",
    cmap="inferno",
)


fig, ax = plt.subplots(figsize=(12, 5))

image = ax.pcolormesh(
    device_z_um,
    x_um,
    device_intensity.T,
    shading="auto",
    cmap="inferno",
)

# taper entrada

input_center_x = INPUT_X_POSITION_UM

ax.plot(
    [0.0, z_input_end],
    [
        input_center_x - 0.5 * INPUT_WIDTH_START_UM,
        input_center_x - 0.5 * INPUT_WIDTH_END_UM,
    ],
    color="white",
    linewidth=0.9,
    alpha=0.8,
)

ax.plot(
    [0.0, z_input_end],
    [
        input_center_x + 0.5 * INPUT_WIDTH_START_UM,
        input_center_x + 0.5 * INPUT_WIDTH_END_UM,
    ],
    color="white",
    linewidth=0.9,
    alpha=0.8,
)


# -mmi

ax.plot(
    [z_input_end, z_mmi_end],
    [-0.5 * MMI_WIDTH_UM, -0.5 * MMI_WIDTH_UM],
    color="white",
    linewidth=0.9,
    alpha=0.8,
)

ax.plot(
    [z_input_end, z_mmi_end],
    [0.5 * MMI_WIDTH_UM, 0.5 * MMI_WIDTH_UM],
    color="white",
    linewidth=0.9,
    alpha=0.8,
)


# tapers salida

for x_center in (DOWN_X_POSITION_UM, UP_X_POSITION_UM):

    ax.plot(
        [z_mmi_end, z_device_end],
        [
            x_center - 0.5 * OUTPUT_WIDTH_START_UM,
            x_center - 0.5 * OUTPUT_WIDTH_END_UM,
        ],
        color="white",
        linewidth=0.9,
        alpha=0.8,
    )

    ax.plot(
        [z_mmi_end, z_device_end],
        [
            x_center + 0.5 * OUTPUT_WIDTH_START_UM,
            x_center + 0.5 * OUTPUT_WIDTH_END_UM,
        ],
        color="white",
        linewidth=0.9,
        alpha=0.8,
    )


# Separación visual entre las tres regiones
ax.axvline(
    z_input_end,
    color="cyan",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7,
)

ax.axvline(
    z_mmi_end,
    color="cyan",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7,
)


ax.set_xlabel("z [µm]")
ax.set_ylabel("x [µm]")
ax.set_xlim(0.0, z_device_end)
ax.set_ylim(
    -0.5 * DOMAIN_X_SPAN_UM,
    0.5 * DOMAIN_X_SPAN_UM,
)

ax.set_title("Complete EME device propagation |E(x, z)|²")

fig.colorbar(
    image,
    ax=ax,
    label="Field intensity |E|²",
)




# ============================================================
# DATOS RELEVANTES
# ============================================================

# ------------------------------------------------------------
# Taper de entrada
# ------------------------------------------------------------

input_taper_final_fundamental_power = float(
    np.abs(current_amplitudes[0]) ** 2
)

input_taper_final_represented_power = float(
    np.sum(np.abs(current_amplitudes) ** 2)
)


# ------------------------------------------------------------
# MMI
# ------------------------------------------------------------

mmi_input_represented_power = float(
    np.sum(np.abs(initial_amplitudes) ** 2)
)

mmi_final_represented_power = float(
    np.sum(np.abs(mmi_propagation.final_amplitudes) ** 2)
)


# ------------------------------------------------------------
# Acoplamiento MMI -> tapers de salida
# ------------------------------------------------------------

down_input_power = float(
    np.sum(np.abs(down_initial_amplitudes) ** 2)
)

up_input_power = float(
    np.sum(np.abs(up_initial_amplitudes) ** 2)
)

collected_output_power = down_input_power + up_input_power

if collected_output_power > 0:
    down_split_fraction = down_input_power / collected_output_power
    up_split_fraction = up_input_power / collected_output_power
else:
    down_split_fraction = 0.0
    up_split_fraction = 0.0


# ------------------------------------------------------------
# Final de los tapers de salida
# ------------------------------------------------------------

down_final_fundamental_power = float(
    np.abs(down_final_amplitudes[0]) ** 2
)

up_final_fundamental_power = float(
    np.abs(up_final_amplitudes[0]) ** 2
)

down_final_represented_power = float(
    np.sum(np.abs(down_final_amplitudes) ** 2)
)

up_final_represented_power = float(
    np.sum(np.abs(up_final_amplitudes) ** 2)
)

total_final_represented_power = (
    down_final_represented_power
    + up_final_represented_power
)


# Fracción de la potencia de cada salida que acaba en TE0
down_final_te0_fraction = (
    down_final_fundamental_power / down_final_represented_power
    if down_final_represented_power > 0
    else 0.0
)

up_final_te0_fraction = (
    up_final_fundamental_power / up_final_represented_power
    if up_final_represented_power > 0
    else 0.0
)


# ------------------------------------------------------------
# PRINTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EME DEVICE SUMMARY")
print("=" * 60)

print("\n--- GEOMETRY ---")
print(f"Wavelength: {WAVELENGTH_UM:.3f} um")
print(f"Input taper length: {TAPER_LENGTH_UM:.6f} um")
print(f"MMI length: {mmi_length_um:.6f} um")
print(f"Output taper length: {TAPER_LENGTH_UM:.6f} um")
print(f"Total device length: {z_device_end:.6f} um")
print(f"Lpi: {lpi_um:.6f} um")


print("\n--- INPUT TAPER ---")
print(
    "Final fundamental TE power: "
    f"{input_taper_final_fundamental_power:.6f}"
)
print(
    "Final represented scalar power: "
    f"{input_taper_final_represented_power:.6f}"
)


print("\n--- MMI ---")
print(f"TE modes used: {len(mmi_result)}")
print(
    "Represented input power: "
    f"{mmi_input_represented_power:.6f}"
)
print(
    "Represented final power: "
    f"{mmi_final_represented_power:.6f}"
)


print("\n--- MMI -> OUTPUT TAPERS ---")
print(f"Down collected power: {down_input_power:.6f}")
print(f"Up collected power:   {up_input_power:.6f}")
print(f"Total collected power: {collected_output_power:.6f}")

print(
    f"Down split fraction: {down_split_fraction:.6f} "
    f"({100 * down_split_fraction:.2f} %)"
)
print(
    f"Up split fraction:   {up_split_fraction:.6f} "
    f"({100 * up_split_fraction:.2f} %)"
)


print("\n--- FINAL OUTPUT TAPERS ---")

print(
    "Down final fundamental TE power: "
    f"{down_final_fundamental_power:.6f}"
)
print(
    "Down final represented power: "
    f"{down_final_represented_power:.6f}"
)
print(
    "Down TE0 fraction: "
    f"{down_final_te0_fraction:.6f} "
    f"({100 * down_final_te0_fraction:.2f} %)"
)

print(
    "Up final fundamental TE power: "
    f"{up_final_fundamental_power:.6f}"
)
print(
    "Up final represented power: "
    f"{up_final_represented_power:.6f}"
)
print(
    "Up TE0 fraction: "
    f"{up_final_te0_fraction:.6f} "
    f"({100 * up_final_te0_fraction:.2f} %)"
)

print(
    "Total final represented output power: "
    f"{total_final_represented_power:.6f}"
)

print("=" * 60)

plt.show()
