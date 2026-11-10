from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td

from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)

from upvfab_design_tools.core.waveguides import strip_waveguide

from upvfab_design_tools.tidy3d_plugin import (
    build_mmi_2x4_fdtd_simulation,
    estimate_tidy3d_cost,
    load_tidy3d_simulation_data,
    mmi_2x4_vertices,
    plot_fdtd_field_xy,
    test_tidy3d_api,
)

RUN_TIDY3D_CLOUD = True
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False

INPUT_CONFIGURATION = "hybrid"  # "top", "bottom", "hybrid"
DELTA_PHI_IN_DEG = 0.0

MIN_STEPS_PER_WVL = 16



TIDY3D_TASK_BASE = ("mmi_2x4_Lminus4.8_dy0_wt3_mins16_sweep")
TIDY3D_TASK_NAME = (f"{TIDY3D_TASK_BASE}_{INPUT_CONFIGURATION}")
TIDY3D_DATA_PATH = (f"data/{TIDY3D_TASK_NAME}.hdf5")
TOP_DATA_PATH = (f"data/{TIDY3D_TASK_BASE}_top.hdf5")
BOTTOM_DATA_PATH = (f"data/{TIDY3D_TASK_BASE}_bottom.hdf5")
HYBRID_DATA_PATH = (f"data/{TIDY3D_TASK_BASE}_hybrid.hdf5")


# WAVELENGTH
WAVELENGTH_UM = 1.55
WAVELENGTH_MIN_UM = 1.50
WAVELENGTH_MAX_UM = 1.60

NUM_WAVELENGTHS = 101

WAVELENGTHS_UM = np.linspace(
    WAVELENGTH_MIN_UM,
    WAVELENGTH_MAX_UM,
    NUM_WAVELENGTHS,
)

FREQS = (td.C_0/ WAVELENGTHS_UM)
BANDWIDTH_UM = (WAVELENGTH_MAX_UM- WAVELENGTH_MIN_UM)

# PARAMETROS DEL DISPOSITIVO
CORE_HEIGHT_UM = 0.30
MMI_WIDTH_UM = 16.0

MMI_LENGTH_UM = 281.350472 - 4.8

ACCESS_WIDTH_UM = 1.0

TAPER_WIDTH_UM = 3.0
TAPER_LENGTH_UM = 20.0

STRAIGHT_IO_LENGTH_UM = 2.0

IO_LENGTH_UM = (TAPER_LENGTH_UM + STRAIGHT_IO_LENGTH_UM)
DY = 0.0


# ============================================================
# POSICIONES DE LOS PUERTOS
#
#               TOP
#
# P1 = +3W/8
# P2 = +W/8        I1 = top
# P3 = -W/8
# P4 = -3W/8       I2 = bottom
#
#              BOTTOM
#
# input_y_positions_um = (top, bottom)
# output_y_positions_um = (P1, P2, P3, P4)
# ============================================================

INPUT_Y_POSITIONS_UM = (
    +MMI_WIDTH_UM / 8.0 + DY,
    -3.0 * MMI_WIDTH_UM / 8.0 - DY,
)

OUTPUT_Y_POSITIONS_UM = (
    +3.0 * MMI_WIDTH_UM / 8.0 - DY,
    +MMI_WIDTH_UM / 8.0 + DY,
    -MMI_WIDTH_UM / 8.0 - DY,
    -3.0 * MMI_WIDTH_UM / 8.0 + DY,
)

# SOURCE / MONITORS


SOURCE_OFFSET_UM = ( 0.5 * STRAIGHT_IO_LENGTH_UM)
MONITOR_OFFSET_UM = (0.5 * STRAIGHT_IO_LENGTH_UM)
MODE_PLANE_Y_SPAN_UM = (2.5 * ACCESS_WIDTH_UM)
DOMAIN_X_SPAN_UM = (MMI_WIDTH_UM + 4.0)

# STACK


cross_section = strip_waveguide(
    width=ACCESS_WIDTH_UM,
    height=CORE_HEIGHT_UM,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    x_center=0.0,
    z_min=0.0,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=1.5,
    top_margin=1.5,
    name="sin_fdtd_stack",
)


# GEOMETRIA


polygons = mmi_2x4_vertices(
    length_um=MMI_LENGTH_UM,
    io_length_um=IO_LENGTH_UM,
    straight_io_length_um=STRAIGHT_IO_LENGTH_UM,
    input_y_positions_um=INPUT_Y_POSITIONS_UM,
    output_y_positions_um=OUTPUT_Y_POSITIONS_UM,
    access_width_um=ACCESS_WIDTH_UM,
    taper_width_um=TAPER_WIDTH_UM,
    mmi_width_um=MMI_WIDTH_UM,
)


# SIMULACION

sim = build_mmi_2x4_fdtd_simulation(
    polygons=polygons,
    cross_section=cross_section,
    wavelength_um=WAVELENGTH_UM,
    input_configuration=INPUT_CONFIGURATION,
    input_y_positions_um=INPUT_Y_POSITIONS_UM,
    output_y_positions_um=OUTPUT_Y_POSITIONS_UM,
    access_width_um=ACCESS_WIDTH_UM,
    delta_phi_rad=np.deg2rad(DELTA_PHI_IN_DEG),
    bandwidth_um=BANDWIDTH_UM,
    pad_x_um=2.0,
    pad_y_um=2.0,
    pad_z_um=0.0,
    source_offset_um=SOURCE_OFFSET_UM,
    monitor_offset_um=MONITOR_OFFSET_UM,
    mode_plane_y_span_um=MODE_PLANE_Y_SPAN_UM,
    min_steps_per_wvl=MIN_STEPS_PER_WVL,
    run_time_s=5e-12,
)



# MONITORES ESPECTRALES
#
# field_core permanece solo a lambda0

SPECTRAL_MONITORS = (
    "flux_reflection_top",
    "mode_reflection_top",
    "flux_reflection_bottom",
    "mode_reflection_bottom",
    "flux_p1",
    "flux_p2",
    "flux_p3",
    "flux_p4",
    "mode_p1",
    "mode_p2",
    "mode_p3",
    "mode_p4",
)

updated_monitors = []

for monitor in sim.monitors:

    if monitor.name in SPECTRAL_MONITORS:

        updated_monitors.append(
            monitor.updated_copy(
                freqs=FREQS
            )
        )

    else:

        updated_monitors.append(
            monitor
        )

sim = sim.updated_copy(
    monitors=updated_monitors
)


sim.validate_pre_upload()


# PLOT GEOMETRIA


fig, ax = plt.subplots(figsize=(12, 4))

sim.plot(z=0.5 * CORE_HEIGHT_UM, ax=ax)
ax.set_aspect("equal", adjustable="box")
ax.set_title(f"MMI 2x4 - {INPUT_CONFIGURATION}")


fig, ax = plt.subplots(figsize=(12, 4))
sim.plot_eps(z=0.5 * CORE_HEIGHT_UM,ax=ax)
ax.set_aspect("equal", adjustable="box",)
ax.set_title("MMI 2x4 - permittivity")

# INFORMACION

print(
    "\n"
    + "=" * 65
)

print("FDTD MMI 2x4 - WAVELENGTH SWEEP")

print("=" * 65)

print(f"Input configuration: "
    f"{INPUT_CONFIGURATION}")

print(f"Wavelength sweep: "
    f"{WAVELENGTH_MIN_UM:.3f} - "
    f"{WAVELENGTH_MAX_UM:.3f} um")

print(f"Number of wavelengths: "
    f"{NUM_WAVELENGTHS}")

print(f"min_steps_per_wvl: "
    f"{MIN_STEPS_PER_WVL}")

print(f"MMI width: "
    f"{MMI_WIDTH_UM:.4f} um")

print(f"MMI length: "
    f"{MMI_LENGTH_UM:.6f} um")

print(f"Taper width: "
    f"{TAPER_WIDTH_UM:.4f} um")

print(f"dy: "
    f"{DY:+.4f} um")

# COST

if ESTIMATE_TIDY3D_COST:

    test_tidy3d_api()

    cost = estimate_tidy3d_cost(
        sim,
        task_name=f"{TIDY3D_TASK_NAME}_cost",
        verbose=True,
    )

    print(f"\nEstimated cost: "
        f"{cost:.6g} FlexCredits")


sim_data = None

# LOAD DATA

if LOAD_TIDY3D_DATA:

    sim_data = load_tidy3d_simulation_data(
        TIDY3D_DATA_PATH
    )

if RUN_TIDY3D_CLOUD:

    import tidy3d.web as web

    test_tidy3d_api()
    sim_data = web.run(
        sim,
        task_name=TIDY3D_TASK_NAME,
        path=TIDY3D_DATA_PATH,
        verbose=True,
    )

# FUNCION AUXILIAR

def wrap_phase_deg(
    phase_deg: float | np.ndarray,
):
    """Wrap phase to [-180, 180) degrees."""

    return (
        phase_deg + 180.0
    ) % 360.0 - 180.0

# POSTPROCESADO ESPECTRAL

if sim_data is not None:

    names = ("p1", "p2", "p3", "p4")
    
    frequencies = np.asarray(
        sim_data["flux_p1"].flux.f.values,
        dtype=float,
    )

    wavelengths_um = (td.C_0 / frequencies)
    order = np.argsort(wavelengths_um)

    wavelengths_um = (wavelengths_um[order])

    # FLUX
    
    flux = np.array([
        np.real(
            np.asarray(
                sim_data[
                    f"flux_{name}"
                ].flux.values
            ).squeeze()
        )[order]
        for name in names
    ])

    total_flux = np.sum(flux, axis=0,)

    split_flux = np.divide(
        flux,
        total_flux,
        out=np.zeros_like(flux),
        where=total_flux > 0,
    )

    excess_loss_flux_db = (-10.0 * np.log10(np.maximum(total_flux,1e-20)))

    # TE0

    amps = np.array([
        np.asarray(
            sim_data[
                f"mode_{name}"
            ].amps
            .sel(
                direction="+",
                mode_index=0,
            )
            .values
        ).squeeze()[order]
        for name in names
    ])

    mode_power = (np.abs(amps) ** 2)
    total_mode_power = np.sum(mode_power, axis=0)

    split_mode = np.divide(
        mode_power,
        total_mode_power,
        out=np.zeros_like(mode_power),
        where=total_mode_power > 0,
    )

    excess_loss_mode_db = (-10.0 * np.log10(np.maximum( total_mode_power, 1e-20)))

    # REFLECTION TE0
  
    if INPUT_CONFIGURATION == "top":

        reflection_amp = np.asarray(
            sim_data[
                "mode_reflection_top"
            ].amps
            .sel(
                direction="-",
                mode_index=0,
            )
            .values
        ).squeeze()[order]

        reflection_power = (np.abs(reflection_amp) ** 2)


    elif INPUT_CONFIGURATION == "bottom":

        reflection_amp = np.asarray(
            sim_data[
                "mode_reflection_bottom"
            ].amps
            .sel(
                direction="-",
                mode_index=0,
            )
            .values
        ).squeeze()[order]

        reflection_power = (np.abs(reflection_amp) ** 2)


    else:

        reflection_power = np.full_like(
            wavelengths_um,
            np.nan,
            dtype=float,
        )

    # UNACCOUNTED POWER
   
    unaccounted_power = 1.0 - total_mode_power - reflection_power

    # FASES OUTPUT RESPECTO A P3
    relative_phase_p3_deg = (np.degrees(np.angle(amps* np.conj(amps[2])) )% 360.0)

    # RESULTADOS EN lambda0

    index_lambda0 = np.argmin(np.abs(wavelengths_um- WAVELENGTH_UM))

    print(
        "\n"
        + "=" * 65
    )

    print("FDTD MMI 2x4 SPECTRAL RESULTS")

    print(
        "=" * 65
    )

    print(f"Input configuration: "
        f"{INPUT_CONFIGURATION}")

    if INPUT_CONFIGURATION == "hybrid":

        print(
            f"Delta phi input: "
            f"{DELTA_PHI_IN_DEG:+.2f} deg"
        )


        # ========================================================
    # BANDWIDTH
    # ========================================================

    COUPLING_TOLERANCE = 0.015

    target_split = None


    if INPUT_CONFIGURATION in (
        "top",
        "bottom",
    ):

        target_split = {
            "p1": 0.25,
            "p2": 0.25,
            "p3": 0.25,
            "p4": 0.25,
        }


    if target_split is not None:

        valid = np.ones(
            len(wavelengths_um),
            dtype=bool,
        )

        for i, name in enumerate(names):

            valid &= (
                np.abs(
                    split_mode[i]
                    - target_split[name]
                )
                <= COUPLING_TOLERANCE
            )


        if valid[index_lambda0]:

            i_left = index_lambda0
            i_right = index_lambda0

            while (
                i_left > 0
                and valid[i_left - 1]
            ):

                i_left -= 1


            while (
                i_right < len(valid) - 1
                and valid[i_right + 1]
            ):

                i_right += 1


            lambda_bw_min = (
                wavelengths_um[i_left]
            )

            lambda_bw_max = (
                wavelengths_um[i_right]
            )

            bandwidth_nm = (
                1000.0
                * (
                    lambda_bw_max
                    - lambda_bw_min
                )
            )


        else:

            lambda_bw_min = np.nan
            lambda_bw_max = np.nan
            bandwidth_nm = 0.0


        # ====================================================
        # GRAFICA BANDWIDTH
        # ====================================================

        plt.figure(
            figsize=(8, 5)
        )

        for i, name in enumerate(names):

            plt.plot(
                wavelengths_um,
                split_mode[i],
                linewidth=2,
                label=name.upper(),
            )

            target = (
                target_split[name]
            )

            plt.axhline(
                target,
                linestyle="--",
                linewidth=1,
            )

            plt.axhline(
                target
                + COUPLING_TOLERANCE,
                linestyle=":",
                linewidth=1,
            )

            plt.axhline(
                target
                - COUPLING_TOLERANCE,
                linestyle=":",
                linewidth=1,
            )


        if bandwidth_nm > 0.0:

            plt.axvspan(
                lambda_bw_min,
                lambda_bw_max,
                alpha=0.2,
                label=(
                    f"Bandwidth = "
                    f"{bandwidth_nm:.1f} nm"
                ),
            )


        plt.axvline(
            WAVELENGTH_UM,
            linestyle="--",
            linewidth=1,
            label=r"$\lambda_0$",
        )

        plt.xlabel(
            "Wavelength (µm)"
        )

        plt.ylabel(
            "Splitting ratio"
        )

        plt.title(
            f"MMI 2x4 spectral bandwidth - "
            f"{INPUT_CONFIGURATION}"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()


        print(
            "\n--- BANDWIDTH ---"
        )

        print(
            f"Tolerance: "
            f"±"
            f"{100 * COUPLING_TOLERANCE:.2f} "
            f"percentage points"
        )

        print(
            f"lambda_min: "
            f"{lambda_bw_min:.4f} um"
        )

        print(
            f"lambda_max: "
            f"{lambda_bw_max:.4f} um"
        )

        print(
            f"Bandwidth: "
            f"{bandwidth_nm:.2f} nm"
        )


    else:

        print(
            "\nBandwidth target not defined "
            "for this input configuration."
        )


    print(f"lambda0: "
        f"{wavelengths_um[index_lambda0]:.4f} um")

    print(f"Flux total output: "
        f"{total_flux[index_lambda0]:.6f}")

    print(f"Flux excess loss: "
        f"{excess_loss_flux_db[index_lambda0]:.4f} dB")

    print(f"TE0 total output: "
        f"{total_mode_power[index_lambda0]:.6f}")

    print(f"TE0 excess loss: "
        f"{excess_loss_mode_db[index_lambda0]:.4f} dB")

    if INPUT_CONFIGURATION != "hybrid":

        print(
            f"TE0 reflection: "
            f"{reflection_power[index_lambda0]:.6f}"
        )

        print(
            f"Unaccounted power: "
            f"{unaccounted_power[index_lambda0]:.6f}"
        )

    # SPLITTING EN lambda0
    
    print(
        "\n--- OUTPUT DETAILS AT lambda0 ---"
    )

    for i, name in enumerate(names):

        print(
            f"{name.upper()}: "
            f"T_TE0="
            f"{mode_power[i, index_lambda0]:.6f}, "
            f"split="
            f"{100.0 * split_mode[i, index_lambda0]:.2f} %"
        )

    # FASES EN lambda0
   
    print(
        "\n--- OUTPUT PHASES RELATIVE TO P3 AT lambda0 ---"
    )

    for i, name in enumerate(names):

        print(
            f"{name.upper()} relative to P3 = "
            f"{relative_phase_p3_deg[i, index_lambda0]:.4f} deg"
        )

    # PLOT EXCESS LOSS

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        wavelengths_um,
        excess_loss_mode_db,
        label="TE0 excess loss",
    )

    ax.plot(
        wavelengths_um,
        excess_loss_flux_db,
        "--",
        label="Flux excess loss",
    )

    ax.axvline(
        WAVELENGTH_UM,
        linestyle=":",
    )

    ax.set_xlabel(
        "Wavelength [um]"
    )

    ax.set_ylabel(
        "Excess loss [dB]"
    )

    ax.set_title(
        f"MMI 2x4 - Excess loss - {INPUT_CONFIGURATION}"
    )

    ax.grid(True)
    ax.legend()

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        wavelengths_um,
        total_mode_power,
        label="TE0 total",
    )

    ax.plot(
        wavelengths_um,
        total_flux,
        "--",
        label="Flux total",
    )

    ax.axvline(
        WAVELENGTH_UM,
        linestyle=":",
    )

    ax.set_xlabel(
        "Wavelength [um]"
    )

    ax.set_ylabel(
        "Transmission"
    )

    ax.set_title(
        f"MMI 2x4 - Transmission - {INPUT_CONFIGURATION}"
    )

    ax.grid(True)
    ax.legend()

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, name in enumerate(names):

        ax.plot(
            wavelengths_um,
            100.0 * split_mode[i],
            label=name.upper(),
        )

    if INPUT_CONFIGURATION != "hybrid":

        ax.axhline(
            25.0,
            linestyle="--",
            label="Ideal 25 %",
        )

    ax.axvline(
        WAVELENGTH_UM,
        linestyle=":",
    )

    ax.set_xlabel(
        "Wavelength [um]"
    )

    ax.set_ylabel(
        "TE0 splitting [%]"
    )

    ax.set_title(
        f"MMI 2x4 - Splitting - {INPUT_CONFIGURATION}"
    )

    ax.grid(True)
    ax.legend()

    # PLOT REFLECTION / UNACCOUNTED

    if INPUT_CONFIGURATION != "hybrid":

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.plot(
            wavelengths_um,
            100.0 * reflection_power,
            label="TE0 reflection",
        )

        ax.plot(
            wavelengths_um,
            100.0 * unaccounted_power,
            label="Unaccounted power",
        )

        ax.axvline(
            WAVELENGTH_UM,
            linestyle=":",
        )

        ax.set_xlabel(
            "Wavelength [um]"
        )

        ax.set_ylabel(
            "Power [%]"
        )

        ax.set_title(
            f"MMI 2x4 - Power balance - {INPUT_CONFIGURATION}"
        )

        ax.grid(True)
        ax.legend()

    # FIELD EN lambda0

    fig, ax = plot_fdtd_field_xy(
        sim_data,
        monitor_name="field_core",
        field_component="Ey",
        field_part="mag",
        polygons=polygons,
    )

    ax.set_title(
        f"FDTD |Ey| - {TIDY3D_TASK_NAME}"
    )


# FASES DEL HIBRIDO - YU ET AL.
# WAVELENGTH SWEEP

top_path = Path(TOP_DATA_PATH)
bottom_path = Path( BOTTOM_DATA_PATH)


if top_path.exists() and bottom_path.exists():

    sim_data_top = load_tidy3d_simulation_data(TOP_DATA_PATH)

    sim_data_bottom = load_tidy3d_simulation_data(BOTTOM_DATA_PATH)

    names = ("p1", "p2", "p3", "p4")

    # EJE ESPECTRAL

    frequencies_phase = np.asarray(
        sim_data_top[
            "mode_p1"
        ].amps.f.values,
        dtype=float,
    )

    wavelengths_phase_um = (td.C_0/ frequencies_phase)

    order_phase = np.argsort( wavelengths_phase_um)

    wavelengths_phase_um = ( wavelengths_phase_um[order_phase])


    # AMPLITUDES COMPLEJAS - TOP

    amps_top = np.array([
        np.asarray(
            sim_data_top[
                f"mode_{name}"
            ].amps
            .sel(
                direction="+",
                mode_index=0,
            )
            .values
        ).squeeze()[order_phase]
        for name in names
    ])


    # AMPLITUDES COMPLEJAS - BOTTOM

    amps_bottom = np.array([
        np.asarray(
            sim_data_bottom[
                f"mode_{name}"
            ].amps
            .sel(
                direction="+",
                mode_index=0,
            )
            .values
        ).squeeze()[order_phase]
        for name in names
    ])


    # ========================================================
    # FASE CARACTERISTICA DE CADA OUTPUT
    #
    # Se mantiene la convencion utilizada en la
    # caracterizacion actual del dispositivo.
    #
    # Orden:
    # [P1, P2, P3, P4]
    # ========================================================

    transfer_phase_deg = np.degrees(
        np.angle(
            amps_top
            * np.conj(
                amps_bottom
            )
        )
    )

    phi_p1_deg = (transfer_phase_deg[0])
    phi_p2_deg = (transfer_phase_deg[1])
    phi_p3_deg = (transfer_phase_deg[2])
    phi_p4_deg = (transfer_phase_deg[3])

    # FASES RESPECTO A P3

    phase_p1_p3_deg = (phi_p1_deg- phi_p3_deg) % 360.0

    phase_p2_p3_deg = (phi_p2_deg- phi_p3_deg) % 360.0

    phase_p3_p3_deg = np.zeros_like(phase_p1_p3_deg)

    phase_p4_p3_deg = (phi_p4_deg- phi_p3_deg ) % 360.0

    # RELACIONES DE FASE DE YU
  
    phase_13_deg = wrap_phase_deg(phi_p1_deg - phi_p3_deg)
    phase_23_deg = wrap_phase_deg( phi_p2_deg - phi_p3_deg)
    phase_34_deg = wrap_phase_deg( phi_p3_deg - phi_p4_deg)

    # PHASE DEVIATIONS

    delta_phi_13_deg = wrap_phase_deg(phase_13_deg- 90.0)
    delta_phi_23_deg = wrap_phase_deg( phase_23_deg - 180.0)
    delta_phi_34_deg = wrap_phase_deg(phase_34_deg- 90.0)

    # RESULTADOS EN lambda0

    index_lambda0_phase = np.argmin(np.abs(wavelengths_phase_um - WAVELENGTH_UM))

        # ========================================================
    # PHASE BANDWIDTH
    # ========================================================

    PHASE_TOLERANCE_DEG = 5.0

    valid_phase = (
        (np.abs(delta_phi_13_deg) <= PHASE_TOLERANCE_DEG)
        & (np.abs(delta_phi_23_deg) <= PHASE_TOLERANCE_DEG)
        & (np.abs(delta_phi_34_deg) <= PHASE_TOLERANCE_DEG)
    )


    if valid_phase[index_lambda0_phase]:

        i_left_phase = index_lambda0_phase
        i_right_phase = index_lambda0_phase


        while (
            i_left_phase > 0
            and valid_phase[i_left_phase - 1]
        ):

            i_left_phase -= 1


        while (
            i_right_phase
            < len(valid_phase) - 1
            and valid_phase[i_right_phase + 1]
        ):

            i_right_phase += 1


        lambda_phase_bw_min = (
            wavelengths_phase_um[
                i_left_phase
            ]
        )

        lambda_phase_bw_max = (
            wavelengths_phase_um[
                i_right_phase
            ]
        )

        phase_bandwidth_nm = (
            1000.0
            * (
                lambda_phase_bw_max
                - lambda_phase_bw_min
            )
        )


    else:

        lambda_phase_bw_min = np.nan
        lambda_phase_bw_max = np.nan
        phase_bandwidth_nm = 0.0

    print(
        "\n"
        + "=" * 65
    )

    print("FDTD MMI 2x4 - YU SPECTRAL PHASE CHARACTERIZATION")
    print(
        "=" * 65
    )

    print(f"lambda0 = "
        f"{wavelengths_phase_um[index_lambda0_phase]:.4f} um")

    print("\n--- PHASES RELATIVE TO P3 AT lambda0 ---")
    print(f"P1 relative to P3 = "
        f"{phase_p1_p3_deg[index_lambda0_phase]:.4f} deg")
    print(f"P2 relative to P3 = "
        f"{phase_p2_p3_deg[index_lambda0_phase]:.4f} deg")
    print(f"P3 relative to P3 = "
        f"{phase_p3_p3_deg[index_lambda0_phase]:.4f} deg")

    print( f"P4 relative to P3 = "
        f"{phase_p4_p3_deg[index_lambda0_phase]:.4f} deg")


    print(
        "\n--- PHASE DEVIATIONS AT lambda0 ---"
    )

    print(
        f"Delta phi13 = "
        f"{delta_phi_13_deg[index_lambda0_phase]:+.4f} deg"
    )

    print(
        f"Delta phi23 = "
        f"{delta_phi_23_deg[index_lambda0_phase]:+.4f} deg"
    )

    print(
        f"Delta phi34 = "
        f"{delta_phi_34_deg[index_lambda0_phase]:+.4f} deg"
    )

    # MAX PHASE ERROR - C BAND
    
    c_band = (
        (wavelengths_phase_um >= 1.530)
        & (wavelengths_phase_um <= 1.565)
    )

    print("\n--- MAX ABS PHASE DEVIATION IN C-BAND ---")
    print(f"Delta phi13: "
        f"{np.max(np.abs(delta_phi_13_deg[c_band])):.4f} deg")
    print(f"Delta phi23: "
        f"{np.max(np.abs(delta_phi_23_deg[c_band])):.4f} deg" )
    print(f"Delta phi34: "
        f"{np.max(np.abs(delta_phi_34_deg[c_band])):.4f} deg")

    # PLOT PHASE DEVIATIONS

        # ========================================================
    # PLOT PHASE DEVIATIONS
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        wavelengths_phase_um,
        delta_phi_13_deg,
        linewidth=2,
        label=r"$\Delta\phi_{13}$",
    )

    ax.plot(
        wavelengths_phase_um,
        delta_phi_23_deg,
        linewidth=2,
        label=r"$\Delta\phi_{23}$",
    )

    ax.plot(
        wavelengths_phase_um,
        delta_phi_34_deg,
        linewidth=2,
        label=r"$\Delta\phi_{34}$",
    )


    # --------------------------------------------------------
    # LIMITES DE TOLERANCIA
    # --------------------------------------------------------

    ax.axhline(
        +PHASE_TOLERANCE_DEG,
        linestyle="--",
        linewidth=1,
        label=r"$\pm 5^\circ$ tolerance",
    )

    ax.axhline(
        -PHASE_TOLERANCE_DEG,
        linestyle="--",
        linewidth=1,
    )

    ax.axhline(
        0.0,
        linestyle=":",
        linewidth=1,
    )


    # --------------------------------------------------------
    # PHASE BANDWIDTH
    # --------------------------------------------------------

    if phase_bandwidth_nm > 0.0:

        ax.axvspan(
            lambda_phase_bw_min,
            lambda_phase_bw_max,
            alpha=0.2,
            label=(
                f"Phase bandwidth = "
                f"{phase_bandwidth_nm:.1f} nm"
            ),
        )


    # --------------------------------------------------------
    # lambda0
    # --------------------------------------------------------

    ax.axvline(
        WAVELENGTH_UM,
        linestyle="--",
        linewidth=1,
        label=r"$\lambda_0$",
    )


    ax.set_xlabel(
        "Wavelength (µm)"
    )

    ax.set_ylabel(
        "Phase deviation (deg)"
    )

    ax.set_title(
        "MMI 2x4 - Yu phase deviations"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    plt.tight_layout()


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        wavelengths_phase_um,
        phase_p1_p3_deg,
        label="P1 - P3",
    )

    ax.plot(
        wavelengths_phase_um,
        phase_p2_p3_deg,
        label="P2 - P3",
    )

    ax.plot(
        wavelengths_phase_um,
        phase_p4_p3_deg,
        label="P4 - P3",
    )

    ax.axvline(
        WAVELENGTH_UM,
        linestyle=":",
    )

    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Relative phase [deg]")
    ax.set_title("MMI 2x4 - Relative transfer phases")

    ax.grid(True)
    ax.legend()


else:

    print(
        "\n--- YU SPECTRAL PHASE CHARACTERIZATION ---"
    )

    if not top_path.exists():

        print(
            f"Missing top sweep: "
            f"{TOP_DATA_PATH}"
        )

    if not bottom_path.exists():

        print(
            f"Missing bottom sweep: "
            f"{BOTTOM_DATA_PATH}"
        )


# GEOMETRIA


I1_UM = ( INPUT_Y_POSITIONS_UM[0])
I2_UM = (INPUT_Y_POSITIONS_UM[1])
P1_UM = (OUTPUT_Y_POSITIONS_UM[0])
P2_UM = (OUTPUT_Y_POSITIONS_UM[1])
P3_UM = ( OUTPUT_Y_POSITIONS_UM[2])
P4_UM = (OUTPUT_Y_POSITIONS_UM[3])
OUTPUT_PITCH_UM = abs(P1_UM - P2_UM)
OUTPUT_GAP_UM = (OUTPUT_PITCH_UM - TAPER_WIDTH_UM)
EDGE_MARGIN_UM = (0.5 * MMI_WIDTH_UM - abs(P1_UM) - 0.5 * TAPER_WIDTH_UM)


print(
    "\n--- 2x4 PORT GEOMETRY ---"
)

print(
    "Input positions [um]: "
    f"I1/top={I1_UM:+.4f}, "
    f"I2/bottom={I2_UM:+.4f}"
)

print(
    "Output positions [um]: "
    f"P1={P1_UM:+.4f}, "
    f"P2={P2_UM:+.4f}, "
    f"P3={P3_UM:+.4f}, "
    f"P4={P4_UM:+.4f}"
)

print(
    f"Output pitch: "
    f"{OUTPUT_PITCH_UM:.4f} um"
)

print(
    f"Taper width: "
    f"{TAPER_WIDTH_UM:.4f} um"
)

print(
    f"Gap between adjacent tapers: "
    f"{OUTPUT_GAP_UM:.4f} um"
)

print(
    f"Outer taper edge margin: "
    f"{EDGE_MARGIN_UM:.4f} um"
)


plt.show()