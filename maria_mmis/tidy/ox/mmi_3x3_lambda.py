from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td

from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.waveguides import strip_waveguide

from upvfab_design_tools.tidy3d_plugin import (
    build_mmi_3x3_fdtd_simulation,
    estimate_tidy3d_cost,
    load_tidy3d_simulation_data,
    mmi_3x3_vertices,
    plot_fdtd_field_xy,
    test_tidy3d_api,
)


# ============================================================
# CONTROL
# ============================================================

RUN_TIDY3D_CLOUD = False
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = True

INPUT_CONFIGURATION = "hybrid"  # "top", "bottom", "hybrid"
DELTA_PHI_IN_DEG = 0.0

TIDY3D_TASK_NAME = (
    f"mmi_3x3_Lminus2.2_dy0.09_wt1.9_"
    f"{INPUT_CONFIGURATION}_sweep"
)

TIDY3D_DATA_PATH = (
    f"data/{TIDY3D_TASK_NAME}.hdf5"
)


# ============================================================
# WAVELENGTH SWEEP
# ============================================================

WAVELENGTH_UM = 1.55

WAVELENGTH_MIN_UM = 1.50
WAVELENGTH_MAX_UM = 1.60

NUM_WAVELENGTHS = 101

WAVELENGTHS_UM = np.linspace(
    WAVELENGTH_MIN_UM,
    WAVELENGTH_MAX_UM,
    NUM_WAVELENGTHS,
)

FREQS = (
    td.C_0
    / WAVELENGTHS_UM
)

SOURCE_BANDWIDTH_UM = (
    WAVELENGTH_MAX_UM
    - WAVELENGTH_MIN_UM
)


# ============================================================
# PARAMETROS DISPOSITIVO
# ============================================================

CORE_HEIGHT_UM = 0.30

MMI_WIDTH_UM = 10.0

MMI_LENGTH_UM = (
    149.968844
    - 2.2
)

DY = 0.09

PORT_POSITION_UM = (
    MMI_WIDTH_UM / 3.0
    + DY
)

ACCESS_WIDTH_UM = 1.0

TAPER_WIDTH_UM = 1.9
TAPER_LENGTH_UM = 20.0

STRAIGHT_IO_LENGTH_UM = 2.0

IO_LENGTH_UM = (
    TAPER_LENGTH_UM
    + STRAIGHT_IO_LENGTH_UM
)

SOURCE_OFFSET_UM = (
    0.5
    * STRAIGHT_IO_LENGTH_UM
)

MONITOR_OFFSET_UM = (
    0.5
    * STRAIGHT_IO_LENGTH_UM
)

MODE_PLANE_Y_SPAN_UM = (
    2.5
    * ACCESS_WIDTH_UM
)

DOMAIN_X_SPAN_UM = (
    MMI_WIDTH_UM
    + 4.0
)


# ============================================================
# GEOMETRIA PUERTOS
# ============================================================

PORT_POSITIONS_UM = np.array(
    [
        -PORT_POSITION_UM,
        0.0,
        +PORT_POSITION_UM,
    ]
)

PORT_PITCH_UM = PORT_POSITION_UM

GAP_UM = (
    PORT_PITCH_UM
    - TAPER_WIDTH_UM
)

EDGE_CLEARANCE_UM = (
    MMI_WIDTH_UM / 2.0
    - PORT_POSITION_UM
    - TAPER_WIDTH_UM / 2.0
)

print("\n--- 3x3 PORT GEOMETRY ---")

print(
    "Port positions [um]: "
    f"bottom={PORT_POSITIONS_UM[0]:+.4f}, "
    f"center={PORT_POSITIONS_UM[1]:+.4f}, "
    f"top={PORT_POSITIONS_UM[2]:+.4f}"
)

print(
    f"Port pitch: "
    f"{PORT_PITCH_UM:.4f} um"
)

print(
    f"Taper width: "
    f"{TAPER_WIDTH_UM:.4f} um"
)

print(
    f"Gap between adjacent tapers: "
    f"{GAP_UM:.4f} um"
)

print(
    f"Edge clearance: "
    f"{EDGE_CLEARANCE_UM:.4f} um"
)


# ============================================================
# STACK
# ============================================================

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


# ============================================================
# GEOMETRIA
# ============================================================

polygons = mmi_3x3_vertices(
    length_um=MMI_LENGTH_UM,

    io_length_um=IO_LENGTH_UM,

    straight_io_length_um=(
        STRAIGHT_IO_LENGTH_UM
    ),

    port_position_um=(
        PORT_POSITION_UM
    ),

    access_width_um=(
        ACCESS_WIDTH_UM
    ),

    taper_width_um=(
        TAPER_WIDTH_UM
    ),

    mmi_width_um=(
        MMI_WIDTH_UM
    ),
)


# ============================================================
# SIMULACION
# ============================================================

sim = build_mmi_3x3_fdtd_simulation(
    polygons=polygons,

    cross_section=cross_section,

    wavelength_um=WAVELENGTH_UM,

    port_position_um=(
        PORT_POSITION_UM
    ),

    access_width_um=(
        ACCESS_WIDTH_UM
    ),

    input_configuration=(
        INPUT_CONFIGURATION
    ),

    delta_phi_rad=np.deg2rad(
        DELTA_PHI_IN_DEG
    ),

    bandwidth_um=(
        SOURCE_BANDWIDTH_UM
    ),

    pad_x_um=2.0,
    pad_y_um=2.0,
    pad_z_um=0.0,

    source_offset_um=(
        SOURCE_OFFSET_UM
    ),

    monitor_offset_um=(
        MONITOR_OFFSET_UM
    ),

    mode_plane_y_span_um=(
        MODE_PLANE_Y_SPAN_UM
    ),

    min_steps_per_wvl=16,

    run_time_s=5e-12,
)


# ============================================================
# CONVERTIR MONITORES A BARRIDO ESPECTRAL
# ============================================================

SPECTRAL_MONITORS = (
    "flux_bottom",
    "flux_center",
    "flux_top",

    "mode_bottom",
    "mode_center",
    "mode_top",
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

        # field_core permanece solamente en lambda0
        updated_monitors.append(
            monitor
        )

sim = sim.updated_copy(
    monitors=updated_monitors,
)


# ============================================================
# VALIDACION
# ============================================================

sim.validate_pre_upload()

print(
    "\nSimulation validated successfully."
)

print(
    f"\nWavelength sweep: "
    f"{WAVELENGTH_MIN_UM:.3f} - "
    f"{WAVELENGTH_MAX_UM:.3f} um"
)

print(
    f"Number of wavelengths: "
    f"{NUM_WAVELENGTHS}"
)

print(
    f"MMI length: "
    f"{MMI_LENGTH_UM:.4f} um"
)

print(
    f"MMI width: "
    f"{MMI_WIDTH_UM:.4f} um"
)

print(
    f"dy: "
    f"{DY:+.4f} um"
)

print(
    f"Input configuration: "
    f"{INPUT_CONFIGURATION}"
)

print(
    f"Input delta phi: "
    f"{DELTA_PHI_IN_DEG:+.2f} deg"
)


# ============================================================
# REPRESENTAR SIMULACION
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 4)
)

sim.plot(
    z=0.5 * CORE_HEIGHT_UM,
    ax=ax,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.set_title(
    "Tidy3D geometry, sources and monitors"
)


# ============================================================
# REPRESENTAR PERMITIVIDAD
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 4)
)

sim.plot_eps(
    z=0.5 * CORE_HEIGHT_UM,
    ax=ax,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.set_title(
    "Tidy3D permittivity"
)


# ============================================================
# COSTE
# ============================================================

if ESTIMATE_TIDY3D_COST:

    test_tidy3d_api()

    cost = estimate_tidy3d_cost(
        sim,
        task_name=(
            f"{TIDY3D_TASK_NAME}_cost"
        ),
        verbose=True,
    )

    print(
        f"\nEstimated cost: "
        f"{cost:.6g} FlexCredits"
    )


# ============================================================
# SIMULATION DATA
# ============================================================

sim_data = None


# ============================================================
# LOAD
# ============================================================

if LOAD_TIDY3D_DATA:

    print(
        f"\nLoading Tidy3D data from:"
        f"\n{TIDY3D_DATA_PATH}"
    )

    sim_data = (
        load_tidy3d_simulation_data(
            TIDY3D_DATA_PATH
        )
    )


# ============================================================
# RUN CLOUD
# ============================================================

if RUN_TIDY3D_CLOUD:

    import tidy3d.web as web

    test_tidy3d_api()

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RUNNING TIDY3D CLOUD SIMULATION"
    )

    print(
        "=" * 60
    )

    sim_data = web.run(
        sim,

        task_name=(
            TIDY3D_TASK_NAME
        ),

        path=(
            TIDY3D_DATA_PATH
        ),

        verbose=True,
    )


# ============================================================
# POSTPROCESADO ESPECTRAL
# ============================================================

if sim_data is not None:

    names = (
        "bottom",
        "center",
        "top",
    )


    # ========================================================
    # FLUX
    # ========================================================

    flux_data = {
        name: sim_data[
            f"flux_{name}"
        ].flux

        for name in names
    }

    frequencies = np.asarray(
        flux_data["center"].f.values,
        dtype=float,
    )

    wavelengths_um = (
        td.C_0
        / frequencies
    )

    T = {
        name: np.real(
            np.asarray(
                flux_data[name].values
            ).squeeze()
        )

        for name in names
    }


    # ========================================================
    # ORDENAR POR LONGITUD DE ONDA
    # ========================================================

    order = np.argsort(
        wavelengths_um
    )

    wavelengths_um = (
        wavelengths_um[order]
    )

    for name in names:

        T[name] = (
            T[name][order]
        )


    # ========================================================
    # TRANSMISION TOTAL
    # ========================================================

    T_total = (
        T["bottom"]
        + T["center"]
        + T["top"]
    )


    # ========================================================
    # SPLITTING RATIO
    # ========================================================

    K = {}

    for name in names:

        K[name] = np.divide(
            T[name],
            T_total,

            out=np.zeros_like(
                T[name]
            ),

            where=(
                T_total > 0.0
            ),
        )


    # ========================================================
    # EXCESS LOSS
    # ========================================================

    excess_loss_db = (
        -10.0
        * np.log10(
            np.maximum(
                T_total,
                1e-20,
            )
        )
    )


    # ========================================================
    # AMPLITUDES MODALES TE0
    # ========================================================

    amplitudes = {}

    for name in names:

        amplitudes[name] = np.asarray(
            sim_data[
                f"mode_{name}"
            ].amps
            .sel(
                direction="+",
                mode_index=0,
            )
            .values
        ).squeeze()

        amplitudes[name] = (
            amplitudes[name][order]
        )


    # ========================================================
    # FASE RELATIVA AL PUERTO CENTRAL
    # ========================================================

    relative_phase_deg = {}

    for name in names:

        if name == "center":

            relative_phase_deg[name] = (
                np.zeros_like(
                    wavelengths_um
                )
            )

        else:

            phase_rad = np.angle(
                amplitudes[name]
                * np.conj(
                    amplitudes["center"]
                )
            )

            relative_phase_deg[name] = (
                np.degrees(
                    np.unwrap(
                        phase_rad
                    )
                )
            )


    # ========================================================
    # RESULTADOS EN lambda0
    # ========================================================

    index_lambda0 = np.argmin(
        np.abs(
            wavelengths_um
            - WAVELENGTH_UM
        )
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FDTD MMI 3x3 SPECTRAL RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Input: "
        f"{INPUT_CONFIGURATION}"
    )

    print(
        f"Delta phi input: "
        f"{DELTA_PHI_IN_DEG:+.2f} deg"
    )

    print(
        f"lambda = "
        f"{wavelengths_um[index_lambda0]:.4f} um"
    )

    print(
        f"T total = "
        f"{T_total[index_lambda0]:.6f}"
    )

    print(
        f"Excess loss = "
        f"{excess_loss_db[index_lambda0]:.4f} dB"
    )

    for name in names:

        print(
            f"{name:>6s}: "
            f"T={T[name][index_lambda0]:.6f}, "
            f"split="
            f"{100*K[name][index_lambda0]:.2f} %, "
            f"phase rel. center="
            f"{relative_phase_deg[name][index_lambda0]:+.2f} deg"
        )


    # ========================================================
    # 1. SPLITTING RATIO VS WAVELENGTH
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    for name in names:

        plt.plot(
            wavelengths_um,
            K[name],
            linewidth=2,
            label=f"{name.capitalize()} output",
        )

    if (
        INPUT_CONFIGURATION == "hybrid"
        and np.isclose(
            DELTA_PHI_IN_DEG,
            0.0,
        )
    ):

        plt.axhline(
            1.0 / 6.0,
            linestyle="--",
            linewidth=1,
            label="Ideal outer = 1/6",
        )

        plt.axhline(
            2.0 / 3.0,
            linestyle="--",
            linewidth=1,
            label="Ideal center = 2/3",
        )

    elif INPUT_CONFIGURATION in (
        "top",
        "bottom",
    ):

        plt.axhline(
            1.0 / 3.0,
            linestyle="--",
            linewidth=1,
            label="Ideal = 1/3",
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
        "MMI 3x3 splitting ratio vs wavelength"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # 2. TOTAL TRANSMITTED POWER
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        wavelengths_um,
        T_total,
        linewidth=2,
    )

    plt.axvline(
        WAVELENGTH_UM,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Wavelength (µm)"
    )

    plt.ylabel(
        "Total transmitted power"
    )

    plt.title(
        "Total transmitted power vs wavelength"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()


    # ========================================================
    # 3. EXCESS LOSS
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        wavelengths_um,
        excess_loss_db,
        linewidth=2,
    )

    plt.axvline(
        WAVELENGTH_UM,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Wavelength (µm)"
    )

    plt.ylabel(
        "Excess loss (dB)"
    )

    plt.title(
        "Excess loss vs wavelength"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()


    # ========================================================
    # 4. FASES RELATIVAS
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        wavelengths_um,
        relative_phase_deg["bottom"],
        linewidth=2,
        label="Bottom - center",
    )

    plt.plot(
        wavelengths_um,
        relative_phase_deg["center"],
        linewidth=2,
        label="Center - center",
    )

    plt.plot(
        wavelengths_um,
        relative_phase_deg["top"],
        linewidth=2,
        label="Top - center",
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
        "Relative phase (deg)"
    )

    plt.title(
        "Relative output phases vs wavelength"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # 5. BANDWIDTH
    # ========================================================

    COUPLING_TOLERANCE = 0.015

    target_split = None

    if (
        INPUT_CONFIGURATION == "hybrid"
        and np.isclose(
            DELTA_PHI_IN_DEG,
            0.0,
        )
    ):

        target_split = {
            "bottom": 1.0 / 6.0,
            "center": 2.0 / 3.0,
            "top": 1.0 / 6.0,
        }

    elif INPUT_CONFIGURATION in (
        "top",
        "bottom",
    ):

        target_split = {
            "bottom": 1.0 / 3.0,
            "center": 1.0 / 3.0,
            "top": 1.0 / 3.0,
        }


    if target_split is not None:

        valid = np.ones(
            len(wavelengths_um),
            dtype=bool,
        )

        for name in names:

            valid &= (
                np.abs(
                    K[name]
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
                i_right
                < len(valid) - 1
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

        for name in names:

            plt.plot(
                wavelengths_um,
                K[name],
                linewidth=2,
                label=(
                    f"{name.capitalize()} output"
                ),
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
            "MMI 3x3 spectral bandwidth"
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
            f"{100*COUPLING_TOLERANCE:.2f} "
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
            "for this hybrid input phase."
        )


    # ========================================================
    # FIELD A lambda0
    # ========================================================

    fig_field, ax_field = (
        plot_fdtd_field_xy(
            sim_data,

            monitor_name=(
                "field_core"
            ),

            field_component=(
                "Ey"
            ),

            field_part=(
                "mag"
            ),

            polygons=polygons,
        )
    )

    ax_field.set_title(
        f"FDTD |Ey| - "
        f"{TIDY3D_TASK_NAME}"
    )


plt.show()