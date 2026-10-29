from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td


from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)

from upvfab_design_tools.core.waveguides import (
    strip_waveguide,
)

from upvfab_design_tools.tidy3d_plugin import (
    build_mmi_1x4_fdtd_simulation,
    estimate_tidy3d_cost,
    extract_fdtd_layers,
    load_tidy3d_simulation_data,
    mmi_1x4_vertices,
    plot_fdtd_field_xy,
    plot_fdtd_polygons,
    test_tidy3d_api,
)


RUN_TIDY3D_CLOUD = True
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False


TIDY3D_TASK_NAME = (
    "mmi_1x4_Lememinus1_wt1.5_dy0.05_elipso_sweep"
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

BANDWIDTH_UM = (
    WAVELENGTH_MAX_UM
    - WAVELENGTH_MIN_UM
)


# ============================================================
# PARAMETROS DISPOSITIVO
# ============================================================

CORE_HEIGHT_UM = 0.3
MMI_WIDTH_UM = 12.0


# Resultado optimizado con EME + Tidy mode solver

DL = -1.0

MMI_LENGTH_UM = (
    39.7234
    + DL
)


ACCESS_WIDTH_UM = 1.0


# Anchura del taper

TAPER_WIDTH_UM = 1.5

TAPER_LENGTH_UM = 15.0


# Tramo recto exterior para source / monitors

STRAIGHT_IO_LENGTH_UM = 2.0


# Longitud total desde el MMI hasta el extremo

IO_LENGTH_UM = (
    TAPER_LENGTH_UM
    + STRAIGHT_IO_LENGTH_UM
)


# Source y monitors en el centro
# de los tramos rectos

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


# ============================================================
# POSICIONES DE LAS 4 SALIDAS
# ============================================================

# IMPORTANTE:
# Deben estar ordenadas de abajo hacia arriba:
# y1 < y2 < y3 < y4

DY = 0.05


OUTPUT_Y_POSITIONS_UM = (
    -4.65 + DY,
    -1.54 + DY,
    +1.54 + DY,
    +4.65 + DY,
)


# ============================================================
# DOMINIO
# ============================================================

DOMAIN_X_SPAN_UM = (
    MMI_WIDTH_UM
    + 4.0
)

BOTTOM_MARGIN_UM = 1.5
TOP_MARGIN_UM = 1.5


# ============================================================
# STACK VERTICAL
# ============================================================

cross_section = strip_waveguide(
    width=ACCESS_WIDTH_UM,

    height=CORE_HEIGHT_UM,

    core_material=(
        SILICON_NITRIDE
    ),

    background_material=(
        THERMAL_SILICON_DIOXIDE
    ),

    x_center=0.0,

    z_min=0.0,

    x_span=DOMAIN_X_SPAN_UM,

    bottom_margin=(
        BOTTOM_MARGIN_UM
    ),

    top_margin=(
        TOP_MARGIN_UM
    ),

    name="sin_strip_fdtd_stack",
)


# ============================================================
# COMPROBAR STACK
# ============================================================

core, surrounding = extract_fdtd_layers(
    cross_section
)

print(
    "FDTD stack core:",
    core.name,
    f"z=[{core.z_min:.3f}, "
    f"{core.z_max:.3f}] um",
    core.material_name,
)

print(
    "FDTD surrounding layers:",
    [
        layer.name
        for layer in surrounding
    ],
)


# ============================================================
# GEOMETRIA MMI 1x4
# ============================================================

polygons = mmi_1x4_vertices(
    length_um=(
        MMI_LENGTH_UM
    ),

    io_length_um=(
        IO_LENGTH_UM
    ),

    straight_io_length_um=(
        STRAIGHT_IO_LENGTH_UM
    ),

    output_y_positions_um=(
        OUTPUT_Y_POSITIONS_UM
    ),

    access_width_um=(
        ACCESS_WIDTH_UM
    ),

    taper_width_um=(
        TAPER_WIDTH_UM
    ),

    mmi_width_input_um=(
        MMI_WIDTH_UM
    ),

    mmi_width_center_um=(
        MMI_WIDTH_UM
    ),
)


# ============================================================
# SIMULACION
# ============================================================

sim = build_mmi_1x4_fdtd_simulation(
    polygons=polygons,

    cross_section=cross_section,

    wavelength_um=(
        WAVELENGTH_UM
    ),

    output_y_positions_um=(
        OUTPUT_Y_POSITIONS_UM
    ),

    access_width_um=(
        ACCESS_WIDTH_UM
    ),

    straight_io_length_um=(
        STRAIGHT_IO_LENGTH_UM
    ),

    bandwidth_um=(
        BANDWIDTH_UM
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

    min_steps_per_wvl=12,

    run_time_s=5e-12,
)


# ============================================================
# MATERIAL CHECK
# ============================================================

print(
    "\n--- MATERIAL CHECK ---"
)

print(
    "Background medium:"
)

print(
    sim.medium
)

print(
    "\nStructure media:"
)

for structure in sim.structures:

    print(
        structure.name,
        structure.medium,
    )


# ============================================================
# CONVERTIR MONITORES A BARRIDO ESPECTRAL
# ============================================================

updated_monitors = []


for monitor in sim.monitors:

    if (
        monitor.name
        != "field_core"
    ):

        updated_monitors.append(
            monitor.updated_copy(
                freqs=FREQS
            )
        )

    else:

        # El field monitor se mantiene
        # solamente en lambda0

        updated_monitors.append(
            monitor
        )


sim = sim.updated_copy(
    monitors=updated_monitors
)


# ============================================================
# VALIDACION LOCAL
# ============================================================

sim.validate_pre_upload()


if ESTIMATE_TIDY3D_COST:

    test_tidy3d_api()

    estimated_cost = estimate_tidy3d_cost(
        sim,

        task_name=(
            f"{TIDY3D_TASK_NAME}_cost"
        ),

        verbose=True,
    )

    print(
        f"\nEstimated Tidy3D cost: "
        f"{estimated_cost:.6g} FlexCredits"
    )


print(
    "\nSimulation validated successfully."
)


# ============================================================
# INFO DE LA SIMULACION
# ============================================================

print(
    f"\nSimulation center: "
    f"{sim.center}"
)

print(
    f"Simulation size: "
    f"{sim.size}"
)

print(
    "Structures:",
    [
        structure.name
        for structure in sim.structures
    ],
)

print(
    "Sources:",
    [
        source.name
        for source in sim.sources
    ],
)

print(
    "Monitors:",
    [
        monitor.name
        for monitor in sim.monitors
    ],
)


if len(sim.sources) > 0:

    print(
        f"Source center: "
        f"{sim.sources[0].center}"
    )

    print(
        f"Source size: "
        f"{sim.sources[0].size}"
    )


# ============================================================
# REPRESENTAR POLIGONOS
# ============================================================

fig_vertices, ax_vertices = (
    plot_fdtd_polygons(
        polygons
    )
)

ax_vertices.set_title(
    "MMI 1x4 - FDTD geometry"
)


# ============================================================
# REPRESENTAR SIMULACION TIDY3D
# ============================================================

fig_sim, ax_sim = plt.subplots(
    figsize=(10, 4)
)

sim.plot(
    z=0.15,
    ax=ax_sim,
)

ax_sim.set_aspect(
    "equal",
    adjustable="box",
)

ax_sim.set_title(
    "Tidy3D geometry, source and monitors"
)


# ============================================================
# REPRESENTAR PERMITIVIDAD
# ============================================================

fig_eps, ax_eps = plt.subplots(
    figsize=(10, 4)
)

sim.plot_eps(
    z=0.15,
    ax=ax_eps,
)

ax_eps.set_aspect(
    "equal",
    adjustable="box",
)

ax_eps.set_title(
    "Tidy3D permittivity at z = 0.15 um"
)


# ============================================================
# SIMULATION DATA
# ============================================================

sim_data = None


# ============================================================
# CARGAR RESULTADOS
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
# EJECUTAR CLOUD
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

    print(
        f"Task name: "
        f"{TIDY3D_TASK_NAME}"
    )

    print(
        f"Data path: "
        f"{TIDY3D_DATA_PATH}"
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

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FDTD SPECTRAL RESULTS"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # FLUX
    # ========================================================

    flux_1_data = (
        sim_data["flux_1"].flux
    )

    flux_2_data = (
        sim_data["flux_2"].flux
    )

    flux_3_data = (
        sim_data["flux_3"].flux
    )

    flux_4_data = (
        sim_data["flux_4"].flux
    )


    frequencies = np.asarray(
        flux_1_data.f.values,
        dtype=float,
    )


    wavelengths_um = (
        td.C_0
        / frequencies
    )


    T_1 = np.real(
        np.asarray(
            flux_1_data.values
        ).squeeze()
    )

    T_2 = np.real(
        np.asarray(
            flux_2_data.values
        ).squeeze()
    )

    T_3 = np.real(
        np.asarray(
            flux_3_data.values
        ).squeeze()
    )

    T_4 = np.real(
        np.asarray(
            flux_4_data.values
        ).squeeze()
    )


    # ========================================================
    # ORDENAR POR LONGITUD DE ONDA
    # ========================================================

    order = np.argsort(
        wavelengths_um
    )


    wavelengths_um = (
        wavelengths_um[order]
    )

    T_1 = T_1[order]
    T_2 = T_2[order]
    T_3 = T_3[order]
    T_4 = T_4[order]


    # ========================================================
    # POTENCIA TOTAL
    # ========================================================

    T_total = (
        T_1
        + T_2
        + T_3
        + T_4
    )


    # ========================================================
    # SPLITTING RATIOS
    # ========================================================

    K_1 = np.divide(
        T_1,
        T_total,

        out=np.zeros_like(
            T_1
        ),

        where=(
            T_total > 0.0
        ),
    )


    K_2 = np.divide(
        T_2,
        T_total,

        out=np.zeros_like(
            T_2
        ),

        where=(
            T_total > 0.0
        ),
    )


    K_3 = np.divide(
        T_3,
        T_total,

        out=np.zeros_like(
            T_3
        ),

        where=(
            T_total > 0.0
        ),
    )


    K_4 = np.divide(
        T_4,
        T_total,

        out=np.zeros_like(
            T_4
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
    # FASES RELATIVAS
    # ========================================================

    output_amplitudes = []


    for index in range(
        1,
        5,
    ):

        amplitude = np.asarray(
            sim_data[f"mode_{index}"].amps
            .sel(
                direction="+",
                mode_index=0,
            )
            .values
        ).squeeze()

        output_amplitudes.append(
            amplitude[order]
        )


    output_amplitudes = np.asarray(
        output_amplitudes
    )


    reference_amplitude = (
        output_amplitudes[0]
    )


    relative_phases_deg = []


    for amplitude in output_amplitudes:

        phase_rad = np.angle(
            amplitude
            * np.conj(
                reference_amplitude
            )
        )

        phase_deg = np.degrees(
            np.unwrap(
                phase_rad
            )
        )

        relative_phases_deg.append(
            phase_deg
        )


    relative_phases_deg = np.asarray(
        relative_phases_deg
    )


    # ========================================================
    # RESULTADO EN lambda0
    # ========================================================

    index_lambda0 = np.argmin(
        np.abs(
            wavelengths_um
            - WAVELENGTH_UM
        )
    )


    print(
        "\n--- RESULTS AT lambda0 ---"
    )

    print(
        f"lambda = "
        f"{wavelengths_um[index_lambda0]:.6f} um"
    )

    print(
        "Powers = "
        f"{[
            f'{value:.6f}'
            for value in [
                T_1[index_lambda0],
                T_2[index_lambda0],
                T_3[index_lambda0],
                T_4[index_lambda0],
            ]
        ]}"
    )

    print(
        f"T_total = "
        f"{T_total[index_lambda0]:.6f}"
    )

    print(
        "Splitting ratio [%] = "
        f"{[
            f'{100 * value:.2f}'
            for value in [
                K_1[index_lambda0],
                K_2[index_lambda0],
                K_3[index_lambda0],
                K_4[index_lambda0],
            ]
        ]}"
    )

    print(
        f"Excess loss = "
        f"{excess_loss_db[index_lambda0]:.4f} dB"
    )


    print(
        "Relative phases [deg] = "
        f"{[
            f'{relative_phases_deg[index, index_lambda0]:+.4f}'
            for index in range(4)
        ]}"
    )


    # ========================================================
    # 1. SPLITTING RATIO
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        wavelengths_um,
        K_1,
        label="Output 1",
        linewidth=2,
    )

    plt.plot(
        wavelengths_um,
        K_2,
        label="Output 2",
        linewidth=2,
    )

    plt.plot(
        wavelengths_um,
        K_3,
        label="Output 3",
        linewidth=2,
    )

    plt.plot(
        wavelengths_um,
        K_4,
        label="Output 4",
        linewidth=2,
    )


    plt.axhline(
        0.25,
        linestyle="--",
        linewidth=1,
        label="Ideal 25%",
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
        "Coupling ratio"
    )

    plt.title(
        "Splitting ratio vs wavelength"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # 2. POTENCIA TOTAL TRANSMITIDA
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


    for index in range(
        4
    ):

        plt.plot(
            wavelengths_um,

            relative_phases_deg[
                index
            ],

            label=(
                f"Output {index + 1}"
            ),

            linewidth=2,
        )


    plt.axhline(
        0.0,
        linestyle=":",
        linewidth=1,
    )

    plt.axhline(
        90.0,
        linestyle=":",
        linewidth=1,
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
        "Phase relative to Output 1 (deg)"
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
    # 5. BANDWIDTH 25:25:25:25
    # ========================================================

    COUPLING_TARGET = 0.25

    COUPLING_TOLERANCE = 0.01


    lower_limit = (
        COUPLING_TARGET
        - COUPLING_TOLERANCE
    )

    upper_limit = (
        COUPLING_TARGET
        + COUPLING_TOLERANCE
    )


    valid = (
        (K_1 >= lower_limit)
        & (K_1 <= upper_limit)

        & (K_2 >= lower_limit)
        & (K_2 <= upper_limit)

        & (K_3 >= lower_limit)
        & (K_3 <= upper_limit)

        & (K_4 >= lower_limit)
        & (K_4 <= upper_limit)
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


        bandwidth_um = (
            lambda_bw_max
            - lambda_bw_min
        )

        bandwidth_nm = (
            1000.0
            * bandwidth_um
        )


    else:

        lambda_bw_min = np.nan
        lambda_bw_max = np.nan

        bandwidth_um = 0.0
        bandwidth_nm = 0.0


    # ========================================================
    # GRAFICA BANDWIDTH
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        wavelengths_um,
        K_1,
        label="Output 1",
        linewidth=2,
    )

    plt.plot(
        wavelengths_um,
        K_2,
        label="Output 2",
        linewidth=2,
    )

    plt.plot(
        wavelengths_um,
        K_3,
        label="Output 3",
        linewidth=2,
    )

    plt.plot(
        wavelengths_um,
        K_4,
        label="Output 4",
        linewidth=2,
    )


    plt.axhline(
        COUPLING_TARGET,
        linestyle="--",
        linewidth=1,
        label="Target 25%",
    )


    plt.axhline(
        lower_limit,
        linestyle=":",
        linewidth=1,
    )

    plt.axhline(
        upper_limit,
        linestyle=":",
        linewidth=1,
    )


    if bandwidth_um > 0.0:

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
        "Coupling ratio"
    )

    plt.title(
        "MMI 1x4 spectral bandwidth"
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
        f"Target coupling: "
        f"{100 * COUPLING_TARGET:.2f}%"
    )

    print(
        f"Tolerance: "
        f"±{100 * COUPLING_TOLERANCE:.2f}%"
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


    # ========================================================
    # FIELD EN lambda0
    # ========================================================

    fig_field, ax_field = (
        plot_fdtd_field_xy(
            sim_data,

            monitor_name=(
                "field_core"
            ),

            field_component="Ey",

            field_part="mag",

            polygons=polygons,
        )
    )

    ax_field.set_title(
        f"FDTD |Ey| - "
        f"{TIDY3D_TASK_NAME}"
    )


plt.show()