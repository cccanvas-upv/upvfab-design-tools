from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td


from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
    BCB, 
    AIR,
)

from upvfab_design_tools.core.waveguides import (
    strip_waveguide_bcb,
)

from upvfab_design_tools.tidy3d_plugin import (
    build_mmi_1x2_fdtd_simulation,
    estimate_tidy3d_cost,
    extract_fdtd_layers,
    load_tidy3d_simulation_data,
    mmi_1x2_vertices,
    plot_fdtd_field_xy,
    plot_mmi_2x2_vertices,
    test_tidy3d_api,
)


RUN_TIDY3D_CLOUD = True
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False


TIDY3D_TASK_NAME = (
    "mmi_1x2_bcb_dy0.0_dLminus1_wt2.1"
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
MMI_WIDTH_UM = 8.0
BCB_HEIGHT_UM = 0.4


# Resultado optimizado con EME + Tidy mode solver

LENGTH_MMI_EME = (
   37.0169
)

dl = -1

MMI_LENGTH_UM = (
    LENGTH_MMI_EME
    + dl
)


ACCESS_WIDTH_UM = 1.0


DY_UM = 0.0

INPUT_Y_POSITION_UM = 0.0


# Dos salidas simetricas

OUTPUT_Y_POSITION_UM = (
    MMI_WIDTH_UM / 3.85
    + DY_UM
)


STRAIGHT_IO_LENGTH_UM = 2.0
TAPER_LENGTH_UM = 15.0


IO_LENGTH_UM = (
    STRAIGHT_IO_LENGTH_UM
    + TAPER_LENGTH_UM
)


# Anchura final del taper

TAPER_WIDTH_UM = 2.1


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
# DOMINIO
# ============================================================

DOMAIN_X_SPAN_UM = 18.0

BOTTOM_MARGIN_UM = 1.5
TOP_MARGIN_UM = 1.5


# ============================================================
# STACK VERTICAL
# ============================================================

cross_section = strip_waveguide_bcb(
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
# GEOMETRIA MMI 1x2
# ============================================================

polygons = mmi_1x2_vertices(
    length_um=(
        MMI_LENGTH_UM
    ),

    io_length_um=(
        IO_LENGTH_UM
    ),

    straight_io_length_um=(
        STRAIGHT_IO_LENGTH_UM
    ),

    output_y_position_um=(
        OUTPUT_Y_POSITION_UM
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

sim = build_mmi_1x2_fdtd_simulation(
    polygons=polygons,

    cross_section=cross_section,

    wavelength_um=(
        WAVELENGTH_UM
    ),

    output_y_position_um=(
        OUTPUT_Y_POSITION_UM
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
# PARAMETROS
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "FDTD MMI 1x2 - SPECTRAL SWEEP"
)

print(
    "=" * 60
)

print(
    f"Wavelength sweep: "
    f"{WAVELENGTH_MIN_UM:.4f} - "
    f"{WAVELENGTH_MAX_UM:.4f} um"
)

print(
    f"Number of wavelengths: "
    f"{NUM_WAVELENGTHS}"
)

print(
    f"MMI width: "
    f"{MMI_WIDTH_UM:.4f} um"
)

print(
    f"MMI length: "
    f"{MMI_LENGTH_UM:.4f} um"
)

print(
    f"Access width: "
    f"{ACCESS_WIDTH_UM:.4f} um"
)

print(
    f"dy: "
    f"{DY_UM:+.4f} um"
)

print(
    f"Output positions: "
    f"+/- {OUTPUT_Y_POSITION_UM:.6f} um"
)

print(
    f"Taper width: "
    f"{TAPER_WIDTH_UM:.4f} um"
)

print(
    "=" * 60
)


# ============================================================
# INFORMACION DE LA SIMULACION
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
    plot_mmi_2x2_vertices(
        polygons
    )
)

ax_vertices.set_title(
    "MMI 1x2 - FDTD geometry"
)


# ============================================================
# REPRESENTAR SIMULACION
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

    flux_top_data = (
        sim_data["flux_top"].flux
    )

    flux_bottom_data = (
        sim_data["flux_bottom"].flux
    )


    frequencies = np.asarray(
        flux_top_data.f.values,
        dtype=float,
    )


    wavelengths_um = (
        td.C_0
        / frequencies
    )


    T_top = np.real(
        np.asarray(
            flux_top_data.values
        ).squeeze()
    )

    T_bottom = np.real(
        np.asarray(
            flux_bottom_data.values
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

    T_top = (
        T_top[order]
    )

    T_bottom = (
        T_bottom[order]
    )


    # ========================================================
    # POTENCIA TOTAL
    # ========================================================

    T_total = (
        T_top
        + T_bottom
    )


    # ========================================================
    # SPLITTING RATIOS
    # ========================================================

    K_top = np.divide(
        T_top,
        T_total,

        out=np.zeros_like(
            T_top
        ),

        where=(
            T_total > 0.0
        ),
    )


    K_bottom = np.divide(
        T_bottom,
        T_total,

        out=np.zeros_like(
            T_bottom
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
    # FASE RELATIVA TE0
    # ========================================================

    amp_top = np.asarray(
        sim_data["mode_top"].amps
        .sel(
            direction="+",
            mode_index=0,
        )
        .values
    ).squeeze()


    amp_bottom = np.asarray(
        sim_data["mode_bottom"].amps
        .sel(
            direction="+",
            mode_index=0,
        )
        .values
    ).squeeze()


    amp_top = (
        amp_top[order]
    )

    amp_bottom = (
        amp_bottom[order]
    )


    phase_difference_rad = np.angle(
        amp_top
        * np.conj(
            amp_bottom
        )
    )


    phase_difference_deg = np.degrees(
        np.unwrap(
            phase_difference_rad
        )
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
        f"T_top = "
        f"{T_top[index_lambda0]:.6f}"
    )

    print(
        f"T_bottom = "
        f"{T_bottom[index_lambda0]:.6f}"
    )

    print(
        f"T_total = "
        f"{T_total[index_lambda0]:.6f}"
    )

    print(
        f"Splitting ratio = "
        f"{100 * K_top[index_lambda0]:.2f}:"
        f"{100 * K_bottom[index_lambda0]:.2f}"
    )

    print(
        f"Excess loss = "
        f"{excess_loss_db[index_lambda0]:.4f} dB"
    )

    print(
        f"Phase TOP - BOTTOM = "
        f"{phase_difference_deg[index_lambda0]:+.4f} deg"
    )


    # ========================================================
    # 1. SPLITTING RATIO
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        wavelengths_um,
        K_top,
        label="Top output",
        linewidth=2,
    )

    plt.plot(
        wavelengths_um,
        K_bottom,
        label="Bottom output",
        linewidth=2,
    )


    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
        label="Ideal 50:50",
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
    # 4. FASE RELATIVA
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        wavelengths_um,
        phase_difference_deg,
        linewidth=2,
    )


    plt.axhline(
        0.0,
        linestyle=":",
        linewidth=1,
        label="Ideal 0 deg",
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
        "Phase TOP - BOTTOM (deg)"
    )

    plt.title(
        "Relative output phase vs wavelength"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # 5. BANDWIDTH 50:50
    # ========================================================

    COUPLING_TARGET = 0.50

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
        (K_top >= lower_limit)
        & (K_top <= upper_limit)

        & (K_bottom >= lower_limit)
        & (K_bottom <= upper_limit)
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
        K_top,
        label="Top output",
        linewidth=2,
    )

    plt.plot(
        wavelengths_um,
        K_bottom,
        label="Bottom output",
        linewidth=2,
    )


    plt.axhline(
        COUPLING_TARGET,
        linestyle="--",
        linewidth=1,
        label="Target 50%",
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
        "MMI 1x2 spectral bandwidth"
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