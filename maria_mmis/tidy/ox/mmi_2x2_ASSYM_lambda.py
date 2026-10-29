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
   build_mmi_2x2_asymmetric_fdtd_simulation,
    estimate_tidy3d_cost,
    extract_fdtd_layers,
    load_tidy3d_simulation_data,
    mmi_2x2_asymmetric_vertices,
    plot_fdtd_field_xy,
    plot_mmi_2x2_vertices,
    test_tidy3d_api,
)


RUN_TIDY3D_CLOUD = True
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False

TIDY3D_TASK_NAME = (
    "mmi72_Lememinus1.6_dyminus0.05_wt2.5_sweep"
)

TIDY3D_DATA_PATH = (
    f"data/{TIDY3D_TASK_NAME}.hdf5"
)

CORE_HEIGHT_UM = 0.30
MMI_WIDTH_UM = 10.0

MMI_LENGTH_UM = 88.7813 - 1.6


ACCESS_WIDTH_UM = 1
TAPER_WIDTH_UM = 2.5

STRAIGHT_IO_LENGTH_UM = 2.0
TAPER_LENGTH_UM = 50.0
IO_LENGTH_UM = STRAIGHT_IO_LENGTH_UM + TAPER_LENGTH_UM

DY_UM = -0.05

# Distancias analiticas desde los bordes
D1_UM = MMI_WIDTH_UM / 5.0
D0_UM = 2.0 * MMI_WIDTH_UM / 5.0

INPUT_Y_POSITIONS_UM = (
    -MMI_WIDTH_UM / 2 + D0_UM + DY_UM,
    +MMI_WIDTH_UM / 2 - D1_UM - DY_UM,
)

OUTPUT_Y_POSITIONS_UM = (
    -MMI_WIDTH_UM / 2 + D1_UM + DY_UM,
    +MMI_WIDTH_UM / 2 - D0_UM - DY_UM,
)

INPUT_PORT = "bottom"

SOURCE_OFFSET_UM = 0.5 * STRAIGHT_IO_LENGTH_UM
MONITOR_OFFSET_UM = 0.5 * STRAIGHT_IO_LENGTH_UM
MODE_PLANE_Y_SPAN_UM = 2.5 * ACCESS_WIDTH_UM

TAPER_GAP_UM = (
    INPUT_Y_POSITIONS_UM[1]
    - INPUT_Y_POSITIONS_UM[0]
    - TAPER_WIDTH_UM
)

EDGE_CLEARANCE_UM = (
    MMI_WIDTH_UM / 2
    - max(
        abs(y)
        for y in (
            *INPUT_Y_POSITIONS_UM,
            *OUTPUT_Y_POSITIONS_UM,
        )
    )
    - TAPER_WIDTH_UM / 2
)

print("\n--- 72:28 GEOMETRY ---")
print(f"Input positions:  {INPUT_Y_POSITIONS_UM} um")
print(f"Output positions: {OUTPUT_Y_POSITIONS_UM} um")
print(f"Taper gap: {TAPER_GAP_UM:.4f} um")
print(f"Edge clearance: {EDGE_CLEARANCE_UM:.4f} um")



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


# DOMINIO VERTICAL


DOMAIN_X_SPAN_UM = 18.0

BOTTOM_MARGIN_UM = 1.5
TOP_MARGIN_UM = 1.5


# ============================================================
# STACK VERTICAL
# ============================================================

cross_section = strip_waveguide(
    width=ACCESS_WIDTH_UM,
    height=CORE_HEIGHT_UM,

    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,

    x_center=0.0,
    z_min=0.0,

    x_span=DOMAIN_X_SPAN_UM,

    bottom_margin=BOTTOM_MARGIN_UM,
    top_margin=TOP_MARGIN_UM,

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
# GEOMETRIA MMI 2x2
# ============================================================

polygons = mmi_2x2_asymmetric_vertices(
    length_um=MMI_LENGTH_UM,
    io_length_um=IO_LENGTH_UM,
    straight_io_length_um=STRAIGHT_IO_LENGTH_UM,
    input_y_positions_um=INPUT_Y_POSITIONS_UM,
    output_y_positions_um=OUTPUT_Y_POSITIONS_UM,
    access_width_um=ACCESS_WIDTH_UM,
    taper_width_um=TAPER_WIDTH_UM,
    mmi_width_um=MMI_WIDTH_UM,
)


# ============================================================
# CONSTRUIR SIMULACION FDTD
# ============================================================

sim = build_mmi_2x2_asymmetric_fdtd_simulation(
    polygons=polygons,
    cross_section=cross_section,
    wavelength_um=WAVELENGTH_UM,

    input_y_positions_um=INPUT_Y_POSITIONS_UM,
    output_y_positions_um=OUTPUT_Y_POSITIONS_UM,

    access_width_um=ACCESS_WIDTH_UM,
    input_port=INPUT_PORT,

    bandwidth_um=BANDWIDTH_UM,

    pad_x_um=2.0,
    pad_y_um=2.0,
    pad_z_um=0.0,

    source_offset_um=SOURCE_OFFSET_UM,
    monitor_offset_um=MONITOR_OFFSET_UM,

    mode_plane_y_span_um=MODE_PLANE_Y_SPAN_UM,

    min_steps_per_wvl=16,
    run_time_s=5e-12,
)

print("\n--- MATERIAL CHECK ---")

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


# CONVERTIR MONITORES A BARRIDO ESPECTRAL
SPECTRAL_MONITORS = (
    "flux_top",
    "flux_bottom",
    "mode_top",
    "mode_bottom",
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
        # field_core solamente en lambda0
        updated_monitors.append(
            monitor
        )

sim = sim.updated_copy(
    monitors=updated_monitors,
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
# INFORMACION
# ============================================================

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
    f"{DY_UM:+.4f} um"
)


print(
    f"Taper width: "
    f"{TAPER_WIDTH_UM:.4f} um"
)

print(
    f"Input port: "
    f"{INPUT_PORT}"
)


print(
    "\nSimulation center:",
    sim.center,
)

print(
    "Simulation size:",
    sim.size,
)

print(
    "Monitors:",
    [
        monitor.name
        for monitor in sim.monitors
    ],
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
    "MMI 2x2 - FDTD geometry"
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

    sim_data = load_tidy3d_simulation_data(
        TIDY3D_DATA_PATH
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
# POSTPROCESADO ESPECTRAL 72:28
# ============================================================

if sim_data is not None:

    flux_top_data = sim_data["flux_top"].flux
    flux_bottom_data = sim_data["flux_bottom"].flux

    frequencies = np.asarray(
        flux_top_data.f.values,
        dtype=float,
    )

    wavelengths_um = td.C_0 / frequencies

    T_top = np.real(
        np.asarray(flux_top_data.values).squeeze()
    )

    T_bottom = np.real(
        np.asarray(flux_bottom_data.values).squeeze()
    )


    # --------------------------------------------------------
    # ORDENAR POR LONGITUD DE ONDA
    # --------------------------------------------------------

    order = np.argsort(wavelengths_um)

    wavelengths_um = wavelengths_um[order]
    T_top = T_top[order]
    T_bottom = T_bottom[order]


    # --------------------------------------------------------
    # TRANSMISION Y SPLITTING
    # --------------------------------------------------------

    T_total = T_top + T_bottom

    K_top = np.divide(
        T_top,
        T_total,
        out=np.zeros_like(T_top),
        where=T_total > 0,
    )

    K_bottom = np.divide(
        T_bottom,
        T_total,
        out=np.zeros_like(T_bottom),
        where=T_total > 0,
    )

    excess_loss_db = -10.0 * np.log10(
        np.maximum(T_total, 1e-20)
    )


    # --------------------------------------------------------
    # FASE TE0
    # --------------------------------------------------------

    a_top = np.asarray(
        sim_data["mode_top"].amps
        .sel(direction="+", mode_index=0)
        .values
    ).squeeze()[order]

    a_bottom = np.asarray(
        sim_data["mode_bottom"].amps
        .sel(direction="+", mode_index=0)
        .values
    ).squeeze()[order]

    phase_difference_deg = np.degrees(
        np.unwrap(
            np.angle(
                a_top * np.conj(a_bottom)
            )
        )
    )


    # --------------------------------------------------------
    # lambda0
    # --------------------------------------------------------

    index_lambda0 = np.argmin(
        np.abs(
            wavelengths_um - WAVELENGTH_UM
        )
    )

    high_is_top = (
        K_top[index_lambda0]
        >= K_bottom[index_lambda0]
    )

    high_output = (
        "top"
        if high_is_top
        else "bottom"
    )

    K_high = (
        K_top
        if high_is_top
        else K_bottom
    )

    K_low = (
        K_bottom
        if high_is_top
        else K_top
    )


    print("\n" + "=" * 60)
    print("FDTD 72:28 SPECTRAL RESULTS")
    print("=" * 60)

    print(
        f"lambda = "
        f"{wavelengths_um[index_lambda0]:.4f} um"
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
        f"Splitting TOP:BOTTOM = "
        f"{100*K_top[index_lambda0]:.2f}:"
        f"{100*K_bottom[index_lambda0]:.2f}"
    )

    print(
        f"High-power output = "
        f"{high_output}"
    )

    print(
        f"72:28 ratio = "
        f"{100*K_high[index_lambda0]:.2f}:"
        f"{100*K_low[index_lambda0]:.2f}"
    )

    print(
        f"Excess loss = "
        f"{excess_loss_db[index_lambda0]:.4f} dB"
    )

    print(
        f"Phase TOP - BOTTOM = "
        f"{phase_difference_deg[index_lambda0]:+.3f} deg"
    )


    # ========================================================
    # COUPLING RATIO VS LAMBDA
    # ========================================================

    plt.figure(figsize=(8, 5))

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
        0.72,
        linestyle="--",
        linewidth=1,
        label="Target 72%",
    )

    plt.axhline(
        0.28,
        linestyle="--",
        linewidth=1,
        label="Target 28%",
    )

    plt.axvline(
        WAVELENGTH_UM,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Coupling ratio")
    plt.title("72:28 coupling ratio vs wavelength")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


    # ========================================================
    # TRANSMISION TOTAL
    # ========================================================

    plt.figure(figsize=(8, 5))

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

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Total transmitted power")
    plt.title("Total transmission vs wavelength")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    # ========================================================
    # EXCESS LOSS
    # ========================================================

    plt.figure(figsize=(8, 5))

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

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Excess loss (dB)")
    plt.title("Excess loss vs wavelength")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    # ========================================================
    # FASE
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        wavelengths_um,
        phase_difference_deg,
        linewidth=2,
    )

    plt.axvline(
        WAVELENGTH_UM,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Phase TOP - BOTTOM (deg)")
    plt.title("Relative phase vs wavelength")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    # ========================================================
    # BANDWIDTH 72:28
    # ========================================================

    TARGET_HIGH = 0.72
    TARGET_LOW = 0.28

    COUPLING_TOLERANCE = 0.01

    valid = (
        (np.abs(K_high - TARGET_HIGH)
         <= COUPLING_TOLERANCE)
        &
        (np.abs(K_low - TARGET_LOW)
         <= COUPLING_TOLERANCE)
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

        lambda_bw_min = wavelengths_um[i_left]
        lambda_bw_max = wavelengths_um[i_right]

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


    # --------------------------------------------------------
    # GRAFICA BANDWIDTH
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        wavelengths_um,
        K_high,
        label="High-power output",
        linewidth=2,
    )

    plt.plot(
        wavelengths_um,
        K_low,
        label="Low-power output",
        linewidth=2,
    )

    plt.axhline(
        TARGET_HIGH,
        linestyle="--",
        linewidth=1,
        label="72%",
    )

    plt.axhline(
        TARGET_LOW,
        linestyle="--",
        linewidth=1,
        label="28%",
    )

    plt.axhline(
        TARGET_HIGH + COUPLING_TOLERANCE,
        linestyle=":",
        linewidth=1,
    )

    plt.axhline(
        TARGET_HIGH - COUPLING_TOLERANCE,
        linestyle=":",
        linewidth=1,
    )

    if bandwidth_nm > 0:
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
    )

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Coupling ratio")
    plt.title("72:28 spectral bandwidth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


    print("\n--- 72:28 BANDWIDTH ---")

    print(
        f"Target = "
        f"{100*TARGET_HIGH:.0f}:"
        f"{100*TARGET_LOW:.0f}"
    )

    print(
        f"Tolerance = "
        f"±{100*COUPLING_TOLERANCE:.2f} percentage points"
    )

    print(
        f"lambda_min = "
        f"{lambda_bw_min:.4f} um"
    )

    print(
        f"lambda_max = "
        f"{lambda_bw_max:.4f} um"
    )

    print(
        f"Bandwidth = "
        f"{bandwidth_nm:.2f} nm"
    )


    # ========================================================
    # FIELD EN lambda0
    # ========================================================

    fig_field, ax_field = plot_fdtd_field_xy(
        sim_data,
        monitor_name="field_core",
        field_component="Ey",
        field_part="mag",
        polygons=polygons,
    )

    ax_field.set_title(
        f"FDTD |Ey| - {TIDY3D_TASK_NAME}"
    )


plt.show()