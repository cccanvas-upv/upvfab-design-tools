from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.waveguides import strip_waveguide

from upvfab_design_tools.tidy3d_plugin import (
    build_mmi_1x4_fdtd_simulation,
    estimate_tidy3d_cost,
    extract_fdtd_layers,
    load_tidy3d_simulation_data,
    mmi_1x4_vertices,
    plot_fdtd_field_xy,
    plot_mmi_2x2_vertices,
    print_mmi_1x4_fluxes,
    test_tidy3d_api,
)


RUN_TIDY3D_CLOUD = True
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False

TIDY3D_TASK_NAME = "mmi_1x4_Lememinus1.3_wt1.5"

TIDY3D_MATERIAL_BACKEND = "upvfab"
# "upvfab" -> materials.py
# "tidy3d" -> material_library nativa de Tidy3D

from upvfab_design_tools.tidy3d_plugin import conversion as tidy_conversion

tidy_conversion.TIDY3D_MATERIAL_BACKEND = (TIDY3D_MATERIAL_BACKEND)

TIDY3D_DATA_PATH = (f"data/{TIDY3D_TASK_NAME}.hdf5")

# PARAMETROS DISPOSITIVO
WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.3
MMI_WIDTH_UM = 12.0

# Resultado optimizado con EME + Tidy mode solver
DL = -1.3
MMI_LENGTH_UM = 39.9954 + DL


ACCESS_WIDTH_UM = 1.0

# Anchura del taper
TAPER_WIDTH_UM = 1.5
TAPER_LENGTH_UM = 15.0

# Tramo recto exterior para source / monitors
STRAIGHT_IO_LENGTH_UM = 2.0

# Longitud total desde el MMI hasta el extremo
IO_LENGTH_UM = (TAPER_LENGTH_UM + STRAIGHT_IO_LENGTH_UM)

# Source y monitors en el centro
# de los tramos rectos

SOURCE_OFFSET_UM = ( 0.5 * STRAIGHT_IO_LENGTH_UM)

MONITOR_OFFSET_UM = (0.5 * STRAIGHT_IO_LENGTH_UM)

MODE_PLANE_Y_SPAN_UM = (2.5 * ACCESS_WIDTH_UM)

# POSICIONES DE LAS 4 SALIDAS

# IMPORTANTE:
# Deben estar ordenadas de abajo hacia arriba:
# (ABAJO) y1 < y2 < y3 < y4 (ARRIBA)


OUTPUT_Y_POSITIONS_UM = (
    -4.65,
    -1.54 ,
    +1.54,
    +4.65,
)

DOMAIN_X_SPAN_UM = (MMI_WIDTH_UM + 4.0)
BOTTOM_MARGIN_UM = 1.5
TOP_MARGIN_UM = 1.5

# STACK VERTICAL

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

# COMPROBAR STACK

core, surrounding = extract_fdtd_layers(cross_section)

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



# GEOMETRIA MMI 1x4

polygons = mmi_1x4_vertices(
    length_um=MMI_LENGTH_UM,
    io_length_um=IO_LENGTH_UM,
    straight_io_length_um=(STRAIGHT_IO_LENGTH_UM),
    output_y_positions_um=( OUTPUT_Y_POSITIONS_UM),
    access_width_um=ACCESS_WIDTH_UM,
    taper_width_um=TAPER_WIDTH_UM,
    mmi_width_input_um=MMI_WIDTH_UM,
    mmi_width_center_um=MMI_WIDTH_UM,
)

# SIMULACIÓN

sim = build_mmi_1x4_fdtd_simulation(
    polygons=polygons,

    cross_section=cross_section,

    wavelength_um=WAVELENGTH_UM,

    output_y_positions_um=(
        OUTPUT_Y_POSITIONS_UM
    ),

    access_width_um=ACCESS_WIDTH_UM,

    straight_io_length_um=(
        STRAIGHT_IO_LENGTH_UM
    ),

    bandwidth_um=0.1,

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


sim.validate_pre_upload()

if ESTIMATE_TIDY3D_COST:

    test_tidy3d_api()

    estimated_cost = estimate_tidy3d_cost(
        sim,
        task_name=f"{TIDY3D_TASK_NAME}_cost",
        verbose=True,
    )

    print(
        f"\nEstimated Tidy3D cost: "
        f"{estimated_cost:.6g} FlexCredits"
    )

print(
    "\nSimulation validated successfully."
)


# INFO DE LA SIMULACION

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


# REPRESENTAR POLIGONOS

fig_vertices, ax_vertices = (
    plot_mmi_2x2_vertices(
        polygons
    )
)

ax_vertices.set_title("MMI 1x4 - FDTD geometry")


# REPRESENTAR SIMULACION TIDY3D

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

ax_sim.set_title("Tidy3D geometry, source and monitors")


# REPRESENTAR PERMITIVIDAD


fig_eps, ax_eps = plt.subplots(figsize=(10, 4))
sim.plot_eps(z=0.15,ax=ax_eps,)

ax_eps.set_aspect("equal", adjustable="box")

ax_eps.set_title("Tidy3D permittivity at z = 0.15 um")


sim_data = None


if LOAD_TIDY3D_DATA:

    print(
        f"\nLoading Tidy3D data from:"
        f"\n{TIDY3D_DATA_PATH}"
    )

    sim_data = load_tidy3d_simulation_data(
        TIDY3D_DATA_PATH
    )


if RUN_TIDY3D_CLOUD:

    import tidy3d.web as web

    test_tidy3d_api()

    print("\n" + "=" * 60)
    print("RUNNING TIDY3D CLOUD SIMULATION")
    print("=" * 60)

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
        task_name=TIDY3D_TASK_NAME,
        path=TIDY3D_DATA_PATH,
        verbose=True,
    )


# POSTPROCESADO


if sim_data is not None:

    print("\n" + "=" * 60)
    print("FDTD RESULTS")
    print("=" * 60)

    (
        R,
        T_1,
        T_2,
        T_3,
        T_4,
    ) = print_mmi_1x4_fluxes(
        sim_data
    )

    T_total = (
        T_1
        + T_2
        + T_3
        + T_4
    )


    # Excess loss respecto a potencia de entrada normalizada a 1

    if T_total > 0.0:

        excess_loss_db = (
            -10.0
            * np.log10(T_total)
        )

    else:

        excess_loss_db = np.inf


    # Splitting ratio normalizado respecto
    # a la potencia total de las cuatro salidas

    if T_total > 0.0:

        splitting_ratio = np.array(
            [
                T_1 / T_total,
                T_2 / T_total,
                T_3 / T_total,
                T_4 / T_total,
            ]
        )

    else:

        splitting_ratio = np.zeros(4)


    print(
        "\n--- PERFORMANCE METRICS ---"
    )

    print(
        f"Excess loss [dB]: "
        f"{excess_loss_db:.4f}"
    )
    print(
            "Splitting ratio: "
            f"{[
                f'{value:.4f}'
                for value in splitting_ratio
            ]}"
        )

    print(
    "Power over outputs: "
    f"{[f'{value:.4f}' for value in [T_1, T_2, T_3, T_4]]}"
    )
  

    print(
        "Splitting ratio [%]: "
        f"{[
            f'{100 * value:.2f}'
            for value in splitting_ratio
        ]}"
    )


    
    # FIELD

    fig_field, ax_field = (
        plot_fdtd_field_xy(
            sim_data,
            monitor_name="field_core",
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



