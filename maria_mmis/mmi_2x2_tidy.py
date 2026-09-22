from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.waveguides import strip_waveguide

from upvfab_design_tools.tidy3d_plugin import (
    build_mmi_2x2_fdtd_simulation,
    estimate_tidy3d_cost,
    extract_fdtd_layers,
    load_tidy3d_simulation_data,
    mmi_2x2_vertices,
    plot_fdtd_field_xy,
    plot_mmi_2x2_vertices,
    print_mmi_2x2_fluxes,
    test_tidy3d_api,
)

RUN_TIDY3D_CLOUD = False 
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False

TIDY3D_TASK_NAME = "mmi_2x2_no_tapers_tidymaterials"

TIDY3D_MATERIAL_BACKEND = "tidy3d"
# "upvfab" -> usa materials.py
# "tidy3d" -> usa la material_library nativa de Tidy3D

# Archivo local donde se guardaran los resultados
TIDY3D_DATA_PATH = f"data/{TIDY3D_TASK_NAME}.hdf5"


# PARAMETROS DISPOSITIVO

WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.3
MMI_WIDTH_UM = 8.0

# Resultado optimizado con EME + Tidy mode solver
MMI_LENGTH_UM = 48.9487

ACCESS_WIDTH_UM = 1.6

DY_UM = 0.12

# Posicion de las guias:
# +/- (Wmmi/6 + dy)
IO_Y_POSITION_UM = (MMI_WIDTH_UM / 6.0+ DY_UM)

# Longitud de las guias rectas antes/despues del MMI
IO_LENGTH_UM = 10.0

# SIN TAPER:
# la anchura final del supuesto taper es igual
# a la anchura de la guia de acceso.
TAPER_WIDTH_UM = ACCESS_WIDTH_UM

INPUT_PORT = "bottom"

MODE_PLANE_Y_SPAN_UM = (2.5 * ACCESS_WIDTH_UM)

# DOMINIO VERTICAL

DOMAIN_X_SPAN_UM = 18.0

DOMAIN_Z_MIN_UM = -2.0
DOMAIN_Z_MAX_UM = 2.0

BOTTOM_MARGIN_UM = 2.0
TOP_MARGIN_UM = 2.0



# STACK VERTICAL
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
    name="sin_strip_fdtd_stack",
)


# COMPROBAR STACK

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


# GEOMETRIA 2x2 SIN TAPERS

polygons = mmi_2x2_vertices(
    length_um=MMI_LENGTH_UM,
    io_length_um=IO_LENGTH_UM,
    io_y_position_um=IO_Y_POSITION_UM,
    access_width_um=ACCESS_WIDTH_UM,
    taper_width_um=TAPER_WIDTH_UM,
    mmi_width_input_um=MMI_WIDTH_UM,
    mmi_width_center_um=MMI_WIDTH_UM,
)


# CONSTRUIR SIMULACION FDTD

sim = build_mmi_2x2_fdtd_simulation(
    polygons=polygons,
    cross_section=cross_section,
    wavelength_um=WAVELENGTH_UM,

    input_port=INPUT_PORT,

    io_y_position_um=IO_Y_POSITION_UM,
    access_width_um=ACCESS_WIDTH_UM,

    bandwidth_um=0.1,

    pad_x_um=2.0,
    pad_y_um=2.0,
    pad_z_um=0.0,

    mode_plane_y_span_um=MODE_PLANE_Y_SPAN_UM,


    min_steps_per_wvl=12,

    run_time_s=5e-12,
)


# VALIDACION LOCAL


sim.validate_pre_upload()

if ESTIMATE_TIDY3D_COST:
    test_tidy3d_api()

    estimated_cost = estimate_tidy3d_cost(
        sim,
        task_name="mmi_2x2_no_tapers_cost",
        verbose=True,
    )

    print(
        f"\nEstimated Tidy3D cost: "
        f"{estimated_cost:.6g} FlexCredits"
    )

print("\nSimulation validated successfully.")



# PARAMETROS


print("\n" + "=" * 60)
print("FDTD MMI 2x2 - LOCAL GEOMETRY CHECK")
print("=" * 60)

print(
    f"Wavelength: "
    f"{WAVELENGTH_UM:.4f} um"
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
    f"Wmmi / 6: "
    f"{MMI_WIDTH_UM / 6.0:.6f} um"
)

print(
    f"IO y position: "
    f"+/- {IO_Y_POSITION_UM:.6f} um"
)

print(
    f"Taper width: "
    f"{TAPER_WIDTH_UM:.4f} um"
)

print(
    f"Input port: "
    f"{INPUT_PORT}"
)

print("=" * 60)



# INFORMACION DE LA SIMULACION

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

ax_vertices.set_title(
    "MMI 2x2 - no tapers"
)



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

ax_sim.set_title(
    "Tidy3D geometry, source and monitors"
)



# REPRESENTAR PERMITIVIDAD

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

sim_data = None


# ------------------------------------------------------------
# CARGAR RESULTADOS YA EXISTENTES
# ------------------------------------------------------------

if LOAD_TIDY3D_DATA:

    print(
        f"\nLoading Tidy3D data from:"
        f"\n{TIDY3D_DATA_PATH}"
    )

    sim_data = load_tidy3d_simulation_data(
        TIDY3D_DATA_PATH
    )


# ------------------------------------------------------------
# EJECUTAR SIMULACION EN TIDY3D CLOUD
# ------------------------------------------------------------

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


# ============================================================
# POSTPROCESADO
# ============================================================

if sim_data is not None:

    print("\n" + "=" * 60)
    print("FDTD RESULTS")
    print("=" * 60)

    print_mmi_2x2_fluxes(sim_data)
    flux_top = float(
        np.real(
            sim_data["flux_top"].flux.values.squeeze()
        )
    )

    flux_bottom = float(
        np.real(
            sim_data["flux_bottom"].flux.values.squeeze()
        )
    )

    T_top = flux_top
    T_bottom = flux_bottom

    T_total = T_top + T_bottom


    # Excess loss respecto a potencia de entrada normalizada a 1
    if T_total > 0.0:
        excess_loss_db = -10.0 * np.log10(T_total)
    else:
        excess_loss_db = np.inf


    # Splitting ratio normalizado respecto a la potencia
    # que llega a ambas salidas
    if T_total > 0.0:
        splitting_ratio = np.array(
            [
                T_top / T_total,
                T_bottom / T_total,
            ]
        )
    else:
        splitting_ratio = np.zeros(2)


    print("\n--- PERFORMANCE METRICS ---")

    print(
        f"Excess loss [dB]: "
        f"{excess_loss_db:.4f}"
    )

    print(
        "Splitting ratio: "
        f"{[f'{value:.4f}' for value in splitting_ratio]}"
    )

    print(
        "Splitting ratio [%]: "
        f"{[f'{100 * value:.2f}' for value in splitting_ratio]}"
    )
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
