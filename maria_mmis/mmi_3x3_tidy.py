from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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


RUN_TIDY3D_CLOUD = True
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False

INPUT_CONFIGURATION = "hybrid"  # "top", "bottom", "hybrid"
DELTA_PHI_IN_DEG = 0.0

TIDY3D_TASK_NAME = f"mmi_3x3_Lminus2.2_dy0.09_wt1.9_{INPUT_CONFIGURATION}"
TIDY3D_DATA_PATH = f"data/{TIDY3D_TASK_NAME}.hdf5"


# PARAMETROS DISPOSITIVO

WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.30

MMI_WIDTH_UM = 10.0

# Sustituye estos dos por tu optimo EME
MMI_LENGTH_UM = 149.968844 -2.2
DY = 0.09
PORT_POSITION_UM = MMI_WIDTH_UM / 3.0 + DY

ACCESS_WIDTH_UM = 1
TAPER_WIDTH_UM = 1.9
TAPER_LENGTH_UM = 20.0
STRAIGHT_IO_LENGTH_UM = 2.0

IO_LENGTH_UM = TAPER_LENGTH_UM + STRAIGHT_IO_LENGTH_UM

SOURCE_OFFSET_UM = 0.5 * STRAIGHT_IO_LENGTH_UM
MONITOR_OFFSET_UM = 0.5 * STRAIGHT_IO_LENGTH_UM
MODE_PLANE_Y_SPAN_UM = 2.5 * ACCESS_WIDTH_UM

DOMAIN_X_SPAN_UM = MMI_WIDTH_UM + 4.0


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

polygons = mmi_3x3_vertices(
    length_um=MMI_LENGTH_UM,
    io_length_um=IO_LENGTH_UM,
    straight_io_length_um=STRAIGHT_IO_LENGTH_UM,
    port_position_um=PORT_POSITION_UM,
    access_width_um=ACCESS_WIDTH_UM,
    taper_width_um=TAPER_WIDTH_UM,
    mmi_width_um=MMI_WIDTH_UM,
)


# SIMULACION

sim = build_mmi_3x3_fdtd_simulation(
    polygons=polygons,
    cross_section=cross_section,
    wavelength_um=WAVELENGTH_UM,
    port_position_um=PORT_POSITION_UM,
    access_width_um=ACCESS_WIDTH_UM,
    input_configuration=INPUT_CONFIGURATION,
    delta_phi_rad=np.deg2rad(DELTA_PHI_IN_DEG),
    bandwidth_um=0.1,
    pad_x_um=2.0,
    pad_y_um=2.0,
    pad_z_um=0.0,
    source_offset_um=SOURCE_OFFSET_UM,
    monitor_offset_um=MONITOR_OFFSET_UM,
    mode_plane_y_span_um=MODE_PLANE_Y_SPAN_UM,
    min_steps_per_wvl=16,
    run_time_s=5e-12,
)

sim.validate_pre_upload()


# COMPROBAR GEOMETRIA

fig, ax = plt.subplots(figsize=(12, 4))
sim.plot(z=0.5 * CORE_HEIGHT_UM, ax=ax)
ax.set_aspect("equal", adjustable="box")

fig, ax = plt.subplots(figsize=(12, 4))
sim.plot_eps(z=0.5 * CORE_HEIGHT_UM, ax=ax)
ax.set_aspect("equal", adjustable="box")


# COSTE

if ESTIMATE_TIDY3D_COST:
    test_tidy3d_api()
    cost = estimate_tidy3d_cost(
        sim,
        task_name=f"{TIDY3D_TASK_NAME}_cost",
        verbose=True,
    )
    print(f"\nEstimated cost: {cost:.6g} FlexCredits")


# CARGAR / EJECUTAR

sim_data = None

if LOAD_TIDY3D_DATA:
    sim_data = load_tidy3d_simulation_data(TIDY3D_DATA_PATH)

if RUN_TIDY3D_CLOUD:
    import tidy3d.web as web

    test_tidy3d_api()
    sim_data = web.run(
        sim,
        task_name=TIDY3D_TASK_NAME,
        path=TIDY3D_DATA_PATH,
        verbose=True,
    )


# POSTPROCESADO

if sim_data is not None:
    names = ("bottom", "center", "top")

    flux = np.array([
        float(np.real(sim_data[f"flux_{name}"].flux.values.squeeze()))
        for name in names
    ])

    total = np.sum(flux)
    split = flux / total if total > 0 else np.zeros(3)
    excess_loss_db = -10 * np.log10(total) if total > 0 else np.inf

    amps = np.array([
        complex(
            sim_data[f"mode_{name}"].amps
            .sel(direction="+", mode_index=0)
            .isel(f=0)
            .values.squeeze()
        )
        for name in names
    ])

    relative_phase_deg = np.degrees(
        np.angle(amps * np.conj(amps[1]))
    )

    print("\n" + "=" * 60)
    print("FDTD MMI 3x3 RESULTS")
    print("=" * 60)
    print(f"Input: {INPUT_CONFIGURATION}")
    print(f"Delta phi input: {DELTA_PHI_IN_DEG:+.2f} deg")
    print(f"Total output: {total:.6f}")
    print(f"Excess loss: {excess_loss_db:.4f} dB")

    for name, T, K, phase in zip(
        names, flux, split, relative_phase_deg, strict=True
    ):
        print(
            f"{name:>6s}: "
            f"T={T:.6f}, "
            f"split={100*K:.2f} %, "
            f"phase rel. center={phase:+.2f} deg"
        )

    fig, ax = plot_fdtd_field_xy(
        sim_data,
        monitor_name="field_core",
        field_component="Ey",
        field_part="mag",
        polygons=polygons,
    )
    ax.set_title(f"FDTD |Ey| - {TIDY3D_TASK_NAME}")

PORT_POSITIONS_UM = np.array(
    [-PORT_POSITION_UM, 0.0, +PORT_POSITION_UM]
)

PORT_PITCH_UM = PORT_POSITION_UM
GAP_UM = PORT_PITCH_UM - TAPER_WIDTH_UM

print("\n--- 3x3 PORT GEOMETRY ---")
print(
    "Port positions [um]: "
    f"bottom={PORT_POSITIONS_UM[0]:+.4f}, "
    f"center={PORT_POSITIONS_UM[1]:+.4f}, "
    f"top={PORT_POSITIONS_UM[2]:+.4f}"
)
print(f"Port pitch: {PORT_PITCH_UM:.4f} um")
print(f"Taper width: {TAPER_WIDTH_UM:.4f} um")
print(f"Gap between adjacent tapers: {GAP_UM:.4f} um")

plt.show()