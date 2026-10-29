from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.waveguides import strip_waveguide

from upvfab_design_tools.tidy3d_plugin import (
    build_mmi_2x2_asymmetric_fdtd_simulation,
    estimate_tidy3d_cost,
    load_tidy3d_simulation_data,
    mmi_2x2_asymmetric_vertices,
    plot_fdtd_field_xy,
    test_tidy3d_api,
)


RUN_TIDY3D_CLOUD = True
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False

TIDY3D_TASK_NAME = "mmi72_L02minus1.8_dyminus0.05_wt2.8"
TIDY3D_DATA_PATH = f"data/{TIDY3D_TASK_NAME}.hdf5"


# ============================================================
# DISPOSITIVO
# ============================================================

WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.30

MMI_WIDTH_UM = 10.0

# Sustituir por el valor optimizado del EME
MMI_LENGTH_UM = 88.7813 - 1.8


ACCESS_WIDTH_UM = 1
TAPER_WIDTH_UM = 2.8

STRAIGHT_IO_LENGTH_UM = 2.0
TAPER_LENGTH_UM = 50.0
IO_LENGTH_UM = STRAIGHT_IO_LENGTH_UM + TAPER_LENGTH_UM

DY_UM = -0.05


# ============================================================
# POSICIONES DESDE LOS BORDES
# ============================================================

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


# ============================================================
# INFORMACION GEOMETRICA
# ============================================================

input_gap = (
    INPUT_Y_POSITIONS_UM[1]
    - INPUT_Y_POSITIONS_UM[0]
    - TAPER_WIDTH_UM
)

output_gap = (
    OUTPUT_Y_POSITIONS_UM[1]
    - OUTPUT_Y_POSITIONS_UM[0]
    - TAPER_WIDTH_UM
)

edge_clearance = (
    MMI_WIDTH_UM / 2
    - max(abs(y) for y in INPUT_Y_POSITIONS_UM + OUTPUT_Y_POSITIONS_UM)
    - TAPER_WIDTH_UM / 2
)

print("\n--- 72:28 GEOMETRY ---")
print(f"Input positions:  {INPUT_Y_POSITIONS_UM} um")
print(f"Output positions: {OUTPUT_Y_POSITIONS_UM} um")
print(f"Taper width: {TAPER_WIDTH_UM:.4f} um")
print(f"Input gap: {input_gap:.4f} um")
print(f"Output gap: {output_gap:.4f} um")
print(f"Minimum edge clearance: {edge_clearance:.4f} um")


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
    x_span=MMI_WIDTH_UM + 4.0,
    bottom_margin=1.5,
    top_margin=1.5,
    name="sin_fdtd_stack",
)


# ============================================================
# GEOMETRIA
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
# FDTD
# ============================================================

sim = build_mmi_2x2_asymmetric_fdtd_simulation(
    polygons=polygons,
    cross_section=cross_section,
    wavelength_um=WAVELENGTH_UM,
    input_y_positions_um=INPUT_Y_POSITIONS_UM,
    output_y_positions_um=OUTPUT_Y_POSITIONS_UM,
    access_width_um=ACCESS_WIDTH_UM,
    input_port=INPUT_PORT,
    bandwidth_um=0.1,
    pad_x_um=2.0,
    pad_y_um=2.0,
    pad_z_um=0.0,
    source_offset_um=STRAIGHT_IO_LENGTH_UM / 2,
    monitor_offset_um=STRAIGHT_IO_LENGTH_UM / 2,
    mode_plane_y_span_um=2.5 * ACCESS_WIDTH_UM,
    min_steps_per_wvl=16,
    run_time_s=5e-12,
)

sim.validate_pre_upload()


# ============================================================
# COMPROBAR GEOMETRIA
# ============================================================

fig, ax = plt.subplots(figsize=(12, 4))
sim.plot(z=CORE_HEIGHT_UM / 2, ax=ax)
ax.set_aspect("equal", adjustable="box")


# ============================================================
# COSTE
# ============================================================

if ESTIMATE_TIDY3D_COST:
    test_tidy3d_api()

    cost = estimate_tidy3d_cost(
        sim,
        task_name=f"{TIDY3D_TASK_NAME}_cost",
        verbose=True,
    )

    print(f"\nEstimated cost: {cost:.6g} FlexCredits")


# ============================================================
# LOAD / RUN
# ============================================================

sim_data = None

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


# ============================================================
# RESULTADOS
# ============================================================

if sim_data is not None:

    T_bottom = float(
        np.real(
            sim_data["flux_bottom"].flux.values.squeeze()
        )
    )

    T_top = float(
        np.real(
            sim_data["flux_top"].flux.values.squeeze()
        )
    )

    T_total = T_bottom + T_top

    K_bottom = T_bottom / T_total
    K_top = T_top / T_total

    a_bottom = complex(
        sim_data["mode_bottom"].amps
        .sel(direction="+", mode_index=0)
        .isel(f=0)
        .values.squeeze()
    )

    a_top = complex(
        sim_data["mode_top"].amps
        .sel(direction="+", mode_index=0)
        .isel(f=0)
        .values.squeeze()
    )

    phase_difference_deg = np.degrees(
        np.angle(a_top * np.conj(a_bottom))
    )

    print("\n" + "=" * 55)
    print("FDTD MMI 2x2 - 72:28")
    print("=" * 55)

    print(f"Input port: {INPUT_PORT}")
    print(f"T bottom: {T_bottom:.6f}")
    print(f"T top:    {T_top:.6f}")
    print(f"T total:  {T_total:.6f}")

    print(
        f"Splitting ratio: "
        f"{100*K_bottom:.2f}:{100*K_top:.2f}"
    )

    print(
        f"Excess loss: "
        f"{-10*np.log10(T_total):.4f} dB"
    )

    print(
        f"Phase top-bottom: "
        f"{phase_difference_deg:+.3f} deg"
    )

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

plt.show()