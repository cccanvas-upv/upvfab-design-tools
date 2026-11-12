from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
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

INPUT_CONFIGURATION = "bottom"  # "top", "bottom", "hybrid"
DELTA_PHI_IN_DEG = 0.0

TIDY3D_TASK_NAME = f"mmi_3x3_Lminus2_dy0.09_wt1.9_{INPUT_CONFIGURATION}"
TIDY3D_DATA_PATH = f"data/{TIDY3D_TASK_NAME}.hdf5"

TOP_DATA_PATH = (
    "data/"
    "mmi_3x3_Lminus2_dy0.09_wt1.9_top.hdf5"
)

BOTTOM_DATA_PATH = (
    "data/"
    "mmi_3x3_Lminus2_dy0.09_wt1.9_bottom.hdf5"
)

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

def wrap_phase_deg(
    phase_deg: float | np.ndarray,
):
    """Wrap phase to [-180, 180) degrees."""

    return (
        phase_deg + 180.0
    ) % 360.0 - 180.0


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

    relative_phase_deg = np.degrees(np.angle(amps * np.conj(amps[1])))

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

# ============================================================
# PHASE CHARACTERIZATION - 120 DEGREE HYBRID
# ============================================================

top_path = Path(TOP_DATA_PATH)
bottom_path = Path(BOTTOM_DATA_PATH)


if top_path.exists() and bottom_path.exists():

    sim_data_top = load_tidy3d_simulation_data(
        TOP_DATA_PATH
    )

    sim_data_bottom = load_tidy3d_simulation_data(
        BOTTOM_DATA_PATH
    )


    names = (
        "bottom",
        "center",
        "top",
    )


    # ========================================================
    # COMPLEX AMPLITUDES - TOP INPUT
    # ========================================================

    amps_top = np.array([
        complex(
            sim_data_top[f"mode_{name}"].amps
            .sel(
                direction="+",
                mode_index=0,
            )
            .isel(f=0)
            .values.squeeze()
        )
        for name in names
    ])


    # ========================================================
    # COMPLEX AMPLITUDES - BOTTOM INPUT
    # ========================================================

    amps_bottom = np.array([
        complex(
            sim_data_bottom[f"mode_{name}"].amps
            .sel(
                direction="+",
                mode_index=0,
            )
            .isel(f=0)
            .values.squeeze()
        )
        for name in names
    ])


    # ========================================================
    # CHARACTERISTIC PHASE OF EACH OUTPUT
    # ========================================================
    #
    # phi_k =
    # arg(
    #     H_k,top
    #     * conj(H_k,bottom)
    # )
    #
    # Order:
    #
    # [bottom, center, top]

    phi_deg = np.degrees(
        np.angle(
            amps_top
            * np.conj(amps_bottom)
        )
    )


    phi_bottom_deg = phi_deg[0]
    phi_center_deg = phi_deg[1]
    phi_top_deg = phi_deg[2]


    # ========================================================
    # PHASES RELATIVE TO CENTER
    # ========================================================

    phi_rel_center_deg = wrap_phase_deg(
        phi_deg
        - phi_center_deg
    )


    phase_bottom_center_deg = (
        phi_rel_center_deg[0]
    )

    phase_center_center_deg = 0.0

    phase_top_center_deg = (
        phi_rel_center_deg[2]
    )


    # ========================================================
    # PAIRWISE PHASE RELATIONS
    # ========================================================

    phase_bottom_center_deg = wrap_phase_deg(
        phi_bottom_deg
        - phi_center_deg
    )

    phase_center_top_deg = wrap_phase_deg(
        phi_center_deg
        - phi_top_deg
    )

    phase_top_bottom_deg = wrap_phase_deg(
        phi_top_deg
        - phi_bottom_deg
    )


    # ========================================================
    # PHASE ERRORS
    # ========================================================
    #
    # In an ideal 120-degree hybrid, every pair of
    # characteristic phases is separated by 120 deg
    # on the phase circle.
    #
    # Therefore the sign can be +120 or -120.

    error_bottom_center_deg = (
        abs(
            abs(phase_bottom_center_deg)
            - 120.0
        )
    )

    error_center_top_deg = (
        abs(
            abs(phase_center_top_deg)
            - 120.0
        )
    )

    error_top_bottom_deg = (
        abs(
            abs(phase_top_bottom_deg)
            - 120.0
        )
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 65
    )

    print(
        "FDTD MMI 3x3 - 120-DEGREE "
        "PHASE CHARACTERIZATION"
    )

    print(
        "=" * 65
    )


    print(
        "\n--- CHARACTERISTIC PHASES ---"
    )

    print(
        f"phi_bottom = "
        f"{phi_bottom_deg:+.4f} deg"
    )

    print(
        f"phi_center = "
        f"{phi_center_deg:+.4f} deg"
    )

    print(
        f"phi_top    = "
        f"{phi_top_deg:+.4f} deg"
    )


    print(
        "\n--- PHASES RELATIVE TO CENTER ---"
    )

    print(
        f"bottom - center = "
        f"{phase_bottom_center_deg:+.4f} deg"
    )

    print(
        f"center - center = "
        f"{phase_center_center_deg:+.4f} deg"
    )

    print(
        f"top - center    = "
        f"{phase_top_center_deg:+.4f} deg"
    )


    print(
        "\nIdeal relative phases:"
    )

    print(
        "0 deg, +120 deg, -120 deg "
        "(up to ordering/sign convention)"
    )


    print(
        "\n--- PAIRWISE PHASE RELATIONS ---"
    )

    print(
        f"bottom - center = "
        f"{phase_bottom_center_deg:+.4f} deg"
    )

    print(
        f"center - top = "
        f"{phase_center_top_deg:+.4f} deg"
    )

    print(
        f"top - bottom = "
        f"{phase_top_bottom_deg:+.4f} deg"
    )


    print(
        "\n--- PHASE ERRORS ---"
    )

    print(
        f"bottom-center error = "
        f"{error_bottom_center_deg:.4f} deg"
    )

    print(
        f"center-top error = "
        f"{error_center_top_deg:.4f} deg"
    )

    print(
        f"top-bottom error = "
        f"{error_top_bottom_deg:.4f} deg"
    )


else:

    print(
        "\n--- 120-DEGREE PHASE CHARACTERIZATION ---"
    )

    if not top_path.exists():

        print(
            f"Missing top simulation: "
            f"{TOP_DATA_PATH}"
        )

    if not bottom_path.exists():

        print(
            f"Missing bottom simulation: "
            f"{BOTTOM_DATA_PATH}"
        )



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