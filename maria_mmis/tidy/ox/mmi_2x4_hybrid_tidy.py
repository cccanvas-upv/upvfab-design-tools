from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

TIDY3D_TASK_BASE = "mmi_2x4_Lminus4.8_dy0_wt3_mins16"

TIDY3D_TASK_NAME = f"{TIDY3D_TASK_BASE}_{INPUT_CONFIGURATION}"
TIDY3D_DATA_PATH = f"data/{TIDY3D_TASK_NAME}.hdf5"

TOP_DATA_PATH = f"data/{TIDY3D_TASK_BASE}_top.hdf5"
BOTTOM_DATA_PATH = f"data/{TIDY3D_TASK_BASE}_bottom.hdf5"
TIDY3D_DATA_PATH = f"data/{TIDY3D_TASK_NAME}.hdf5"


WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.30
MMI_WIDTH_UM = 16.0

MMI_LENGTH_UM = 281.350472 -4.8

ACCESS_WIDTH_UM = 1
TAPER_WIDTH_UM = 3
TAPER_LENGTH_UM = 20.0

STRAIGHT_IO_LENGTH_UM = 2.0
IO_LENGTH_UM = TAPER_LENGTH_UM + STRAIGHT_IO_LENGTH_UM

DY = 0.0


# ============================================================
# POSICIONES DE LOS PUERTOS
#
# Convencion:
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
    bandwidth_um=0.1,
    pad_x_um=2.0,
    pad_y_um=2.0,
    pad_z_um=0.0,
    source_offset_um=SOURCE_OFFSET_UM,
    monitor_offset_um=MONITOR_OFFSET_UM,
    mode_plane_y_span_um=MODE_PLANE_Y_SPAN_UM,
    min_steps_per_wvl=18,
    run_time_s=5e-12,
)

sim.validate_pre_upload()


fig, ax = plt.subplots(figsize=(12, 4))
sim.plot(z=0.5 * CORE_HEIGHT_UM, ax=ax)
ax.set_aspect("equal", adjustable="box")
ax.set_title(f"MMI 2x4 - {INPUT_CONFIGURATION}")


fig, ax = plt.subplots(figsize=(12, 4))
sim.plot_eps(z=0.5 * CORE_HEIGHT_UM, ax=ax)
ax.set_aspect("equal", adjustable="box")
ax.set_title("MMI 2x4 - permittivity")


if ESTIMATE_TIDY3D_COST:
    test_tidy3d_api()

    cost = estimate_tidy3d_cost(
        sim,
        task_name=f"{TIDY3D_TASK_NAME}_cost",
        verbose=True,
    )

    print(
        f"\nEstimated cost: "
        f"{cost:.6g} FlexCredits"
    )


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


# FUNCION AUXILIAR

def wrap_phase_deg(
    phase_deg: float | np.ndarray,
):
    """Wrap phase to [-180, 180) degrees."""

    return (phase_deg + 180.0) % 360.0 - 180.0



# POSTPROCESADO

if sim_data is not None:

    names = ("p1", "p2", "p3", "p4")



    flux = np.array([
        float( np.real(sim_data[ f"flux_{name}"].flux.values.squeeze()))
        for name in names
    ])


    total = np.sum(flux)

    split = (flux / total
        if total > 0
        else np.zeros(4)
    )

    excess_loss_db = (-10.0 * np.log10(total)
        if total > 0
        else np.inf
    )


    amps = np.array([
        complex(sim_data[f"mode_{name}"].amps.sel(direction="+", mode_index=0 ).isel(f=0).values.squeeze())
        for name in names])

        # ========================================================
    # POTENCIA MODAL TE0
    # ========================================================

    mode_power = np.abs(amps) ** 2

    total_mode_power = np.sum(
        mode_power
    )

    # ========================================================
    # REFLECTION TE0
    # ========================================================

    if INPUT_CONFIGURATION == "top":

        reflection_amp = complex(
            sim_data["mode_reflection_top"].amps
            .sel(direction="-", mode_index=0)
            .isel(f=0)
            .values.squeeze()
        )

        reflection_power = np.abs(
            reflection_amp
        ) ** 2


    elif INPUT_CONFIGURATION == "bottom":

        reflection_amp = complex(
            sim_data["mode_reflection_bottom"].amps
            .sel(direction="-", mode_index=0)
            .isel(f=0)
            .values.squeeze()
        )

        reflection_power = np.abs(
            reflection_amp
        ) ** 2


    else:

        reflection_power = np.nan


    if np.isfinite(reflection_power):

        unaccounted_power = (
            1.0
            - total_mode_power
            - reflection_power
        )

    else:

        unaccounted_power = np.nan

    split_mode = (
        mode_power / total_mode_power
        if total_mode_power > 0
        else np.zeros(4)
    )

    excess_loss_mode_db = (
        -10.0 * np.log10(total_mode_power)
        if total_mode_power > 0
        else np.inf
    )

    output_phase_deg = np.degrees(np.angle(amps))


    # Fases respecto a P3
    #
    # names:
    #
    # 0 -> P1
    # 1 -> P2
    # 2 -> P3
    # 3 -> P4

    relative_phase_p3_deg = np.degrees(np.angle(amps* np.conj(amps[2])))
    relative_phase_p3_360_deg = (relative_phase_p3_deg % 360.0)
    relative_phase_p3_360_deg = np.where(
    np.isclose(
        relative_phase_p3_360_deg,
        360.0,
        atol=1e-9,
    ),
    0.0,
    relative_phase_p3_360_deg,
    )


    print(
        "\n"
        + "=" * 65
    )

    print("FDTD MMI 2x4 90-DEGREE HYBRID RESULTS")

    print(
        "=" * 65
    )

    print(
        f"Input configuration: "
        f"{INPUT_CONFIGURATION}"
    )

    if INPUT_CONFIGURATION == "hybrid":
        print(
            f"Delta phi input: "
            f"{DELTA_PHI_IN_DEG:+.2f} deg"
        )

    print(
    f"Flux total output: "
    f"{total:.6f}"
    )

    print(
        f"Flux excess loss: "
        f"{excess_loss_db:.4f} dB"
    )

    print(
        f"TE0 total output: "
        f"{total_mode_power:.6f}"
    )

    print(
        f"TE0 excess loss: "
        f"{excess_loss_mode_db:.4f} dB"
    )

    print(
        f"TE0 reflection: "
        f"{reflection_power:.6f}"
    )

    print(
        f"Unaccounted power: "
        f"{unaccounted_power:.6f}"
    )

    # OUTPUT DETAILS


    print(
        "\n--- OUTPUT DETAILS ---"
    )

    for name, T, K in zip(
        names,
        mode_power,
        split_mode,
        strict=True,
    ):
        print(
            f"{name.upper():>2s}: "
            f"T_TE0={T:.6f}, "
            f"split={100*K:.2f} %"
        )


    print(
        "\n--- OUTPUT PHASES RELATIVE TO P3 ---"
    )

    print(
        f"P1 relative to P3 = "
        f"{relative_phase_p3_360_deg[0]:.4f} deg"
    )

    print(
        f"P2 relative to P3 = "
        f"{relative_phase_p3_360_deg[1]:.4f} deg"
    )

    print(
        f"P3 relative to P3 = "
        f"{relative_phase_p3_360_deg[2]:.4f} deg"
    )

    print(
        f"P4 relative to P3 = "
        f"{relative_phase_p3_360_deg[3]:.4f} deg"
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
# ============================================================
# FASES DEL HIBRIDO - YU ET AL.
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
        "p1",
        "p2",
        "p3",
        "p4",
    )


    # ========================================================
    # AMPLITUDES COMPLEJAS PARA I1 = TOP
    # ========================================================

    amps_top = np.array([
        complex(
            sim_data_top[f"mode_{name}"].amps
            .sel(direction="+", mode_index=0)
            .isel(f=0)
            .values.squeeze()
        )
        for name in names
    ])


    # ========================================================
    # AMPLITUDES COMPLEJAS PARA I2 = BOTTOM
    # ========================================================

    amps_bottom = np.array([
        complex(
            sim_data_bottom[f"mode_{name}"].amps
            .sel(direction="+", mode_index=0)
            .isel(f=0)
            .values.squeeze()
        )
        for name in names
    ])


    # ========================================================
    # FASE CARACTERISTICA DE CADA OUTPUT
    # ========================================================

    # top    = I1
    # bottom = I2
    #
    # phi_k = arg(H_k,I2 * conj(H_k,I1))
    #
    # Orden:
    #
    # [P1, P2, P3, P4]

    transfer_phase_deg = np.degrees(
        np.angle(
            amps_top
            * np.conj(amps_bottom)
        )
    )

    phi_p1_deg = transfer_phase_deg[0]
    phi_p2_deg = transfer_phase_deg[1]
    phi_p3_deg = transfer_phase_deg[2]
    phi_p4_deg = transfer_phase_deg[3]


    # ========================================================
    # FASES RESPECTO A P3
    # ========================================================

    phase_p1_p3_deg = (
        phi_p1_deg
        - phi_p3_deg
    ) % 360.0

    phase_p2_p3_deg = (
        phi_p2_deg
        - phi_p3_deg
    ) % 360.0

    phase_p3_p3_deg = 0.0

    phase_p4_p3_deg = (
        phi_p4_deg
        - phi_p3_deg
    ) % 360.0


    # ========================================================
    # RELACIONES DE FASE DE YU
    # ========================================================

    phase_13_deg = wrap_phase_deg(
        phi_p1_deg
        - phi_p3_deg
    )

    phase_23_deg = wrap_phase_deg(
        phi_p2_deg
        - phi_p3_deg
    )

    phase_34_deg = wrap_phase_deg(
        phi_p3_deg
        - phi_p4_deg
    )


    # ========================================================
    # PHASE DEVIATIONS
    # ========================================================

    delta_phi_13_deg = wrap_phase_deg(
        phase_13_deg
        - 90.0
    )

    delta_phi_23_deg = wrap_phase_deg(
        phase_23_deg
        - 180.0
    )

    delta_phi_34_deg = wrap_phase_deg(
        phase_34_deg
        - 90.0
    )


    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 65
    )

    print(
        "FDTD MMI 2x4 - YU PHASE CHARACTERIZATION"
    )

    print(
        "=" * 65
    )


    print(
        "\n--- PHASES RELATIVE TO P3 ---"
    )

    print(
        f"P1 relative to P3 = "
        f"{phase_p1_p3_deg:.4f} deg"
    )

    print(
        f"P2 relative to P3 = "
        f"{phase_p2_p3_deg:.4f} deg"
    )

    print(
        f"P3 relative to P3 = "
        f"{phase_p3_p3_deg:.4f} deg"
    )

    print(
        f"P4 relative to P3 = "
        f"{phase_p4_p3_deg:.4f} deg"
    )


    print(
        "\nIdeal:"
    )

    print(
        "P1 = 90 deg"
    )

    print(
        "P2 = 180 deg"
    )

    print(
        "P3 = 0 deg"
    )

    print(
        "P4 = 270 deg"
    )


    print(
        "\n--- PHASE RELATIONS ---"
    )

    print(
        f"phi1 - phi3 = "
        f"{phase_13_deg:+.4f} deg "
        f"(ideal: +90 deg)"
    )

    print(
        f"phi2 - phi3 = "
        f"{phase_23_deg:+.4f} deg "
        f"(ideal: +/-180 deg)"
    )

    print(
        f"phi3 - phi4 = "
        f"{phase_34_deg:+.4f} deg "
        f"(ideal: +90 deg)"
    )


    print(
        "\n--- PHASE DEVIATIONS ---"
    )

    print(
        f"Delta phi13 = "
        f"{delta_phi_13_deg:+.4f} deg"
    )

    print(
        f"Delta phi23 = "
        f"{delta_phi_23_deg:+.4f} deg"
    )

    print(
        f"Delta phi34 = "
        f"{delta_phi_34_deg:+.4f} deg"
    )


else:

    print(
        "\n--- YU PHASE CHARACTERIZATION ---"
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

I1_UM = INPUT_Y_POSITIONS_UM[0]
I2_UM = INPUT_Y_POSITIONS_UM[1]

P1_UM = OUTPUT_Y_POSITIONS_UM[0]
P2_UM = OUTPUT_Y_POSITIONS_UM[1]
P3_UM = OUTPUT_Y_POSITIONS_UM[2]
P4_UM = OUTPUT_Y_POSITIONS_UM[3]

OUTPUT_PITCH_UM = abs(
    P1_UM
    - P2_UM
)

OUTPUT_GAP_UM = (
    OUTPUT_PITCH_UM
    - TAPER_WIDTH_UM
)

EDGE_MARGIN_UM = (
    0.5 * MMI_WIDTH_UM
    - abs(P1_UM)
    - 0.5 * TAPER_WIDTH_UM
)


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