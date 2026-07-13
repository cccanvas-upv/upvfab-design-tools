"""Build and inspect a Tidy3D FDTD simulation for a linear waveguide taper.

The taper is defined from an input cross-section, an output cross-section, and
a taper length. The FDTD helper builds a centered linear taper with short
straight input/output sections so the source and output monitor sit on uniform
waveguides.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.waveguides import strip_waveguide
from upvfab_design_tools.tidy3d_plugin import (
    build_taper_fdtd_simulation,
    estimate_tidy3d_cost,
    load_tidy3d_simulation_data,
    plot_fdtd_field_xy,
    plot_fdtd_polygons,
    print_taper_transmission,
    test_tidy3d_api,
)

ESTIMATE_TIDY3D_COST = True
LOAD_TIDY3D_DATA = False
RUN_TIDY3D_CLOUD = True
TIDY3D_DATA_PATH = "data/upvfab_taper_fdtd.hdf5"

WAVELENGTH_UM = 1.55
CORE_HEIGHT_UM = 0.3
INPUT_WIDTH_UM = 1.0
OUTPUT_WIDTH_UM = 8.0
TAPER_LENGTH_UM = 50.0
INPUT_STRAIGHT_LENGTH_UM = 5.0
OUTPUT_STRAIGHT_LENGTH_UM = 5.0
DOMAIN_X_SPAN_UM = 10.0
BOTTOM_MARGIN_UM = 2.0
TOP_MARGIN_UM = 2.0
MODE_PLANE_Y_SPAN_UM = 2.0 * max(INPUT_WIDTH_UM, OUTPUT_WIDTH_UM)
RUN_TIME_S = 3e-12

input_cross_section = strip_waveguide(
    width=INPUT_WIDTH_UM,
    height=CORE_HEIGHT_UM,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=BOTTOM_MARGIN_UM,
    top_margin=TOP_MARGIN_UM,
    name="taper_input_strip",
)

output_cross_section = strip_waveguide(
    width=OUTPUT_WIDTH_UM,
    height=CORE_HEIGHT_UM,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=BOTTOM_MARGIN_UM,
    top_margin=TOP_MARGIN_UM,
    name="taper_output_strip",
)

sim, polygons = build_taper_fdtd_simulation(
    input_cross_section=input_cross_section,
    output_cross_section=output_cross_section,
    length_um=TAPER_LENGTH_UM,
    wavelength_um=WAVELENGTH_UM,
    input_straight_length_um=INPUT_STRAIGHT_LENGTH_UM,
    output_straight_length_um=OUTPUT_STRAIGHT_LENGTH_UM,
    bandwidth_um=0.1,
    pad_x_um=2.0,
    pad_y_um=2.0,
    pad_z_um=0.0,
    source_offset_um=1.0,
    monitor_offset_um=1.0,
    mode_plane_y_span_um=MODE_PLANE_Y_SPAN_UM,
    min_steps_per_wvl=12,
    run_time_s=RUN_TIME_S,
)

sim.validate_pre_upload()

fig_vertices, ax_vertices = plot_fdtd_polygons(polygons)
ax_vertices.set_title("Linear taper polygons used for Tidy3D FDTD")

fig_sim, ax_sim = plt.subplots(figsize=(10, 3))
sim.plot(z=0.15, ax=ax_sim)
ax_sim.set_aspect("equal", adjustable="box")
ax_sim.set_title("Tidy3D taper FDTD geometry, source, and monitors")

fig_eps, ax_eps = plt.subplots(figsize=(10, 3))
sim.plot_eps(z=0.15, ax=ax_eps)
ax_eps.set_aspect("equal", adjustable="box")
ax_eps.set_title("Tidy3D taper FDTD permittivity at core mid-plane")

try:
    plt.show()
except Exception:
    fig_vertices.savefig("taper_fdtd_vertices.png", dpi=300, bbox_inches="tight")
    fig_sim.savefig("taper_fdtd_simulation.png", dpi=300, bbox_inches="tight")
    fig_eps.savefig("taper_fdtd_epsilon.png", dpi=300, bbox_inches="tight")

print(f"Taper: {INPUT_WIDTH_UM:.3f} um -> {OUTPUT_WIDTH_UM:.3f} um")
print(f"Taper length: {TAPER_LENGTH_UM:.3f} um")
print(f"Simulation center: {sim.center}")
print(f"Simulation size: {sim.size}")
print(f"Structures: {[structure.name for structure in sim.structures]}")
print(f"Sources: {[source.name for source in sim.sources]}")
print(f"Source center: {sim.sources[0].center}")
print(f"Source size: {sim.sources[0].size}")
print(f"Monitors: {[monitor.name for monitor in sim.monitors]}")

if ESTIMATE_TIDY3D_COST:
    test_tidy3d_api()
    estimated_cost = estimate_tidy3d_cost(
        sim,
        task_name="upvfab_taper_fdtd_cost",
        verbose=True,
    )
    print(f"Estimated Tidy3D cost: {estimated_cost:.6g} FlexCredits")

sim_data = None

if LOAD_TIDY3D_DATA:
    sim_data = load_tidy3d_simulation_data(TIDY3D_DATA_PATH)

if RUN_TIDY3D_CLOUD:
    import tidy3d.web as web

    test_tidy3d_api()
    sim_data = web.run(
        sim,
        task_name="upvfab_taper_fdtd",
        path=TIDY3D_DATA_PATH,
        verbose=True,
    )

if sim_data is not None:
    print_taper_transmission(sim_data)

    fig_field, ax_field = plot_fdtd_field_xy(
        sim_data,
        monitor_name="field_core",
        field_component="Ey",
        field_part="mag",
        polygons=polygons,
    )
    ax_field.set_title("FDTD |Ey| through linear taper, propagation left to right")

    try:
        plt.show()
    except Exception:
        fig_field.savefig("taper_fdtd_Ey.png", dpi=300, bbox_inches="tight")
