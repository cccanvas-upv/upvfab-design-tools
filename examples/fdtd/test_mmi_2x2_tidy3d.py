from __future__ import annotations

import matplotlib.pyplot as plt

from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle
from upvfab_design_tools.core.materials import (
    AIR,
    BCB,
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
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

ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = True
RUN_TIDY3D_CLOUD = False
TIDY3D_DATA_PATH = "data/upvfab_mmi_2x2_fdtd.hdf5"

WAVELENGTH_UM = 1.55
MMI_LENGTH_UM = 105
IO_LENGTH_UM = 10.0
ACCESS_WIDTH_UM = 1.0
TAPER_WIDTH_UM = 2.0
MMI_WIDTH_UM = 12.0
IO_Y_POSITION_UM = 2.0
INPUT_PORT = "bottom"
MODE_PLANE_Y_SPAN_UM = 2.5 * ACCESS_WIDTH_UM

CORE_HEIGHT_UM = 0.3
DOMAIN_X_SPAN_UM = 18.0
BOTTOM_MARGIN_UM = 2.0
TOP_MARGIN_UM = 2.0

# To switch back to the simple strip cross-section, uncomment this import:
# from upvfab_design_tools.core.waveguides import strip_waveguide
#
# cross_section = strip_waveguide(
#     width=ACCESS_WIDTH_UM,
#     height=CORE_HEIGHT_UM,
#     core_material=SILICON_NITRIDE,
#     background_material=THERMAL_SILICON_DIOXIDE,
#     x_span=DOMAIN_X_SPAN_UM,
#     bottom_margin=BOTTOM_MARGIN_UM,
#     top_margin=TOP_MARGIN_UM,
#     name="sin_strip_fdtd_stack",
# )

DOMAIN_Z_MIN_UM = -2.0
DOMAIN_Z_MAX_UM = 2.0

bcb_layer = Rectangle(
    x_min=-0.5 * DOMAIN_X_SPAN_UM,
    x_max=0.5 * DOMAIN_X_SPAN_UM,
    z_min=0,
    z_max=0.5,
    material=BCB,
    name="BCB",
)
air_layer = Rectangle(
    x_min=-0.5 * DOMAIN_X_SPAN_UM,
    x_max=0.5 * DOMAIN_X_SPAN_UM,
    z_min=0.5,
    z_max=DOMAIN_Z_MAX_UM,
    material=AIR,
    name="Air",
)
core_layer = Rectangle(
    x_min=-0.5 * ACCESS_WIDTH_UM,
    x_max=0.5 * ACCESS_WIDTH_UM,
    z_min=0.0,
    z_max=CORE_HEIGHT_UM,
    material=SILICON_NITRIDE,
    name="core",
)

cross_section = CrossSection(
    name="sin_bcb_air_fdtd_stack",
    background_material=THERMAL_SILICON_DIOXIDE,
    x_min=-0.5 * DOMAIN_X_SPAN_UM,
    x_max=0.5 * DOMAIN_X_SPAN_UM,
    z_min=DOMAIN_Z_MIN_UM,
    z_max=DOMAIN_Z_MAX_UM,
    structures=(bcb_layer, air_layer, core_layer),
)

core, surrounding = extract_fdtd_layers(cross_section)
print(
    "FDTD stack core:",
    core.name,
    f"z=[{core.z_min:.3f}, {core.z_max:.3f}] um",
    core.material_name,
)
print("FDTD surrounding layers:", [layer.name for layer in surrounding])

polygons = mmi_2x2_vertices(
    length_um=MMI_LENGTH_UM,
    io_length_um=IO_LENGTH_UM,
    io_y_position_um=IO_Y_POSITION_UM,
    access_width_um=ACCESS_WIDTH_UM,
    taper_width_um=TAPER_WIDTH_UM,
    mmi_width_input_um=MMI_WIDTH_UM,
    mmi_width_center_um=MMI_WIDTH_UM,
)

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

sim.validate_pre_upload()

fig_vertices, ax_vertices = plot_mmi_2x2_vertices(polygons)
ax_vertices.set_title("MMI 2x2 polygons used for Tidy3D FDTD")

fig_sim, ax_sim = plt.subplots(figsize=(10, 3))
sim.plot(z=0.15, ax=ax_sim)
ax_sim.set_aspect("equal", adjustable="box")
ax_sim.set_title("Tidy3D FDTD geometry, sources, and monitors")

fig_eps, ax_eps = plt.subplots(figsize=(10, 3))
sim.plot_eps(z=0.15, ax=ax_eps)
ax_eps.set_aspect("equal", adjustable="box")
ax_eps.set_title("Tidy3D FDTD permittivity at core mid-plane")

try:
    plt.show()
except Exception:
    fig_vertices.savefig("mmi_2x2_fdtd_vertices.png", dpi=300, bbox_inches="tight")
    fig_sim.savefig("mmi_2x2_fdtd_simulation.png", dpi=300, bbox_inches="tight")
    fig_eps.savefig("mmi_2x2_fdtd_epsilon.png", dpi=300, bbox_inches="tight")

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
        task_name="upvfab_mmi_2x2_fdtd_cost",
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
        task_name="upvfab_mmi_2x2_fdtd",
        path=TIDY3D_DATA_PATH,
        verbose=True,
    )

if sim_data is not None:
    print_mmi_2x2_fluxes(sim_data)

    fig_field, ax_field = plot_fdtd_field_xy(
        sim_data,
        monitor_name="field_core",
        field_component="Ey",
        field_part="mag",
        polygons=polygons,
    )
    ax_field.set_title("FDTD Re(Ey), propagation left to right")

    try:
        plt.show()
    except Exception:
        fig_field.savefig("mmi_2x2_fdtd_Ey.png", dpi=300, bbox_inches="tight")
