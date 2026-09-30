from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle
from upvfab_design_tools.core.materials import Material

from .conversion import material_to_tidy3d_medium


InputPort = Literal["top", "bottom"]
FDTDFieldComponent = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
FDTDFieldPart = Literal["real", "mag", "phase", "intensity"]


@dataclass(frozen=True)
class FDTDLayer:
    """A vertical material layer extracted from a cross-section for FDTD."""

    name: str
    z_min: float
    z_max: float
    material_name: str
    material: Material
    priority: int
    is_core: bool = False

    @property
    def thickness(self) -> float:
        return self.z_max - self.z_min

    @property
    def z_center(self) -> float:
        return 0.5 * (self.z_min + self.z_max)


# def mmi_2x2_vertices(
#     *,
#     length_um: float,
#     io_length_um: float,
#     io_y_position_um: float,
#     access_width_um: float,
#     taper_width_um: float,
#     mmi_width_input_um: float,
#     mmi_width_center_um: float | None = None,
# ) -> dict[str, tuple[tuple[float, float], ...]]:
#     """Return 2D MMI polygons in the FDTD propagation-lateral plane.

#     Coordinates are ``(x, y)`` where ``x`` is the propagation direction and
#     ``y`` is the lateral coordinate. This mirrors the exploratory Annex 3
#     notebook geometry while keeping the vertices available as reusable data.
#     """

#     _validate_positive("length_um", length_um)
#     _validate_positive("io_length_um", io_length_um)
#     _validate_positive("access_width_um", access_width_um)
#     _validate_positive("taper_width_um", taper_width_um)
#     _validate_positive("mmi_width_input_um", mmi_width_input_um)

#     if io_y_position_um <= 0:
#         raise ValueError("io_y_position_um must be positive.")

#     if mmi_width_center_um is None:
#         mmi_width_center_um = mmi_width_input_um

#     _validate_positive("mmi_width_center_um", mmi_width_center_um)

#     x0 = -0.5 * length_um
#     x1 = 0.5 * length_um
#     x_left = x0 - io_length_um
#     x_right = x1 + io_length_um
#     y = io_y_position_um
#     w_wg = access_width_um
#     w_taper = taper_width_um
#     w0 = mmi_width_input_um
#     w1 = mmi_width_center_um

#     return {
#         "mmi": (
#             (x0, -0.5 * w0),
#             (0.0, -0.5 * w1),
#             (x1, -0.5 * w0),
#             (x1, 0.5 * w0),
#             (0.0, 0.5 * w1),
#             (x0, 0.5 * w0),
#         ),
#         "in_top": (
#             (x_left, y - 0.5 * w_wg),
#             (x0, y - 0.5 * w_taper),
#             (x0, y + 0.5 * w_taper),
#             (x_left, y + 0.5 * w_wg),
#         ),
#         "in_bottom": (
#             (x_left, -y - 0.5 * w_wg),
#             (x0, -y - 0.5 * w_taper),
#             (x0, -y + 0.5 * w_taper),
#             (x_left, -y + 0.5 * w_wg),
#         ),
#         "out_top": (
#             (x1, y - 0.5 * w_taper),
#             (x_right, y - 0.5 * w_wg),
#             (x_right, y + 0.5 * w_wg),
#             (x1, y + 0.5 * w_taper),
#         ),
#         "out_bottom": (
#             (x1, -y - 0.5 * w_taper),
#             (x_right, -y - 0.5 * w_wg),
#             (x_right, -y + 0.5 * w_wg),
#             (x1, -y + 0.5 * w_taper),
#         ),
#     }

def mmi_2x2_vertices(
    *,
    length_um: float,
    io_length_um: float,
    io_y_position_um: float,
    access_width_um: float,
    taper_width_um: float,
    mmi_width_input_um: float,
    mmi_width_center_um: float | None = None,
    straight_io_length_um: float = 0.0,
) -> dict[str, tuple[tuple[float, float], ...]]:
    """Return 2D MMI polygons in the FDTD propagation-lateral plane.

    Coordinates are ``(x, y)`` where ``x`` is the propagation direction and
    ``y`` is the lateral coordinate.

    ``io_length_um`` is the total length of each access section:

        straight section + taper section

    ``straight_io_length_um`` specifies the straight waveguide length at the
    outer side of each access. Setting it to zero reproduces the previous
    geometry.
    """

    _validate_positive("length_um", length_um)
    _validate_positive("io_length_um", io_length_um)
    _validate_positive("access_width_um", access_width_um)
    _validate_positive("taper_width_um", taper_width_um)
    _validate_positive("mmi_width_input_um", mmi_width_input_um)

    if straight_io_length_um < 0:
        raise ValueError(
            "straight_io_length_um must be non-negative."
        )

    if straight_io_length_um >= io_length_um:
        raise ValueError(
            "straight_io_length_um must be smaller than io_length_um."
        )

    if io_y_position_um <= 0:
        raise ValueError(
            "io_y_position_um must be positive."
        )

    if mmi_width_center_um is None:
        mmi_width_center_um = mmi_width_input_um

    _validate_positive(
        "mmi_width_center_um",
        mmi_width_center_um,
    )

    # MMI limits
    x0 = -0.5 * length_um
    x1 = +0.5 * length_um

    # Total access limits
    x_left = x0 - io_length_um
    x_right = x1 + io_length_um

    # Transition between straight guide and taper
    x_taper_left = x_left + straight_io_length_um
    x_taper_right = x_right - straight_io_length_um

    y = io_y_position_um

    w_wg = access_width_um
    w_taper = taper_width_um

    w0 = mmi_width_input_um
    w1 = mmi_width_center_um

    return {
        "mmi": (
            (x0, -0.5 * w0),
            (0.0, -0.5 * w1),
            (x1, -0.5 * w0),
            (x1, +0.5 * w0),
            (0.0, +0.5 * w1),
            (x0, +0.5 * w0),
        ),

        # ====================================================
        # INPUT TOP
        # straight guide -> taper -> MMI
        # ====================================================

        "in_top": (
            (x_left, y - 0.5 * w_wg),
            (x_taper_left, y - 0.5 * w_wg),
            (x0, y - 0.5 * w_taper),
            (x0, y + 0.5 * w_taper),
            (x_taper_left, y + 0.5 * w_wg),
            (x_left, y + 0.5 * w_wg),
        ),

        # ====================================================
        # INPUT BOTTOM
        # straight guide -> taper -> MMI
        # ====================================================

        "in_bottom": (
            (x_left, -y - 0.5 * w_wg),
            (x_taper_left, -y - 0.5 * w_wg),
            (x0, -y - 0.5 * w_taper),
            (x0, -y + 0.5 * w_taper),
            (x_taper_left, -y + 0.5 * w_wg),
            (x_left, -y + 0.5 * w_wg),
        ),

        # ====================================================
        # OUTPUT TOP
        # MMI -> taper -> straight guide
        # ====================================================

        "out_top": (
            (x1, y - 0.5 * w_taper),
            (x_taper_right, y - 0.5 * w_wg),
            (x_right, y - 0.5 * w_wg),
            (x_right, y + 0.5 * w_wg),
            (x_taper_right, y + 0.5 * w_wg),
            (x1, y + 0.5 * w_taper),
        ),

        # ====================================================
        # OUTPUT BOTTOM
        # MMI -> taper -> straight guide
        # ====================================================

        "out_bottom": (
            (x1, -y - 0.5 * w_taper),
            (x_taper_right, -y - 0.5 * w_wg),
            (x_right, -y - 0.5 * w_wg),
            (x_right, -y + 0.5 * w_wg),
            (x_taper_right, -y + 0.5 * w_wg),
            (x1, -y + 0.5 * w_taper),
        ),
    }


def mmi_1x2_vertices(
    *,
    length_um: float,
    io_length_um: float,
    output_y_position_um: float,
    access_width_um: float,
    taper_width_um: float,
    mmi_width_input_um: float,
    mmi_width_center_um: float | None = None,
    straight_io_length_um: float = 0.0,
) -> dict[str, tuple[tuple[float, float], ...]]:
    """Return polygons for a 1x2 MMI.

    The input is centered at y=0.
    The two outputs are at +/- output_y_position_um.

    io_length_um = straight_io_length_um + taper_length_um
    """

    if straight_io_length_um < 0:
        raise ValueError(
            "straight_io_length_um must be non-negative."
        )

    if straight_io_length_um >= io_length_um:
        raise ValueError(
            "straight_io_length_um must be smaller than io_length_um."
        )

    if mmi_width_center_um is None:
        mmi_width_center_um = mmi_width_input_um

    _validate_positive(
        "mmi_width_center_um",
        mmi_width_center_um,
    )

    # MMI limits
    x0 = -0.5 * length_um
    x1 = +0.5 * length_um

    # Total IO limits
    x_left = x0 - io_length_um
    x_right = x1 + io_length_um

    # Straight-to-taper transitions
    x_taper_left = x_left + straight_io_length_um
    x_taper_right = x_right - straight_io_length_um

    y = output_y_position_um

    w_wg = access_width_um
    w_taper = taper_width_um

    w0 = mmi_width_input_um
    w1 = mmi_width_center_um

    return {

        # Multimode region
        "mmi": (
            (x0, -0.5 * w0),
            (0.0, -0.5 * w1),
            (x1, -0.5 * w0),
            (x1, +0.5 * w0),
            (0.0, +0.5 * w1),
            (x0, +0.5 * w0),
        ),

        # Single centered input
        "in_center": (
            (x_left, -0.5 * w_wg),
            (x_taper_left, -0.5 * w_wg),
            (x0, -0.5 * w_taper),
            (x0, +0.5 * w_taper),
            (x_taper_left, +0.5 * w_wg),
            (x_left, +0.5 * w_wg),
        ),

        # Upper output
        "out_top": (
            (x1, y - 0.5 * w_taper),
            (x_taper_right, y - 0.5 * w_wg),
            (x_right, y - 0.5 * w_wg),
            (x_right, y + 0.5 * w_wg),
            (x_taper_right, y + 0.5 * w_wg),
            (x1, y + 0.5 * w_taper),
        ),

        # Lower output
        "out_bottom": (
            (x1, -y - 0.5 * w_taper),
            (x_taper_right, -y - 0.5 * w_wg),
            (x_right, -y - 0.5 * w_wg),
            (x_right, -y + 0.5 * w_wg),
            (x_taper_right, -y + 0.5 * w_wg),
            (x1, -y + 0.5 * w_taper),
        ),
    }

def mmi_1x4_vertices(
    *,
    length_um: float,
    io_length_um: float,
    output_y_positions_um: tuple[float, float, float, float],
    access_width_um: float,
    taper_width_um: float,
    mmi_width_input_um: float,
    mmi_width_center_um: float | None = None,
    straight_io_length_um: float = 0.0,
) -> dict[str, tuple[tuple[float, float], ...]]:
    """Return 2D polygons for a centered-input 1x4 MMI.

    Coordinates are (x, y), where x is propagation
    and y is the lateral coordinate.

    The input is centered at y = 0.

    output_y_positions_um must contain the four output
    positions ordered from bottom to top.
    """

    if len(output_y_positions_um) != 4:
        raise ValueError(
            "output_y_positions_um must contain exactly 4 positions."
        )

    if straight_io_length_um < 0:
        raise ValueError(
            "straight_io_length_um must be non-negative."
        )

    if straight_io_length_um >= io_length_um:
        raise ValueError(
            "straight_io_length_um must be smaller than io_length_um."
        )

    if mmi_width_center_um is None:
        mmi_width_center_um = mmi_width_input_um

    _validate_positive(
        "mmi_width_center_um",
        mmi_width_center_um,
    )

    x0 = -0.5 * length_um
    x1 = +0.5 * length_um

    x_left = x0 - io_length_um
    x_right = x1 + io_length_um

    x_taper_left = (
        x_left + straight_io_length_um
    )

    x_taper_right = (
        x_right - straight_io_length_um
    )


    w_wg = access_width_um
    w_taper = taper_width_um

    w0 = mmi_width_input_um
    w1 = mmi_width_center_um

    # Output positions, bottom -> top
    y1, y2, y3, y4 = output_y_positions_um


    polygons = {

        "mmi": (
            (x0, -0.5 * w0),
            (0.0, -0.5 * w1),
            (x1, -0.5 * w0),
            (x1, +0.5 * w0),
            (0.0, +0.5 * w1),
            (x0, +0.5 * w0),
        ),

        # Single centered input
        "in_center": (
            (x_left, -0.5 * w_wg),
            (x_taper_left, -0.5 * w_wg),
            (x0, -0.5 * w_taper),
            (x0, +0.5 * w_taper),
            (x_taper_left, +0.5 * w_wg),
            (x_left, +0.5 * w_wg),
        ),
    }


    for index, y in enumerate(
        (y1, y2, y3, y4),
        start=1,
    ):

        polygons[f"out_{index}"] = (
            (x1, y - 0.5 * w_taper),
            (x_taper_right, y - 0.5 * w_wg),
            (x_right, y - 0.5 * w_wg),
            (x_right, y + 0.5 * w_wg),
            (x_taper_right, y + 0.5 * w_wg),
            (x1, y + 0.5 * w_taper),
        )

    return polygons



def taper_vertices(
    *,
    length_um: float,
    input_width_um: float,
    output_width_um: float,
    input_straight_length_um: float = 5.0,
    output_straight_length_um: float = 5.0,
) -> dict[str, tuple[tuple[float, float], ...]]:
    """Return taper polygons in the FDTD propagation-lateral plane.

    Coordinates are ``(x, y)`` where ``x`` is the propagation direction and
    ``y`` is the lateral coordinate. The returned layout includes short
    straight input/output sections so mode source and monitor planes can be
    placed in uniform waveguides.
    """

    _validate_positive("length_um", length_um)
    _validate_positive("input_width_um", input_width_um)
    _validate_positive("output_width_um", output_width_um)
    _validate_nonnegative("input_straight_length_um", input_straight_length_um)
    _validate_nonnegative("output_straight_length_um", output_straight_length_um)

    x0 = -0.5 * length_um
    x1 = 0.5 * length_um
    x_left = x0 - input_straight_length_um
    x_right = x1 + output_straight_length_um
    w_in = input_width_um
    w_out = output_width_um

    polygons = {
        "taper": (
            (x0, -0.5 * w_in),
            (x1, -0.5 * w_out),
            (x1, 0.5 * w_out),
            (x0, 0.5 * w_in),
        ),
    }

    if input_straight_length_um > 0:
        polygons["input"] = (
            (x_left, -0.5 * w_in),
            (x0, -0.5 * w_in),
            (x0, 0.5 * w_in),
            (x_left, 0.5 * w_in),
        )

    if output_straight_length_um > 0:
        polygons["output"] = (
            (x1, -0.5 * w_out),
            (x_right, -0.5 * w_out),
            (x_right, 0.5 * w_out),
            (x1, 0.5 * w_out),
        )

    return polygons


def plot_fdtd_polygons(
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    *,
    ax=None,
):
    """Plot FDTD layout polygons before converting them to Tidy3D."""

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure

    for name, polygon in polygons.items():
        points = np.asarray(polygon, dtype=float)
        closed = np.vstack([points, points[0]])
        ax.plot(closed[:, 0], closed[:, 1], marker="o", label=name)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x propagation [µm]")
    ax.set_ylabel("y lateral [µm]")
    ax.grid(True)
    ax.legend(loc="best")

    return fig, ax


def plot_mmi_2x2_vertices(
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    *,
    ax=None,
):
    """Plot MMI FDTD layout polygons before converting them to Tidy3D."""

    return plot_fdtd_polygons(polygons, ax=ax)


def plot_fdtd_field_xy(
    sim_data,
    *,
    monitor_name: str = "field_core",
    field_component: FDTDFieldComponent = "Ey",
    field_part: FDTDFieldPart = "real",
    frequency_index: int = 0,
    z_index: int = 0,
    polygons: Mapping[str, tuple[tuple[float, float], ...]] | None = None,
    show_geometry: bool = True,
    geometry_color: str = "white",
    cmap: str | None = None,
    ax=None,
):
    """Plot a planar FDTD field monitor with EME-style orientation.

    The horizontal axis is Tidy3D ``x`` (propagation, left to right) and the
    vertical axis is Tidy3D ``y`` (lateral position). This is intentionally the
    same visual convention used by the EME propagation plots.
    """

    import matplotlib.pyplot as plt

    monitor_data_lookup = getattr(sim_data, "monitor_data", None)
    available_monitors = (
        list(monitor_data_lookup.keys()) if monitor_data_lookup is not None else []
    )
    resolved_monitor_name = monitor_name

    try:
        monitor_data = sim_data[resolved_monitor_name]
    except KeyError:
        legacy_monitor_name = "field_z0"
        if monitor_name == "field_core":
            resolved_monitor_name = legacy_monitor_name
            try:
                monitor_data = sim_data[resolved_monitor_name]
            except KeyError as exc:
                raise KeyError(
                    f"Monitor '{monitor_name}' was not found in simulation data. "
                    f"Available monitors: {available_monitors}. "
                    "Older saved FDTD data may use monitor_name='field_z0'."
                ) from exc
        else:
            raise KeyError(
                f"Monitor '{monitor_name}' was not found in simulation data. "
                f"Available monitors: {available_monitors}."
            ) from None
    field = _get_fdtd_field_component(monitor_data, field_component)
    field_xy = _select_fdtd_field_xy(
        field,
        frequency_index=frequency_index,
        z_index=z_index,
    )
    x_um = np.asarray(field_xy.coords["x"].values, dtype=float)
    y_um = np.asarray(field_xy.coords["y"].values, dtype=float)
    values = np.asarray(field_xy.transpose("y", "x").values)
    plot_values = _field_part_values(values, field_part)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    if cmap is None:
        cmap = _default_fdtd_field_cmap(field_part)

    mesh = ax.pcolormesh(x_um, y_um, plot_values, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(_fdtd_field_colorbar_label(field_component, field_part))

    if show_geometry and polygons is not None:
        _plot_polygon_outlines(ax, polygons, color=geometry_color)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x propagation [µm]")
    ax.set_ylabel("y lateral [µm]")
    ax.set_title(f"{field_part}({field_component}) from {resolved_monitor_name}")

    return fig, ax


def extract_fdtd_layers(
    cross_section: CrossSection,
    *,
    core_name: str | None = "core",
    full_width_tolerance_um: float = 1e-9,
) -> tuple[FDTDLayer, tuple[FDTDLayer, ...]]:
    """Extract an FDTD core layer and supported surrounding layers.

    The core footprint is supplied by the MMI polygons. The cross-section
    supplies the vertical bounds, material, and ordering. Non-core structures
    are currently supported when they are full-width rectangles, which covers
    common BCB / Air / oxide stack examples. More complex surrounding geometry
    raises ``NotImplementedError`` rather than being silently ignored.
    """

    if not cross_section.structures:
        raise ValueError("cross_section must contain at least one core structure.")

    core_index = _find_core_structure_index(cross_section, core_name=core_name)
    layers = []

    for index, structure in enumerate(cross_section.structures):
        bounds = structure.bounds()

        if index == core_index:
            layers.append(
                FDTDLayer(
                    name=structure.name or "core",
                    z_min=bounds.z_min,
                    z_max=bounds.z_max,
                    material_name=structure.material.name,
                    material=structure.material,
                    priority=index + 1,
                    is_core=True,
                )
            )
            continue

        if not isinstance(structure, Rectangle):
            raise NotImplementedError(
                "FDTD stack conversion currently supports non-core surrounding "
                "layers only when they are full-width Rectangle objects. "
                f"Structure '{structure.name}' has type {type(structure).__name__}."
            )

        is_full_width = (
            structure.x_min <= cross_section.x_min + full_width_tolerance_um
            and structure.x_max >= cross_section.x_max - full_width_tolerance_um
        )
        if not is_full_width:
            raise NotImplementedError(
                "FDTD stack conversion currently supports non-core surrounding "
                "layers only when they span the full cross-section x domain. "
                f"Structure '{structure.name}' spans x=[{structure.x_min}, "
                f"{structure.x_max}], while the domain is "
                f"x=[{cross_section.x_min}, {cross_section.x_max}]."
            )

        layers.append(
            FDTDLayer(
                name=structure.name or f"layer_{index}",
                z_min=structure.z_min,
                z_max=structure.z_max,
                material_name=structure.material.name,
                material=structure.material,
                priority=index + 1,
                is_core=False,
            )
        )

    core_layers = tuple(layer for layer in layers if layer.is_core)
    surrounding_layers = tuple(layer for layer in layers if not layer.is_core)

    if len(core_layers) != 1:
        raise RuntimeError("Expected exactly one extracted FDTD core layer.")

    return core_layers[0], surrounding_layers


def mmi_2x2_simulation_bounds(
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    cross_section: CrossSection,
    *,
    pad_x_um: float,
    pad_y_um: float,
    pad_z_um: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return Tidy3D simulation center and size for MMI FDTD."""

    _validate_nonnegative("pad_x_um", pad_x_um)
    _validate_nonnegative("pad_y_um", pad_y_um)
    _validate_nonnegative("pad_z_um", pad_z_um)

    all_points = np.asarray(
        [point for polygon in polygons.values() for point in polygon],
        dtype=float,
    )
    xy_min = all_points.min(axis=0)
    xy_max = all_points.max(axis=0)

    min_corner = (
        float(xy_min[0] - pad_x_um),
        float(xy_min[1] - pad_y_um),
        float(cross_section.z_min - pad_z_um),
    )
    max_corner = (
        float(xy_max[0] + pad_x_um),
        float(xy_max[1] + pad_y_um),
        float(cross_section.z_max + pad_z_um),
    )
    center = tuple(0.5 * (a + b) for a, b in zip(min_corner, max_corner, strict=True))
    size = tuple(b - a for a, b in zip(min_corner, max_corner, strict=True))

    return center, size


def mmi_2x2_tidy3d_structures(
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    cross_section: CrossSection,
    *,
    wavelength_um: float,
    simulation_center: tuple[float, float, float],
    simulation_size: tuple[float, float, float],
    core_name: str | None = "core",
):
    """Convert MMI polygons and a cross-section stack into Tidy3D structures."""

    td = _import_tidy3d()
    core_layer, surrounding_layers = extract_fdtd_layers(
        cross_section,
        core_name=core_name,
    )
    structures = []

    for layer in surrounding_layers:
        structures.append(
            td.Structure(
                geometry=td.Box(
                    center=(
                        simulation_center[0],
                        simulation_center[1],
                        layer.z_center,
                    ),
                    size=(
                        simulation_size[0],
                        simulation_size[1],
                        layer.thickness,
                    ),
                ),
                medium=material_to_tidy3d_medium(
                    layer.material,
                    wavelength_um=wavelength_um,
                ),
                name=layer.name,
                priority=layer.priority,
            )
        )

    core_medium = material_to_tidy3d_medium(
        core_layer.material,
        wavelength_um=wavelength_um,
    )

    for name, polygon in polygons.items():
        structures.append(
            td.Structure(
                geometry=td.PolySlab(
                    vertices=polygon,
                    axis=2,
                    slab_bounds=(core_layer.z_min, core_layer.z_max),
                ),
                medium=core_medium,
                name=name,
                priority=core_layer.priority,
            )
        )

    return tuple(structures)


def build_mmi_2x2_fdtd_simulation(
    *,
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    cross_section: CrossSection,
    wavelength_um: float,
    input_port: InputPort = "bottom",
    io_y_position_um: float,
    access_width_um: float,
    bandwidth_um: float = 0.1,
    pad_x_um: float = 2.0,
    pad_y_um: float = 2.0,
    pad_z_um: float = 1.0,
    source_offset_um: float = 1.0,
    monitor_offset_um: float = 1.0,
    mode_plane_y_span_um: float | None = None,
    mode_plane_z_span_um: float | None = None,
    field_monitor_z_um: float | None = None,
    field_monitor_name: str = "field_core",
    min_steps_per_wvl: int = 20,
    run_time_s: float = 2e-12,
    core_name: str | None = "core",
):
    """Build a Tidy3D FDTD simulation for a 2x2 MMI layout."""

    _validate_positive("wavelength_um", wavelength_um)
    _validate_positive("access_width_um", access_width_um)
    _validate_positive("bandwidth_um", bandwidth_um)
    _validate_positive("source_offset_um", source_offset_um)
    _validate_positive("monitor_offset_um", monitor_offset_um)
    if mode_plane_y_span_um is not None:
        _validate_positive("mode_plane_y_span_um", mode_plane_y_span_um)
    if mode_plane_z_span_um is not None:
        _validate_positive("mode_plane_z_span_um", mode_plane_z_span_um)
    _validate_positive("min_steps_per_wvl", min_steps_per_wvl)
    _validate_positive("run_time_s", run_time_s)

    td = _import_tidy3d()
    center, size = mmi_2x2_simulation_bounds(
        polygons,
        cross_section,
        pad_x_um=pad_x_um,
        pad_y_um=pad_y_um,
        pad_z_um=pad_z_um,
    )
    structures = mmi_2x2_tidy3d_structures(
        polygons,
        cross_section,
        wavelength_um=wavelength_um,
        simulation_center=center,
        simulation_size=size,
        core_name=core_name,
    )
    sources, monitors = mmi_2x2_sources_and_monitors(
        polygons=polygons,
        cross_section=cross_section,
        wavelength_um=wavelength_um,
        input_port=input_port,
        io_y_position_um=io_y_position_um,
        access_width_um=access_width_um,
        bandwidth_um=bandwidth_um,
        source_offset_um=source_offset_um,
        monitor_offset_um=monitor_offset_um,
        mode_plane_y_span_um=mode_plane_y_span_um,
        mode_plane_z_span_um=mode_plane_z_span_um,
        field_monitor_z_um=field_monitor_z_um,
        field_monitor_name=field_monitor_name,
    )

    return td.Simulation(
        center=center,
        size=size,
        medium=material_to_tidy3d_medium(
            cross_section.background_material,
            wavelength_um=wavelength_um,
        ),
        structures=structures,
        sources=sources,
        monitors=monitors,
        run_time=run_time_s,
        grid_spec=td.GridSpec.auto(
            wavelength=wavelength_um,
            min_steps_per_wvl=min_steps_per_wvl,
        ),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    )

def build_mmi_1x2_fdtd_simulation(
    *,
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    cross_section: CrossSection,
    wavelength_um: float,
    output_y_position_um: float,
    access_width_um: float,
    straight_io_length_um: float,
    bandwidth_um: float = 0.1,
    pad_x_um: float = 2.0,
    pad_y_um: float = 2.0,
    pad_z_um: float = 1.0,
    source_offset_um: float = 2.0,
    monitor_offset_um: float = 2.0,
    mode_plane_y_span_um: float | None = None,
    mode_plane_z_span_um: float | None = None,
    field_monitor_z_um: float | None = None,
    field_monitor_name: str = "field_core",
    min_steps_per_wvl: int = 20,
    run_time_s: float = 2e-12,
    core_name: str | None = "core",
):
    """Build a Tidy3D FDTD simulation for a 1x2 MMI."""

    _validate_positive("wavelength_um", wavelength_um)
    _validate_positive("output_y_position_um", output_y_position_um)
    _validate_positive("access_width_um", access_width_um)
    _validate_positive("bandwidth_um", bandwidth_um)
    _validate_positive("min_steps_per_wvl", min_steps_per_wvl)
    _validate_positive("run_time_s", run_time_s)

    td = _import_tidy3d()

    # Simulation domain
    center, size = mmi_2x2_simulation_bounds(
        polygons,
        cross_section,
        pad_x_um=pad_x_um,
        pad_y_um=pad_y_um,
        pad_z_um=pad_z_um,
    )

    # Convert geometry into Tidy3D structures
    structures = mmi_2x2_tidy3d_structures(
        polygons,
        cross_section,
        wavelength_um=wavelength_um,
        simulation_center=center,
        simulation_size=size,
        core_name=core_name,
    )

    # 1x2 source and monitors
    sources, monitors = mmi_1x2_sources_and_monitors(
        polygons=polygons,
        cross_section=cross_section,
        wavelength_um=wavelength_um,
        output_y_position_um=output_y_position_um,
        access_width_um=access_width_um,
        straight_io_length_um=straight_io_length_um,
        bandwidth_um=bandwidth_um,
        source_offset_um=source_offset_um,
        monitor_offset_um=monitor_offset_um,
        mode_plane_y_span_um=mode_plane_y_span_um,
        mode_plane_z_span_um=mode_plane_z_span_um,
        field_monitor_z_um=field_monitor_z_um,
        field_monitor_name=field_monitor_name,
    )

    return td.Simulation(
        center=center,
        size=size,
        medium=material_to_tidy3d_medium(
            cross_section.background_material,
            wavelength_um=wavelength_um,
        ),
        structures=structures,
        sources=sources,
        monitors=monitors,
        run_time=run_time_s,
        grid_spec=td.GridSpec.auto(
            wavelength=wavelength_um,
            min_steps_per_wvl=min_steps_per_wvl,
        ),
        boundary_spec=td.BoundarySpec.all_sides(
            boundary=td.PML()
        ),
    )


def build_mmi_1x4_fdtd_simulation(
    *,
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    cross_section: CrossSection,
    wavelength_um: float,
    output_y_positions_um: tuple[float, float, float, float],
    access_width_um: float,
    straight_io_length_um: float,
    bandwidth_um: float = 0.1,
    pad_x_um: float = 2.0,
    pad_y_um: float = 2.0,
    pad_z_um: float = 1.0,
    source_offset_um: float = 1.0,
    monitor_offset_um: float = 1.0,
    mode_plane_y_span_um: float | None = None,
    mode_plane_z_span_um: float | None = None,
    field_monitor_z_um: float | None = None,
    field_monitor_name: str = "field_core",
    min_steps_per_wvl: int = 20,
    run_time_s: float = 2e-12,
    core_name: str | None = "core",
):
    """Build a Tidy3D FDTD simulation for a 1x4 MMI."""

    _validate_positive(
        "wavelength_um",
        wavelength_um,
    )

    _validate_positive(
        "access_width_um",
        access_width_um,
    )

    _validate_positive(
        "straight_io_length_um",
        straight_io_length_um,
    )

    _validate_positive(
        "bandwidth_um",
        bandwidth_um,
    )

    _validate_positive(
        "source_offset_um",
        source_offset_um,
    )

    _validate_positive(
        "monitor_offset_um",
        monitor_offset_um,
    )

    _validate_positive(
        "min_steps_per_wvl",
        min_steps_per_wvl,
    )

    _validate_positive(
        "run_time_s",
        run_time_s,
    )

    td = _import_tidy3d()

    # ========================================================
    # SIMULATION BOUNDS
    # ========================================================

    center, size = (
        mmi_2x2_simulation_bounds(
            polygons,
            cross_section,
            pad_x_um=pad_x_um,
            pad_y_um=pad_y_um,
            pad_z_um=pad_z_um,
        )
    )

    # ========================================================
    # STRUCTURES
    # ========================================================

    structures = (
        mmi_2x2_tidy3d_structures(
            polygons,
            cross_section,
            wavelength_um=wavelength_um,
            simulation_center=center,
            simulation_size=size,
            core_name=core_name,
        )
    )

    # ========================================================
    # SOURCE + MONITORS
    # ========================================================

    sources, monitors = (
        mmi_1x4_sources_and_monitors(
            polygons=polygons,
            cross_section=cross_section,
            wavelength_um=wavelength_um,

            output_y_positions_um=(
                output_y_positions_um
            ),

            access_width_um=access_width_um,

            straight_io_length_um=(
                straight_io_length_um
            ),

            bandwidth_um=bandwidth_um,

            source_offset_um=(
                source_offset_um
            ),

            monitor_offset_um=(
                monitor_offset_um
            ),

            mode_plane_y_span_um=(
                mode_plane_y_span_um
            ),

            mode_plane_z_span_um=(
                mode_plane_z_span_um
            ),

            field_monitor_z_um=(
                field_monitor_z_um
            ),

            field_monitor_name=(
                field_monitor_name
            ),
        )
    )

    # ========================================================
    # SIMULATION
    # ========================================================

    return td.Simulation(
        center=center,
        size=size,

        medium=material_to_tidy3d_medium(
            cross_section.background_material,
            wavelength_um=wavelength_um,
        ),

        structures=structures,

        sources=sources,

        monitors=monitors,

        run_time=run_time_s,

        grid_spec=td.GridSpec.auto(
            wavelength=wavelength_um,
            min_steps_per_wvl=min_steps_per_wvl,
        ),

        boundary_spec=td.BoundarySpec.all_sides(
            boundary=td.PML()
        ),
    )


def build_taper_fdtd_simulation(
    *,
    input_cross_section: CrossSection,
    output_cross_section: CrossSection,
    length_um: float,
    wavelength_um: float,
    input_straight_length_um: float = 5.0,
    output_straight_length_um: float = 5.0,
    bandwidth_um: float = 0.1,
    pad_x_um: float = 2.0,
    pad_y_um: float = 2.0,
    pad_z_um: float = 0.0,
    source_offset_um: float = 1.0,
    monitor_offset_um: float = 1.0,
    mode_plane_y_span_um: float | None = None,
    mode_plane_z_span_um: float | None = None,
    field_monitor_z_um: float | None = None,
    field_monitor_name: str = "field_core",
    min_steps_per_wvl: int = 20,
    run_time_s: float = 2e-12,
    core_name: str | None = "core",
):
    """Build a Tidy3D FDTD simulation for a centered linear waveguide taper."""

    _validate_positive("length_um", length_um)
    _validate_positive("wavelength_um", wavelength_um)
    _validate_nonnegative("input_straight_length_um", input_straight_length_um)
    _validate_nonnegative("output_straight_length_um", output_straight_length_um)
    _validate_positive("bandwidth_um", bandwidth_um)
    _validate_positive("source_offset_um", source_offset_um)
    _validate_positive("monitor_offset_um", monitor_offset_um)
    if mode_plane_y_span_um is not None:
        _validate_positive("mode_plane_y_span_um", mode_plane_y_span_um)
    if mode_plane_z_span_um is not None:
        _validate_positive("mode_plane_z_span_um", mode_plane_z_span_um)
    _validate_positive("min_steps_per_wvl", min_steps_per_wvl)
    _validate_positive("run_time_s", run_time_s)

    td = _import_tidy3d()
    input_core_layer, _ = _validate_taper_cross_sections(
        input_cross_section=input_cross_section,
        output_cross_section=output_cross_section,
        core_name=core_name,
    )
    input_width_um = _core_width(input_cross_section, core_name=core_name)
    output_width_um = _core_width(output_cross_section, core_name=core_name)
    polygons = taper_vertices(
        length_um=length_um,
        input_width_um=input_width_um,
        output_width_um=output_width_um,
        input_straight_length_um=input_straight_length_um,
        output_straight_length_um=output_straight_length_um,
    )
    center, size = mmi_2x2_simulation_bounds(
        polygons,
        input_cross_section,
        pad_x_um=pad_x_um,
        pad_y_um=pad_y_um,
        pad_z_um=pad_z_um,
    )
    structures = mmi_2x2_tidy3d_structures(
        polygons,
        input_cross_section,
        wavelength_um=wavelength_um,
        simulation_center=center,
        simulation_size=size,
        core_name=core_name,
    )
    mode_y_span = mode_plane_y_span_um or 2.0 * max(input_width_um, output_width_um)
    sources, monitors = taper_sources_and_monitors(
        polygons=polygons,
        cross_section=input_cross_section,
        wavelength_um=wavelength_um,
        mode_plane_y_span_um=mode_y_span,
        mode_plane_z_span_um=mode_plane_z_span_um,
        bandwidth_um=bandwidth_um,
        source_offset_um=source_offset_um,
        monitor_offset_um=monitor_offset_um,
        field_monitor_z_um=field_monitor_z_um or input_core_layer.z_center,
        field_monitor_name=field_monitor_name,
    )

    return td.Simulation(
        center=center,
        size=size,
        medium=material_to_tidy3d_medium(
            input_cross_section.background_material,
            wavelength_um=wavelength_um,
        ),
        structures=structures,
        sources=sources,
        monitors=monitors,
        run_time=run_time_s,
        grid_spec=td.GridSpec.auto(
            wavelength=wavelength_um,
            min_steps_per_wvl=min_steps_per_wvl,
        ),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
    ), polygons


def mmi_2x2_sources_and_monitors(
    *,
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    cross_section: CrossSection,
    wavelength_um: float,
    input_port: InputPort,
    io_y_position_um: float,
    access_width_um: float,
    bandwidth_um: float,
    source_offset_um: float,
    monitor_offset_um: float,
    mode_plane_y_span_um: float | None = None,
    mode_plane_z_span_um: float | None = None,
    field_monitor_z_um: float | None = None,
    field_monitor_name: str = "field_core",
):
    """Create mode source, flux/mode monitors, and a central field monitor."""

    if input_port not in ("top", "bottom"):
        raise ValueError("input_port must be 'top' or 'bottom'.")

    td = _import_tidy3d()
    freq0 = td.C_0 / wavelength_um
    fwidth = td.C_0 / wavelength_um**2 * bandwidth_um
    source_time = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    mode_spec = td.ModeSpec(num_modes=1)
    x_min, x_max = _polygon_x_bounds(polygons)
    y_source = io_y_position_um if input_port == "top" else -io_y_position_um
    x_source = x_min + source_offset_um
    x_monitor = x_max - monitor_offset_um
    x_ref_monitor = x_source - monitor_offset_um
    z_center = 0.5 * (cross_section.z_min + cross_section.z_max)
    y_span = mode_plane_y_span_um or 2.5 * access_width_um
    z_span = mode_plane_z_span_um or cross_section.z_span
    monitor_size = (
        0.0,
        y_span,
        z_span,
    )

    mode_source = td.ModeSource(
        center=(x_source, y_source, z_center),
        size=monitor_size,
        source_time=source_time,
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
        name=f"mode_source_{input_port}",
    )
    flux_reflection = td.FluxMonitor(
        center=(x_ref_monitor, y_source, z_center),
        size=monitor_size,
        freqs=[freq0],
        name="flux_reflection",
    )
    mode_reflection = td.ModeMonitor(
        center=(x_ref_monitor, y_source, z_center),
        size=monitor_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_reflection",
    )
    flux_top = td.FluxMonitor(
        center=(x_monitor, io_y_position_um, z_center),
        size=monitor_size,
        freqs=[freq0],
        name="flux_top",
    )
    flux_bottom = td.FluxMonitor(
        center=(x_monitor, -io_y_position_um, z_center),
        size=monitor_size,
        freqs=[freq0],
        name="flux_bottom",
    )
    mode_top = td.ModeMonitor(
        center=(x_monitor, io_y_position_um, z_center),
        size=monitor_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_top",
    )
    mode_bottom = td.ModeMonitor(
        center=(x_monitor, -io_y_position_um, z_center),
        size=monitor_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_bottom",
    )

    field_center, field_size = mmi_2x2_simulation_bounds(
        polygons,
        cross_section,
        pad_x_um=0.0,
        pad_y_um=0.0,
        pad_z_um=0.0,
    )
    if field_monitor_z_um is None:
        core_layer, _ = extract_fdtd_layers(cross_section)
        field_monitor_z_um = core_layer.z_center

    field_monitor = td.FieldMonitor(
        center=(field_center[0], field_center[1], field_monitor_z_um),
        size=(field_size[0], field_size[1], 0.0),
        freqs=[freq0],
        name=field_monitor_name,
    )

    return (
        (mode_source,),
        (
            flux_reflection,
            mode_reflection,
            flux_top,
            flux_bottom,
            mode_top,
            mode_bottom,
            field_monitor,
        ),
    )

def mmi_1x2_sources_and_monitors(
    *,
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    cross_section: CrossSection,
    wavelength_um: float,
    output_y_position_um: float,
    access_width_um: float,
    straight_io_length_um: float,
    bandwidth_um: float,
    source_offset_um: float,
    monitor_offset_um: float,
    mode_plane_y_span_um: float | None = None,
    mode_plane_z_span_um: float | None = None,
    field_monitor_z_um: float | None = None,
    field_monitor_name: str = "field_core",
):
    """Create sources and monitors for a 1x2 MMI."""

    _validate_positive("straight_io_length_um", straight_io_length_um)
    _validate_positive("source_offset_um", source_offset_um)
    _validate_positive("monitor_offset_um", monitor_offset_um)

    if source_offset_um >= straight_io_length_um:
        raise ValueError(
            "The source must be inside the straight input section."
        )

    if monitor_offset_um >= straight_io_length_um:
        raise ValueError(
            "Output monitors must be inside the straight output sections."
        )

    td = _import_tidy3d()

    freq0 = td.C_0 / wavelength_um
    fwidth = td.C_0 / wavelength_um**2 * bandwidth_um

    source_time = td.GaussianPulse(
        freq0=freq0,
        fwidth=fwidth,
    )

    mode_spec = td.ModeSpec(num_modes=1)

    x_min, x_max = _polygon_x_bounds(polygons)

    # Longitudinal positions
    x_source = x_min + source_offset_um

    # Reflection monitor upstream of the source,
    # but still inside the straight input section.
    x_ref_monitor = x_min + 0.5 * source_offset_um

    x_monitor = x_max - monitor_offset_um

    # Vertical position
    z_center = 0.5 * (
        cross_section.z_min + cross_section.z_max
    )

    y_span = (
        mode_plane_y_span_um
        if mode_plane_y_span_um is not None
        else 2.5 * access_width_um
    )

    z_span = (
        mode_plane_z_span_um
        if mode_plane_z_span_um is not None
        else cross_section.z_span
    )

    monitor_size = (
        0.0,
        y_span,
        z_span,
    )

    # ========================================================
    # SINGLE CENTERED SOURCE
    # ========================================================

    mode_source = td.ModeSource(
        center=(x_source, 0.0, z_center),
        size=monitor_size,
        source_time=source_time,
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
        name="mode_source_input",
    )

    # ========================================================
    # REFLECTION MONITORS
    # ========================================================

    flux_reflection = td.FluxMonitor(
        center=(x_ref_monitor, 0.0, z_center),
        size=monitor_size,
        freqs=[freq0],
        name="flux_reflection",
    )

    mode_reflection = td.ModeMonitor(
        center=(x_ref_monitor, 0.0, z_center),
        size=monitor_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_reflection",
    )

    # ========================================================
    # OUTPUT FLUX MONITORS
    # ========================================================

    flux_top = td.FluxMonitor(
        center=(x_monitor, +output_y_position_um, z_center),
        size=monitor_size,
        freqs=[freq0],
        name="flux_top",
    )

    flux_bottom = td.FluxMonitor(
        center=(x_monitor, -output_y_position_um, z_center),
        size=monitor_size,
        freqs=[freq0],
        name="flux_bottom",
    )

    # ========================================================
    # OUTPUT MODE MONITORS
    # ========================================================

    mode_top = td.ModeMonitor(
        center=(x_monitor, +output_y_position_um, z_center),
        size=monitor_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_top",
    )

    mode_bottom = td.ModeMonitor(
        center=(x_monitor, -output_y_position_um, z_center),
        size=monitor_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_bottom",
    )

    # ========================================================
    # FIELD MONITOR
    # ========================================================

    field_center, field_size = mmi_2x2_simulation_bounds(
        polygons,
        cross_section,
        pad_x_um=0.0,
        pad_y_um=0.0,
        pad_z_um=0.0,
    )

    if field_monitor_z_um is None:
        core_layer, _ = extract_fdtd_layers(cross_section)
        field_monitor_z_um = core_layer.z_center

    field_monitor = td.FieldMonitor(
        center=(
            field_center[0],
            field_center[1],
            field_monitor_z_um,
        ),
        size=(
            field_size[0],
            field_size[1],
            0.0,
        ),
        freqs=[freq0],
        name=field_monitor_name,
    )

    return (
        (mode_source,),
        (
            flux_reflection,
            mode_reflection,
            flux_top,
            flux_bottom,
            mode_top,
            mode_bottom,
            field_monitor,
        ),
    )

def mmi_1x4_sources_and_monitors(
    *,
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    cross_section: CrossSection,
    wavelength_um: float,
    output_y_positions_um: tuple[float, float, float, float],
    access_width_um: float,
    straight_io_length_um: float,
    bandwidth_um: float,
    source_offset_um: float,
    monitor_offset_um: float,
    mode_plane_y_span_um: float | None = None,
    mode_plane_z_span_um: float | None = None,
    field_monitor_z_um: float | None = None,
    field_monitor_name: str = "field_core",
):
    """Create source and monitors for a 1x4 MMI."""

    td = _import_tidy3d()

    freq0 = td.C_0 / wavelength_um

    fwidth = (
        td.C_0
        / wavelength_um**2
        * bandwidth_um
    )

    source_time = td.GaussianPulse(
        freq0=freq0,
        fwidth=fwidth,
    )

    mode_spec = td.ModeSpec(
        num_modes=1
    )

    x_min, x_max = _polygon_x_bounds(
        polygons
    )

    # --------------------------------------------------------
    # POSICIONES LONGITUDINALES
    # --------------------------------------------------------

    x_source = (
        x_min + source_offset_um
    )

    # Reflection monitor antes de la fuente,
    # pero dentro del tramo recto
    x_ref_monitor = (
        x_min + 0.5 * source_offset_um
    )

    x_monitor = (
        x_max - monitor_offset_um
    )

    # --------------------------------------------------------
    # POSICION VERTICAL
    # --------------------------------------------------------

    z_center = 0.5 * (
        cross_section.z_min
        + cross_section.z_max
    )

    y_span = (
        mode_plane_y_span_um
        if mode_plane_y_span_um is not None
        else 2.5 * access_width_um
    )

    z_span = (
        mode_plane_z_span_um
        if mode_plane_z_span_um is not None
        else cross_section.z_span
    )

    monitor_size = (
        0.0,
        y_span,
        z_span,
    )

    # ========================================================
    # SOURCE - ENTRADA CENTRADA
    # ========================================================

    mode_source = td.ModeSource(
        center=(
            x_source,
            0.0,
            z_center,
        ),
        size=monitor_size,
        source_time=source_time,
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
        name="mode_source_input",
    )

    # ========================================================
    # REFLECTION MONITORS
    # ========================================================

    flux_reflection = td.FluxMonitor(
        center=(
            x_ref_monitor,
            0.0,
            z_center,
        ),
        size=monitor_size,
        freqs=[freq0],
        name="flux_reflection",
    )

    mode_reflection = td.ModeMonitor(
        center=(
            x_ref_monitor,
            0.0,
            z_center,
        ),
        size=monitor_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_reflection",
    )

    # ========================================================
    # FOUR OUTPUTS
    # ========================================================

    output_flux_monitors = []
    output_mode_monitors = []

    for index, y_position in enumerate(
        output_y_positions_um,
        start=1,
    ):

        output_flux_monitors.append(
            td.FluxMonitor(
                center=(
                    x_monitor,
                    y_position,
                    z_center,
                ),
                size=monitor_size,
                freqs=[freq0],
                name=f"flux_{index}",
            )
        )

        output_mode_monitors.append(
            td.ModeMonitor(
                center=(
                    x_monitor,
                    y_position,
                    z_center,
                ),
                size=monitor_size,
                freqs=[freq0],
                mode_spec=mode_spec,
                name=f"mode_{index}",
            )
        )

    # ========================================================
    # FIELD MONITOR
    # ========================================================

    field_center, field_size = (
        mmi_2x2_simulation_bounds(
            polygons,
            cross_section,
            pad_x_um=0.0,
            pad_y_um=0.0,
            pad_z_um=0.0,
        )
    )

    if field_monitor_z_um is None:

        core_layer, _ = extract_fdtd_layers(
            cross_section
        )

        field_monitor_z_um = (
            core_layer.z_center
        )

    field_monitor = td.FieldMonitor(
        center=(
            field_center[0],
            field_center[1],
            field_monitor_z_um,
        ),
        size=(
            field_size[0],
            field_size[1],
            0.0,
        ),
        freqs=[freq0],
        name=field_monitor_name,
    )

    return (
        (mode_source,),
        (
            flux_reflection,
            mode_reflection,
            *output_flux_monitors,
            *output_mode_monitors,
            field_monitor,
        ),
    )

def taper_sources_and_monitors(
    *,
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    cross_section: CrossSection,
    wavelength_um: float,
    mode_plane_y_span_um: float,
    bandwidth_um: float,
    source_offset_um: float,
    monitor_offset_um: float,
    mode_plane_z_span_um: float | None = None,
    field_monitor_z_um: float | None = None,
    field_monitor_name: str = "field_core",
):
    """Create mode source and monitors for a centered taper FDTD simulation."""

    td = _import_tidy3d()
    freq0 = td.C_0 / wavelength_um
    fwidth = td.C_0 / wavelength_um**2 * bandwidth_um
    source_time = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    mode_spec = td.ModeSpec(num_modes=1)
    x_min, x_max = _polygon_x_bounds(polygons)
    x_source = x_min + source_offset_um
    x_monitor = x_max - monitor_offset_um
    x_ref_monitor = x_source - monitor_offset_um
    z_center = 0.5 * (cross_section.z_min + cross_section.z_max)
    z_span = mode_plane_z_span_um or cross_section.z_span
    mode_plane_size = (0.0, mode_plane_y_span_um, z_span)

    mode_source = td.ModeSource(
        center=(x_source, 0.0, z_center),
        size=mode_plane_size,
        source_time=source_time,
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
        name="mode_source_input",
    )
    flux_reflection = td.FluxMonitor(
        center=(x_ref_monitor, 0.0, z_center),
        size=mode_plane_size,
        freqs=[freq0],
        name="flux_reflection",
    )
    mode_reflection = td.ModeMonitor(
        center=(x_ref_monitor, 0.0, z_center),
        size=mode_plane_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_reflection",
    )
    flux_transmission = td.FluxMonitor(
        center=(x_monitor, 0.0, z_center),
        size=mode_plane_size,
        freqs=[freq0],
        name="flux_transmission",
    )
    mode_transmission = td.ModeMonitor(
        center=(x_monitor, 0.0, z_center),
        size=mode_plane_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="mode_transmission",
    )

    field_center, field_size = mmi_2x2_simulation_bounds(
        polygons,
        cross_section,
        pad_x_um=0.0,
        pad_y_um=0.0,
        pad_z_um=0.0,
    )
    if field_monitor_z_um is None:
        core_layer, _ = extract_fdtd_layers(cross_section)
        field_monitor_z_um = core_layer.z_center

    field_monitor = td.FieldMonitor(
        center=(field_center[0], field_center[1], field_monitor_z_um),
        size=(field_size[0], field_size[1], 0.0),
        freqs=[freq0],
        name=field_monitor_name,
    )

    return (
        (mode_source,),
        (
            flux_reflection,
            mode_reflection,
            flux_transmission,
            mode_transmission,
            field_monitor,
        ),
    )


def print_mmi_2x2_fluxes(sim_data):
    """Print and return reflected/transmitted fluxes from an MMI FDTD run."""

    reflected = -float(sim_data["flux_reflection"].flux.values[0])
    top = float(sim_data["flux_top"].flux.values[0])
    bottom = float(sim_data["flux_bottom"].flux.values[0])
    total = top + bottom
    balance = reflected + total

    print(f"R        = {reflected:.6f}")
    print(f"T_top    = {top:.6f}")
    print(f"T_bottom = {bottom:.6f}")
    print(f"T_total  = {total:.6f}")
    print(f"Balance  = {balance:.6f}")

    return reflected, top, bottom

def print_mmi_1x4_fluxes(sim_data):
    """Print and return reflected/transmitted fluxes from a 1x4 MMI FDTD run."""

    reflected = -float(
        sim_data["flux_reflection"].flux.values[0]
    )

    transmissions = [
        float(
            sim_data[f"flux_{index}"].flux.values[0]
        )
        for index in range(1, 5)
    ]

    total = float(sum(transmissions))

    balance = (reflected + total)

    print(f"R       = {reflected:.6f}")

    for index, transmission in enumerate(
        transmissions,
        start=1,
    ):
        print(
            f"T_{index}     = {transmission:.6f}"
        )

    print(f"T_total = {total:.6f}")

    print(f"Balance = {balance:.6f}")

    return (
        reflected,
        *transmissions,
    )


def print_taper_transmission(sim_data):
    """Print and return reflected/transmitted fluxes from a taper FDTD run."""

    reflected = -float(sim_data["flux_reflection"].flux.values[0])
    transmitted = float(sim_data["flux_transmission"].flux.values[0])
    balance = reflected + transmitted

    print(f"R        = {reflected:.6f}")
    print(f"T        = {transmitted:.6f}")
    print(f"Balance  = {balance:.6f}")

    return reflected, transmitted


def _validate_taper_cross_sections(
    *,
    input_cross_section: CrossSection,
    output_cross_section: CrossSection,
    core_name: str | None,
) -> tuple[FDTDLayer, FDTDLayer]:
    input_core, input_surrounding = extract_fdtd_layers(
        input_cross_section,
        core_name=core_name,
    )
    output_core, output_surrounding = extract_fdtd_layers(
        output_cross_section,
        core_name=core_name,
    )

    if input_cross_section.background_material.name != (
        output_cross_section.background_material.name
    ):
        raise ValueError(
            "input_cross_section and output_cross_section must use the same "
            "background material for the initial taper FDTD helper."
        )

    if not np.isclose(input_cross_section.z_min, output_cross_section.z_min):
        raise ValueError("Input and output cross-sections must share z_min.")

    if not np.isclose(input_cross_section.z_max, output_cross_section.z_max):
        raise ValueError("Input and output cross-sections must share z_max.")

    if input_core.material_name != output_core.material_name:
        raise ValueError("Input and output taper cores must use the same material.")

    if not np.isclose(input_core.z_min, output_core.z_min) or not np.isclose(
        input_core.z_max,
        output_core.z_max,
    ):
        raise ValueError("Input and output taper cores must share vertical bounds.")

    if len(input_surrounding) != len(output_surrounding):
        raise ValueError(
            "Input and output cross-sections must have the same surrounding stack."
        )

    for input_layer, output_layer in zip(
        input_surrounding,
        output_surrounding,
        strict=True,
    ):
        if input_layer.material_name != output_layer.material_name:
            raise ValueError(
                "Input and output surrounding layers must use the same materials."
            )
        if not np.isclose(input_layer.z_min, output_layer.z_min) or not np.isclose(
            input_layer.z_max,
            output_layer.z_max,
        ):
            raise ValueError(
                "Input and output surrounding layers must share vertical bounds."
            )

    return input_core, output_core


def _core_width(
    cross_section: CrossSection,
    *,
    core_name: str | None,
) -> float:
    core_index = _find_core_structure_index(cross_section, core_name=core_name)
    bounds = cross_section.structures[core_index].bounds()
    return bounds.x_max - bounds.x_min


def _find_core_structure_index(
    cross_section: CrossSection,
    *,
    core_name: str | None,
) -> int:
    if core_name is not None:
        for index, structure in enumerate(cross_section.structures):
            if structure.name.lower() == core_name.lower():
                return index

    material_indices = [
        float(np.real(structure.material.n()))
        for structure in cross_section.structures
    ]

    return int(np.argmax(material_indices))


def _polygon_x_bounds(
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
) -> tuple[float, float]:
    all_points = np.asarray(
        [point for polygon in polygons.values() for point in polygon],
        dtype=float,
    )

    return float(all_points[:, 0].min()), float(all_points[:, 0].max())


def _get_fdtd_field_component(
    monitor_data,
    field_component: FDTDFieldComponent,
):
    if hasattr(monitor_data, field_component):
        return getattr(monitor_data, field_component)

    field_components = getattr(monitor_data, "field_components", None)
    if field_components is not None and field_component in field_components:
        return field_components[field_component]

    available = [
        component
        for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        if hasattr(monitor_data, component)
    ]
    raise ValueError(
        f"Field component '{field_component}' was not found in monitor data. "
        f"Available components: {available}."
    )


def _select_fdtd_field_xy(
    field,
    *,
    frequency_index: int,
    z_index: int,
):
    field_xy = field

    if "f" in field_xy.dims:
        field_xy = field_xy.isel(f=frequency_index)

    if "z" in field_xy.dims:
        field_xy = field_xy.isel(z=z_index)

    extra_dims = set(field_xy.dims) - {"x", "y"}
    if extra_dims:
        raise ValueError(
            "Expected a planar FDTD field with only x and y dimensions after "
            f"selecting frequency and z. Remaining dimensions: {sorted(extra_dims)}."
        )

    if not {"x", "y"}.issubset(field_xy.dims):
        raise ValueError(
            "Expected FDTD field monitor data to contain x and y coordinates."
        )

    return field_xy


def _field_part_values(
    values: np.ndarray,
    field_part: FDTDFieldPart,
) -> np.ndarray:
    if field_part == "real":
        return np.real(values)
    if field_part == "mag":
        return np.abs(values)
    if field_part == "phase":
        return np.angle(values)
    if field_part == "intensity":
        return np.abs(values) ** 2

    raise ValueError(
        "field_part must be one of 'real', 'mag', 'phase', or 'intensity'."
    )


def _default_fdtd_field_cmap(field_part: FDTDFieldPart) -> str:
    if field_part == "real":
        return "RdBu_r"
    if field_part == "phase":
        return "twilight"
    if field_part == "intensity":
        return "inferno"
    return "viridis"


def _fdtd_field_colorbar_label(
    field_component: FDTDFieldComponent,
    field_part: FDTDFieldPart,
) -> str:
    if field_part == "real":
        return f"Re({field_component})"
    if field_part == "mag":
        return f"|{field_component}|"
    if field_part == "phase":
        return f"phase({field_component}) [rad]"
    if field_part == "intensity":
        return f"|{field_component}|²"
    return f"{field_part}({field_component})"


def _plot_polygon_outlines(
    ax,
    polygons: Mapping[str, tuple[tuple[float, float], ...]],
    *,
    color: str,
) -> None:
    for polygon in polygons.values():
        points = np.asarray(polygon, dtype=float)
        closed = np.vstack([points, points[0]])
        ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=0.8)


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _import_tidy3d():
    try:
        import tidy3d as td
    except ImportError as exc:
        raise ImportError(
            "The Tidy3D FDTD helpers require the optional 'tidy3d' dependency. "
            "Install it with `uv sync --extra tidy3d`."
        ) from exc

    return td
