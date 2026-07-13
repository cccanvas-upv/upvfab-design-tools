"""Tidy3D conversion, configuration, and FDTD helpers.

The public helpers are imported lazily so lightweight configuration checks do
not import plotting or Tidy3D-related modules unnecessarily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "TIDY3D_API_KEY_ENV_VAR": ".config",
    "build_mmi_2x2_fdtd_simulation": ".fdtd",
    "build_taper_fdtd_simulation": ".fdtd",
    "cross_section_to_tidy3d_simulation": ".conversion",
    "estimate_tidy3d_cost": ".config",
    "extract_fdtd_layers": ".fdtd",
    "geometry_to_tidy3d_structure": ".conversion",
    "has_tidy3d_api_key": ".config",
    "load_tidy3d_simulation_data": ".config",
    "material_to_tidy3d_medium": ".conversion",
    "mmi_2x2_simulation_bounds": ".fdtd",
    "mmi_2x2_sources_and_monitors": ".fdtd",
    "mmi_2x2_tidy3d_structures": ".fdtd",
    "mmi_2x2_vertices": ".fdtd",
    "mode_solver_plane": ".conversion",
    "plot_fdtd_field_xy": ".fdtd",
    "plot_fdtd_polygons": ".fdtd",
    "plot_mmi_2x2_vertices": ".fdtd",
    "print_mmi_2x2_fluxes": ".fdtd",
    "print_taper_transmission": ".fdtd",
    "require_tidy3d_api_key": ".config",
    "taper_vertices": ".fdtd",
    "test_tidy3d_api": ".config",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Import Tidy3D plugin helpers only when they are first requested."""

    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORT_MODULES[name], package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
