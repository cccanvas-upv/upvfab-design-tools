from __future__ import annotations

import numpy as np

from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import GeometryPrimitive, Polygon
from upvfab_design_tools.core.materials import Material


def material_to_tidy3d_medium(
    material: Material,
    *,
    wavelength_um: float,
):
    """Convert a backend-neutral material into a Tidy3D medium."""

    td = _import_tidy3d()
    epsilon = material.epsilon(wavelength_um)
    epsilon = _real_epsilon_for_tidy3d(epsilon, material.name)

    return td.Medium(permittivity=epsilon, name=material.name)


def geometry_to_tidy3d_structure(
    geometry: GeometryPrimitive,
    *,
    wavelength_um: float,
    propagation_length_um: float,
    priority: int,
):
    """Convert one x-z cross-section geometry into a Tidy3D z-extruded structure.

    UPVfab coordinates use ``x`` as the lateral coordinate and ``z`` as the
    vertical coordinate. Tidy3D uses ``x`` and ``y`` as the transverse mode-plane
    coordinates when propagation is along its ``z`` axis. Therefore the mapping
    is:

    ``UPVfab (x, z_vertical) -> Tidy3D (x, y)``.
    """

    td = _import_tidy3d()
    polygon = _to_polygon(geometry)
    medium = material_to_tidy3d_medium(
        polygon.material,
        wavelength_um=wavelength_um,
    )
    return td.Structure(
        geometry=td.PolySlab(
            vertices=polygon.vertices,
            axis=2,
            slab_bounds=(-td.inf, td.inf),
        ),
        medium=medium,
        name=polygon.name or None,
        priority=priority,
    )


def cross_section_to_tidy3d_simulation(
    cross_section: CrossSection,
    *,
    wavelength_um: float,
    propagation_length_um: float = 1.0,
    min_steps_per_wvl: int = 20,
    run_time_s: float = 1e-12,
):
    """Build a Tidy3D simulation from a backend-neutral cross-section.

    The simulation is intended for Tidy3D's mode solver, not for an FDTD device
    run. Structures are extruded along Tidy3D ``z``. Later
    ``CrossSection.structures`` receive larger priorities so the central
    cross-section priority rule is preserved in the Tidy3D representation.
    """

    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")

    if propagation_length_um <= 0:
        raise ValueError("propagation_length_um must be positive.")

    if min_steps_per_wvl <= 0:
        raise ValueError("min_steps_per_wvl must be positive.")

    td = _import_tidy3d()
    background = material_to_tidy3d_medium(
        cross_section.background_material,
        wavelength_um=wavelength_um,
    )
    structures = tuple(
        geometry_to_tidy3d_structure(
            structure,
            wavelength_um=wavelength_um,
            propagation_length_um=propagation_length_um,
            priority=index + 1,
        )
        for index, structure in enumerate(cross_section.structures)
    )

    center = (
        0.5 * (cross_section.x_min + cross_section.x_max),
        0.5 * (cross_section.z_min + cross_section.z_max),
        0.0,
    )
    size = (
        cross_section.x_span,
        cross_section.z_span,
        propagation_length_um,
    )

    return td.Simulation(
        center=center,
        size=size,
        medium=background,
        structures=structures,
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=False),
        grid_spec=td.GridSpec.auto(
            wavelength=wavelength_um,
            min_steps_per_wvl=min_steps_per_wvl,
        ),
        run_time=run_time_s,
    )


def mode_solver_plane(cross_section: CrossSection):
    """Return the transverse Tidy3D mode plane for a cross-section."""

    td = _import_tidy3d()
    center = (
        0.5 * (cross_section.x_min + cross_section.x_max),
        0.5 * (cross_section.z_min + cross_section.z_max),
        0.0,
    )

    return td.Box(center=center, size=(td.inf, td.inf, 0.0))


def _to_polygon(geometry: GeometryPrimitive) -> Polygon:
    if isinstance(geometry, Polygon):
        return geometry

    if hasattr(geometry, "to_polygon"):
        return geometry.to_polygon()

    raise TypeError(f"Cannot convert geometry of type {type(geometry)} to polygon.")


def _real_epsilon_for_tidy3d(
    epsilon: float | complex,
    material_name: str,
    tol: float = 1e-12,
) -> float:
    epsilon_complex = complex(epsilon)

    if abs(epsilon_complex.imag) > tol:
        raise NotImplementedError(
            "Complex permittivity is not supported yet in the initial Tidy3D "
            f"mode-solver backend. Material '{material_name}' has "
            f"epsilon={epsilon_complex}."
        )

    epsilon_real = float(np.real(epsilon_complex))

    if epsilon_real < 1.0:
        raise ValueError(
            "Tidy3D Medium requires permittivity >= 1. "
            f"Material '{material_name}' has epsilon={epsilon_real}."
        )

    return epsilon_real


def _import_tidy3d():
    try:
        import tidy3d as td
    except ImportError as exc:
        raise ImportError(
            "The Tidy3D backend requires the optional 'tidy3d' dependency. "
            "Install it with `uv sync --extra tidy3d`."
        ) from exc

    return td
