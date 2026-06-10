from __future__ import annotations

from typing import Iterable, Literal

from .cross_section import CrossSection
from .geometry import GeometryPrimitive, Rectangle, Trapezoid
from .materials import Material


def strip_waveguide(
    *,
    width: float,
    height: float,
    core_material: Material,
    background_material: Material,
    x_center: float = 0.0,
    z_min: float = 0.0,
    sidewall_angle_deg: float = 0.0,
    width_reference: Literal["top", "bottom"] = "top",
    x_span: float = 6.0,
    bottom_margin: float = 2.0,
    top_margin: float = 2.0,
    surrounding_regions: Iterable[GeometryPrimitive] = (),
    name: str = "strip_waveguide",
) -> CrossSection:
    """
    Create a strip waveguide cross-section.

    If sidewall_angle_deg = 0, the core is rectangular.
    If sidewall_angle_deg != 0, the core is trapezoidal.
    """

    if width <= 0:
        raise ValueError("width must be positive.")

    if height <= 0:
        raise ValueError("height must be positive.")

    x_min = x_center - 0.5 * x_span
    x_max = x_center + 0.5 * x_span
    domain_z_min = z_min - bottom_margin
    domain_z_max = z_min + height + top_margin

    core = _make_waveguide_core(
        width=width,
        height=height,
        material=core_material,
        x_center=x_center,
        z_min=z_min,
        sidewall_angle_deg=sidewall_angle_deg,
        width_reference=width_reference,
        name="core",
    )

    return CrossSection(
        name=name,
        background_material=background_material,
        x_min=x_min,
        x_max=x_max,
        z_min=domain_z_min,
        z_max=domain_z_max,
        structures=(*tuple(surrounding_regions), core),
    )


def rib_waveguide(
    *,
    ridge_width: float,
    total_thickness: float,
    slab_thickness: float,
    slab_width: float,
    core_material: Material,
    background_material: Material,
    x_center: float = 0.0,
    z_min: float = 0.0,
    sidewall_angle_deg: float = 0.0,
    width_reference: Literal["top", "bottom"] = "top",
    x_span: float = 8.0,
    bottom_margin: float = 2.0,
    top_margin: float = 2.0,
    surrounding_regions: Iterable[GeometryPrimitive] = (),
    name: str = "rib_waveguide",
) -> CrossSection:
    """
    Create a rib waveguide cross-section.

    The rib is represented as:

        slab  -> rectangle
        ridge -> rectangle or trapezoid
    """

    if ridge_width <= 0:
        raise ValueError("ridge_width must be positive.")

    if total_thickness <= 0:
        raise ValueError("total_thickness must be positive.")

    if slab_thickness < 0:
        raise ValueError("slab_thickness cannot be negative.")

    if slab_thickness >= total_thickness:
        raise ValueError("slab_thickness must be smaller than total_thickness.")

    if slab_width <= 0:
        raise ValueError("slab_width must be positive.")

    ridge_height = total_thickness - slab_thickness

    x_min = x_center - 0.5 * x_span
    x_max = x_center + 0.5 * x_span
    domain_z_min = z_min - bottom_margin
    domain_z_max = z_min + total_thickness + top_margin

    slab = Rectangle(
        x_min=x_center - 0.5 * slab_width,
        x_max=x_center + 0.5 * slab_width,
        z_min=z_min,
        z_max=z_min + slab_thickness,
        material=core_material,
        name="slab",
    )

    ridge = _make_waveguide_core(
        width=ridge_width,
        height=ridge_height,
        material=core_material,
        x_center=x_center,
        z_min=z_min + slab_thickness,
        sidewall_angle_deg=sidewall_angle_deg,
        width_reference=width_reference,
        name="ridge",
    )

    return CrossSection(
        name=name,
        background_material=background_material,
        x_min=x_min,
        x_max=x_max,
        z_min=domain_z_min,
        z_max=domain_z_max,
        structures=(*tuple(surrounding_regions), slab, ridge),
    )


def _make_waveguide_core(
    *,
    width: float,
    height: float,
    material: Material,
    x_center: float,
    z_min: float,
    sidewall_angle_deg: float,
    width_reference: Literal["top", "bottom"],
    name: str,
) -> GeometryPrimitive:
    """
    Create a rectangular or trapezoidal waveguide core.
    """

    if abs(sidewall_angle_deg) < 1e-12:
        return Rectangle(
            x_min=x_center - 0.5 * width,
            x_max=x_center + 0.5 * width,
            z_min=z_min,
            z_max=z_min + height,
            material=material,
            name=name,
        )

    return Trapezoid.from_sidewall_angle(
        x_center=x_center,
        z_min=z_min,
        height=height,
        width=width,
        width_reference=width_reference,
        sidewall_angle_deg=sidewall_angle_deg,
        material=material,
        name=name,
    )