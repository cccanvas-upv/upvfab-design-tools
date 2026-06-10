from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping

from .geometry import Bounds2D, GeometryPrimitive, Polygon, Rectangle
from .materials import Material


@dataclass(frozen=True)
class CrossSection:
    """
    2D optical cross-section in the x-z plane.

    Parameters
    ----------
    name:
        Cross-section name.
    background_material:
        Default material filling the whole simulation domain.
    x_min, x_max:
        Horizontal simulation domain limits in micrometers.
    z_min, z_max:
        Vertical simulation domain limits in micrometers.
    structures:
        Ordered list of geometry primitives.

        The order matters: later structures have priority over previous ones.
        This allows complex material layouts such as SiO2 background,
        BCB upper cladding, air windows, and a SiN core.
    metadata:
        Optional user-defined information.
    """

    name: str
    background_material: Material
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    structures: tuple[GeometryPrimitive, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be larger than x_min.")

        if self.z_max <= self.z_min:
            raise ValueError("z_max must be larger than z_min.")

        object.__setattr__(self, "structures", tuple(self.structures))

    @property
    def bounds(self) -> Bounds2D:
        return Bounds2D(
            x_min=self.x_min,
            x_max=self.x_max,
            z_min=self.z_min,
            z_max=self.z_max,
        )

    @property
    def x_span(self) -> float:
        return self.x_max - self.x_min

    @property
    def z_span(self) -> float:
        return self.z_max - self.z_min

    def background_region(self) -> Rectangle:
        """
        Return the simulation domain as a rectangle filled with the
        background material.
        """

        return Rectangle(
            x_min=self.x_min,
            x_max=self.x_max,
            z_min=self.z_min,
            z_max=self.z_max,
            material=self.background_material,
            name="background",
        )

    def all_regions(self, include_background: bool = True) -> tuple[GeometryPrimitive, ...]:
        """
        Return all material regions.

        If include_background=True, the first region is the full simulation
        domain filled with the background material.
        """

        if include_background:
            return (self.background_region(), *self.structures)

        return self.structures

    def polygons(self, include_background: bool = True) -> tuple[Polygon, ...]:
        """
        Return all regions converted to polygons.

        This is useful for plotting and for solver backends such as Femwell
        or Tidy3D.
        """

        return tuple(
            _to_polygon(region)
            for region in self.all_regions(include_background=include_background)
        )

    def materials(self) -> dict[str, Material]:
        """
        Return materials used in the cross-section, indexed by material name.
        """

        materials = {
            self.background_material.name: self.background_material,
        }

        for structure in self.structures:
            materials[structure.material.name] = structure.material

        return materials

    def with_structures(self, *structures: GeometryPrimitive) -> CrossSection:
        """
        Return a new cross-section with additional structures appended.

        Since CrossSection is frozen, this does not modify the original object.
        """

        return replace(
            self,
            structures=(*self.structures, *structures),
        )


def _to_polygon(region: GeometryPrimitive) -> Polygon:
    if isinstance(region, Polygon):
        return region

    if hasattr(region, "to_polygon"):
        return region.to_polygon()

    raise TypeError(f"Cannot convert region of type {type(region)} to Polygon.")