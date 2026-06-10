from __future__ import annotations

from dataclasses import dataclass
from math import radians, tan
from typing import Literal

from .materials import Material

Point2D = tuple[float, float]

""" 
Dimensions in micrometers are assumed throughout this geometry module.
"""

@dataclass()
class Bounds2D:
    """
    Bounding box of a 2D geometry in the x-z plane.
    """

    x_min: float
    x_max: float
    z_min: float
    z_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.z_max - self.z_min


@dataclass()
class Rectangle:
    """
    Rectangular 2D geometry in the x-z plane.

    Parameters
    ----------
    x_min:
        Minimum x coordinate in micrometers.
    x_max:
        Maximum x coordinate in micrometers.
    z_min:
        Minimum z coordinate in micrometers.
    z_max:
        Maximum z coordinate in micrometers.
    material:
        Material assigned to the rectangle.
    name:
        Optional geometry name.
    """

    x_min: float
    x_max: float
    z_min: float
    z_max: float
    material: Material
    name: str = ""

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be larger than x_min.")

        if self.z_max <= self.z_min:
            raise ValueError("z_max must be larger than z_min.")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.z_max - self.z_min

    @property
    def x_center(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def z_center(self) -> float:
        return 0.5 * (self.z_min + self.z_max)

    def bounds(self) -> Bounds2D:
        return Bounds2D(
            x_min=self.x_min,
            x_max=self.x_max,
            z_min=self.z_min,
            z_max=self.z_max,
        )

    def vertices(self) -> tuple[Point2D, ...]:
        """
        Return rectangle vertices in counter-clockwise order.
        """

        return (
            (self.x_min, self.z_min),
            (self.x_max, self.z_min),
            (self.x_max, self.z_max),
            (self.x_min, self.z_max),
        )

    def to_polygon(self) -> Polygon:
        return Polygon(
            vertices=self.vertices(),
            material=self.material,
            name=self.name,
        )


@dataclass()
class Polygon:
    """
    Generic polygonal 2D geometry in the x-z plane.

    Parameters
    ----------
    vertices:
        Polygon vertices as (x, z) pairs in micrometers.
    material:
        Material assigned to the polygon.
    name:
        Optional geometry name.
    """

    vertices: tuple[Point2D, ...]
    material: Material
    name: str = ""

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("A polygon requires at least three vertices.")

    def bounds(self) -> Bounds2D:
        xs = [point[0] for point in self.vertices]
        zs = [point[1] for point in self.vertices]

        return Bounds2D(
            x_min=min(xs),
            x_max=max(xs),
            z_min=min(zs),
            z_max=max(zs),
        )


@dataclass()
class Trapezoid:
    """
    Trapezoidal 2D geometry in the x-z plane.

    This is useful for waveguides with angled sidewalls.

    Parameters
    ----------
    x_center:
        Center x coordinate in micrometers.
    z_min:
        Bottom z coordinate in micrometers.
    height:
        Trapezoid height in micrometers.
    top_width:
        Width at the top of the trapezoid in micrometers.
    bottom_width:
        Width at the bottom of the trapezoid in micrometers.
    material:
        Material assigned to the trapezoid.
    name:
        Optional geometry name.

    Notes
    -----
    Vertices are returned in counter-clockwise order.
    """

    x_center: float
    z_min: float
    height: float
    top_width: float
    bottom_width: float
    material: Material
    name: str = ""

    def __post_init__(self) -> None:
        if self.height <= 0:
            raise ValueError("height must be positive.")

        if self.top_width <= 0:
            raise ValueError("top_width must be positive.")

        if self.bottom_width <= 0:
            raise ValueError("bottom_width must be positive.")

    @property
    def z_max(self) -> float:
        return self.z_min + self.height

    @property
    def top_x_min(self) -> float:
        return self.x_center - 0.5 * self.top_width

    @property
    def top_x_max(self) -> float:
        return self.x_center + 0.5 * self.top_width

    @property
    def bottom_x_min(self) -> float:
        return self.x_center - 0.5 * self.bottom_width

    @property
    def bottom_x_max(self) -> float:
        return self.x_center + 0.5 * self.bottom_width

    def bounds(self) -> Bounds2D:
        return Bounds2D(
            x_min=min(self.top_x_min, self.bottom_x_min),
            x_max=max(self.top_x_max, self.bottom_x_max),
            z_min=self.z_min,
            z_max=self.z_max,
        )

    def vertices(self) -> tuple[Point2D, ...]:
        """
        Return trapezoid vertices in counter-clockwise order.
        """

        return (
            (self.bottom_x_min, self.z_min),
            (self.bottom_x_max, self.z_min),
            (self.top_x_max, self.z_max),
            (self.top_x_min, self.z_max),
        )

    def to_polygon(self) -> Polygon:
        return Polygon(
            vertices=self.vertices(),
            material=self.material,
            name=self.name,
        )

    @classmethod
    def from_sidewall_angle(
        cls,
        *,
        x_center: float,
        z_min: float,
        height: float,
        width: float,
        sidewall_angle_deg: float,
        material: Material,
        width_reference: Literal["top", "bottom"] = "top",
        name: str = "",
    ) -> Trapezoid:
        """
        Create a trapezoid from a sidewall angle.

        Parameters
        ----------
        x_center:
            Center x coordinate in micrometers.
        z_min:
            Bottom z coordinate in micrometers.
        height:
            Trapezoid height in micrometers.
        width:
            Reference width in micrometers.
        sidewall_angle_deg:
            Sidewall angle in degrees, measured from the vertical direction.

            sidewall_angle_deg = 0 means vertical sidewalls.

            A positive angle means the bottom is wider than the top,
            which is the usual case for many etched waveguides.
        material:
            Material assigned to the trapezoid.
        width_reference:
            Whether `width` refers to the top or bottom width.
        name:
            Optional geometry name.
        """

        if height <= 0:
            raise ValueError("height must be positive.")

        if width <= 0:
            raise ValueError("width must be positive.")

        lateral_offset = height * tan(radians(sidewall_angle_deg))

        if width_reference == "top":
            top_width = width
            bottom_width = width + 2 * lateral_offset

        elif width_reference == "bottom":
            bottom_width = width
            top_width = width - 2 * lateral_offset

        else:
            raise ValueError("width_reference must be either 'top' or 'bottom'.")

        if top_width <= 0:
            raise ValueError(
                "Computed top_width is not positive. "
                "Check width, height and sidewall_angle_deg."
            )

        if bottom_width <= 0:
            raise ValueError(
                "Computed bottom_width is not positive. "
                "Check width, height and sidewall_angle_deg."
            )

        return cls(
            x_center=x_center,
            z_min=z_min,
            height=height,
            top_width=top_width,
            bottom_width=bottom_width,
            material=material,
            name=name,
        )


GeometryPrimitive = Rectangle | Polygon | Trapezoid