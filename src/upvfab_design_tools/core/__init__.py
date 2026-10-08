from .cross_section import CrossSection
from .visualization import plot_cross_section, save_figure
from .waveguides import rib_waveguide, strip_waveguide

from .geometry import (
    Bounds2D,
    GeometryPrimitive,
    Polygon,
    Rectangle,
    Trapezoid,
)

from .materials import (
    BCB,
    CauchyIndexModel,
    LPCVD_SILICON_DIOXIDE,
    MATERIALS,
    SILICON,
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
    Material,
    get_material,
)

__all__ = [
    "CauchyIndexModel",
    "Material",
    "SILICON_NITRIDE",
    "THERMAL_SILICON_DIOXIDE",
    "LPCVD_SILICON_DIOXIDE",
    "SILICON",
    "BCB",
    "MATERIALS",
    "get_material",
    "Bounds2D",
    "Rectangle",
    "Polygon",
    "Trapezoid",
    "GeometryPrimitive",
    "CrossSection",
    "strip_waveguide",
    "rib_waveguide",
    "plot_cross_section",
    "save_figure",
]
