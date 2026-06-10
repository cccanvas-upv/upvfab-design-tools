from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box
from shapely.ops import unary_union
from skfem.io.meshio import from_meshio

from femwell.mesh import mesh_from_OrderedDict
from femwell.visualization import plot_domains
import femwell.maxwell.waveguide as fmwg

from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Polygon, Rectangle, Trapezoid
from upvfab_design_tools.core.materials import Material

from .base import BaseModeSolver
from .results import Mode, ModeSolverResult


ResolutionSpec = Mapping[str, float]


@dataclass(frozen=True)
class FemwellModeSolver(BaseModeSolver):
    """
    FEMWELL mode solver backend.

    Parameters
    ----------
    order:
        FEMWELL mode solver order.
    default_resolution:
        Default mesh resolution in micrometers.
    min_resolution:
        Minimum automatically generated resolution in micrometers.
    resolution_factor:
        The automatic resolution is approximately min_feature / resolution_factor.
    default_distance:
        Default distance parameter passed to FEMWELL meshing.
    default_resolution_max:
        Maximum default resolution passed to mesh_from_OrderedDict.
    resolution_overrides:
        Optional dictionary with per-domain meshing parameters.

        Example:
            {
                "core": {"resolution": 0.03, "distance": 0.3},
                "BCB": {"resolution": 0.01, "distance": 0.2},
            }
    filter_guided:
        If True, only keep modes with n_eff above the reference material.
    guided_tolerance:
        Minimum index difference for guided mode filtering.
    reference_material:
        Material used for guided-mode filtering. If None, the cross-section
        background material is used.
    enable_plots:
        If True, plot mesh, domains and epsilon.
    """

    backend: str = "femwell"

    order: int = 2
    default_resolution: float = 0.5
    min_resolution: float = 0.01
    resolution_factor: float = 5.0
    default_distance: float = 0.5
    default_resolution_max: float = 10.0
    resolution_overrides: Mapping[str, ResolutionSpec] = field(default_factory=dict)

    filter_guided: bool = True
    guided_tolerance: float = 1e-2
    reference_material: Material | None = None

    enable_plots: bool = False

    def solve(
        self,
        cross_section: CrossSection,
        wavelength_um: float,
        num_modes: int,
    ) -> ModeSolverResult:
        if wavelength_um <= 0:
            raise ValueError("wavelength_um must be positive.")

        if num_modes <= 0:
            raise ValueError("num_modes must be positive.")

        polygons, domain_materials = cross_section_to_femwell_domains(cross_section)

        resolutions = self._build_resolutions(polygons)

        mesh = from_meshio(
            mesh_from_OrderedDict(
                polygons,
                resolutions,
                default_resolution_max=self.default_resolution_max,
            )
        )

        if self.enable_plots:
            mesh.draw().show()
            plot_domains(mesh)

        basis0 = fmwg.Basis(mesh, fmwg.ElementTriP0())
        epsilon = basis0.zeros()

        for domain_name, material in domain_materials.items():
            # Known issue. FEMWELL expects a real-valued epsilon, so we take the real part if it's close to real.
            # Might check how to add support to complex values of n in the future. 
            n = material.n(wavelength_um)
            eps = np.real_if_close(material.epsilon(wavelength_um))
            epsilon[basis0.get_dofs(elements=domain_name)] = float(eps)
  

        if self.enable_plots:
            basis0.plot(epsilon, colorbar=True).show()

        femwell_modes = fmwg.compute_modes(
            basis0,
            epsilon,
            wavelength=wavelength_um,
            num_modes=num_modes,
            order=self.order,
        )

        modes = tuple(
            Mode(
                index=i,
                neff=complex(mode.n_eff),
                wavelength_um=wavelength_um,
                te_fraction=getattr(mode, "te_fraction", None),
                tm_fraction=getattr(mode, "tm_fraction", None),
                backend=self.backend,
                raw=mode,
            )
            for i, mode in enumerate(femwell_modes)
        )

        if self.filter_guided:
            reference_material = self.reference_material or cross_section.background_material
            reference_index = np.real(reference_material.n(wavelength_um))

            modes = tuple(
                mode
                for mode in modes
                if np.real(mode.neff) - reference_index >= self.guided_tolerance
            )

        return ModeSolverResult(
            modes=modes,
            wavelength_um=wavelength_um,
            backend=self.backend,
            cross_section=cross_section,
            metadata={
                "mesh": mesh,
                "basis": basis0,
                "epsilon": epsilon,
                "domain_materials": domain_materials,
                "resolutions": resolutions,
                "raw_modes": femwell_modes,
            },
        )

    def _build_resolutions(
        self,
        polygons: OrderedDict[str, ShapelyPolygon],
    ) -> dict[str, dict[str, float]]:
        resolutions: dict[str, dict[str, float]] = {}

        for domain_name, polygon in polygons.items():
            if domain_name in self.resolution_overrides:
                resolutions[domain_name] = dict(self.resolution_overrides[domain_name])
                continue

            x_min, z_min, x_max, z_max = polygon.bounds
            width = abs(x_max - x_min)
            height = abs(z_max - z_min)

            min_feature = min(width, height)

            if min_feature <= 0:
                resolution = self.default_resolution
            else:
                resolution = min(
                    self.default_resolution,
                    max(min_feature / self.resolution_factor, self.min_resolution),
                )

            resolutions[domain_name] = {
                "resolution": resolution,
                "distance": self.default_distance,
            }

        return resolutions


def cross_section_to_femwell_domains(
    cross_section: CrossSection,
) -> tuple[OrderedDict[str, ShapelyPolygon], dict[str, Material]]:
    """
    Convert a CrossSection into non-overlapping FEMWELL domains.

    Later regions in the cross-section have priority over earlier regions.
    """

    regions = cross_section.all_regions(include_background=True)

    domain_box = box(
        cross_section.x_min,
        cross_section.z_min,
        cross_section.x_max,
        cross_section.z_max,
    )

    raw_domains: list[tuple[str, ShapelyPolygon, Material]] = []

    for i, region in enumerate(regions):
        name = _safe_domain_name(region.name or f"region_{i}")
        shape = _geometry_to_shapely(region)
        shape = shape.intersection(domain_box)

        if not shape.is_empty:
            raw_domains.append((name, shape, region.material))

    polygons: OrderedDict[str, ShapelyPolygon] = OrderedDict()
    domain_materials: dict[str, Material] = {}

    for i, (name, shape, material) in enumerate(raw_domains):
        later_shapes = [raw_domains[j][1] for j in range(i + 1, len(raw_domains))]

        if later_shapes:
            shape = shape.difference(unary_union(later_shapes))

        if shape.is_empty:
            continue

        unique_name = _make_unique_name(name, polygons)

        polygons[unique_name] = shape
        domain_materials[unique_name] = material

    return polygons, domain_materials


def _geometry_to_shapely(region: Rectangle | Trapezoid | Polygon) -> ShapelyPolygon:
    if isinstance(region, Rectangle):
        return box(region.x_min, region.z_min, region.x_max, region.z_max)

    if isinstance(region, Trapezoid):
        return ShapelyPolygon(region.vertices())

    if isinstance(region, Polygon):
        return ShapelyPolygon(region.vertices)

    if hasattr(region, "to_polygon"):
        polygon = region.to_polygon()
        return ShapelyPolygon(polygon.vertices)

    raise TypeError(f"Unsupported geometry type: {type(region)}")


def _safe_domain_name(name: str) -> str:
    clean = name.strip().replace(" ", "_").replace("-", "_")

    if not clean:
        clean = "region"

    return clean


def _make_unique_name(
    name: str,
    existing: OrderedDict[str, ShapelyPolygon],
) -> str:
    if name not in existing:
        return name

    counter = 1

    while f"{name}_{counter}" in existing:
        counter += 1

    return f"{name}_{counter}"