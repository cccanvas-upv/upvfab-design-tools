# Arquitectura y reglas técnicas — UPVfab Design Tools

Este documento es la fuente de verdad técnica para la arquitectura, las convenciones físicas y las decisiones de implementación. Las palabras **DEBE**, **NO DEBE** y **SOLO** expresan requisitos del repositorio; una excepción debe ser explícita y actualizar esta documentación junto con el cambio de código.

## Reglas arquitectónicas obligatorias

1. `core/` describe materiales, geometría y secciones físicas. **NO DEBE** importar backends como FEMWELL, Tidy3D o gplugins.
2. `modesolver/` define la interfaz y los resultados genéricos; cada backend adapta la geometría neutral y devuelve `Mode` / `ModeSolverResult`.
3. `eme/` consume resultados genéricos. **NO DEBE** depender directamente de objetos FEMWELL o Tidy3D.
4. Conversiones a formatos y objetos de proveedores pertenecen al módulo del backend o plugin correspondiente, nunca a `core/`.
5. La geometría y el stack físicos se definen una sola vez en `core.CrossSection`. Los plugins deben convertir esa descripción y respetar prioridad de estructuras; **NO DEBEN** duplicar silenciosamente la geometría física.
6. Las dimensiones y longitudes de onda se expresan en micrómetros salvo que una API indique explícitamente otra unidad.
7. Los materiales, geometrías, secciones y resultados físicos se mantienen inmutables cuando sea práctico (`@dataclass(frozen=True)`).
8. Una limitación de backend debe producir un error claro (`NotImplementedError` cuando corresponda), nunca omitir una región o aproximar silenciosamente la física.
9. Las llamadas a servicios cloud, simulaciones con coste y estimaciones de coste deben ser opt-in; ejemplos locales no las ejecutan implícitamente.
10. Una modificación de estas reglas debe incluir el cambio correspondiente en este documento y explicar la decisión en el PR.

Para las instrucciones operativas de agentes, consulta [`../AGENTS.md`](../AGENTS.md). Para flujo de trabajo humano, consulta [`../CONTRIBUTING.md`](../CONTRIBUTING.md).

This file provides persistent context and working instructions for code agents working on the `upvfab-design-tools` repository.

The project is being developed by/with Camilo Cano for UPVfab-related photonic integrated circuit design and simulation workflows.

## Repository purpose

`upvfab-design-tools` is a Python package for photonic integrated circuit design and simulation tools used at UPVfab.

The package should provide reusable building blocks for:

- Defining optical materials and material stacks.
- Defining 2D cross-sections for waveguides and PIC components.
- Solving optical modes using different backends, initially FEMWELL and later Tidy3D.
- Passing generic mode results to other modules such as EME.
- Building higher-level component design tools, such as MMI design tools.
- Eventually supporting layout/simulation/export workflows.

Repository name:

```text
upvfab-design-tools
```

Python package name:

```python
upvfab_design_tools
```

Recommended import style:

```python
import upvfab_design_tools as udt
```

## Expected repository structure

The repository uses a `src/` layout:

```text
upvfab-design-tools/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .gitignore
├── examples/
│   └── notebooks/
├── tests/
└── src/
    └── upvfab_design_tools/
        ├── __init__.py
        ├── core/
        │   ├── __init__.py
        │   ├── materials.py
        │   ├── geometry.py
        │   ├── cross_section.py
        │   ├── waveguides.py
        │   ├── visualization.py
        │   └── units.py
        ├── modesolver/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── results.py
        │   ├── femwell_solver.py
        │   └── visualization.py
        ├── eme/
        │   ├── __init__.py
        │   ├── solver.py
        │   └── sections.py
        ├── tidy3d_plugin/
        │   ├── __init__.py
        │   ├── simulation.py
        │   └── geometry_export.py
        └── components/
            ├── __init__.py
            └── mmi/
                ├── __init__.py
                ├── geometry.py
                ├── design.py
                ├── simulation.py
                ├── sweep.py
                └── visualization.py
```

The package may grow, but preserve this separation of concerns.

## Core architectural principles

1. `core/` defines physical descriptions, not solver implementations.
2. `modesolver/` defines solver interfaces, generic result objects, and backend-specific solvers.
3. `components/` defines device-level design tools such as MMIs.
4. `eme/` should consume generic mode results, not FEMWELL-specific or Tidy3D-specific objects directly.
5. `tidy3d_plugin/` should contain conversion and simulation helpers specific to Tidy3D.
6. Backend-specific dependencies such as FEMWELL or Tidy3D must not be imported from `core/`.
7. Keep units in micrometers unless a function explicitly says otherwise.
8. Prefer immutable physical objects using `@dataclass(frozen=True)` for materials, geometries, cross-sections and mode results.
9. Prefer clean, explicit APIs over hidden global state.
10. Favor simple implementations first, but keep extension points for later backends and component tools.

Useful mental model:

```text
core/                -> what structure/material system is being simulated
modesolver/          -> how modes are computed
modesolver/femwell   -> how the generic cross-section is converted to FEMWELL
modesolver/tidy3d    -> future equivalent backend using Tidy3D
eme/                 -> how generic modes are propagated/coupled
components/mmi/      -> how MMI structures are designed using core + solvers
```

## Packaging

The project should use a minimal `pyproject.toml` based on `setuptools` and `src/` package discovery.

Recommended minimal structure:

```toml
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "upvfab-design-tools"
version = "0.1.0"
description = "Python tools for photonic integrated circuit design and simulation at UPVfab"
readme = "README.md"
requires-python = ">=3.10"
authors = [
    { name = "UPVfab" }
]
dependencies = [
    "numpy",
    "scipy",
    "matplotlib",
    "pyyaml",
]

[project.optional-dependencies]
modesolver = [
    "femwell",
    "shapely",
    "scikit-fem",
    "meshio",
]

tidy3d = [
    "tidy3d",
]

layout = [
    "gdstk",
    "shapely",
]

plot = [
    "PyQt6",
]

dev = [
    "pytest",
    "ruff",
    "black",
    "ipykernel",
]

all = [
    "femwell",
    "shapely",
    "scikit-fem",
    "meshio",
    "tidy3d",
    "gdstk",
    "PyQt6",
]

[tool.setuptools.packages.find]
where = ["src"]
```

Using `uv`, sync extras like this:

```bash
uv sync --extra modesolver --extra dev
uv sync --extra modesolver --extra plot --extra dev
uv sync --all-extras
```

Run examples through `uv`:

```bash
uv run python examples/some_example.py
```

## `core/materials.py`

Materials are backend-independent and should not import FEMWELL, Tidy3D or gplugins.

A material is represented by a frozen dataclass, for example:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

IndexModel = float | complex | Callable[[float], float | complex]

@dataclass(frozen=True)
class Material:
    name: str
    index_model: IndexModel
    description: str = ""
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def n(self, wavelength_um: float = 1.55) -> complex:
        if wavelength_um <= 0:
            raise ValueError("wavelength_um must be positive.")

        if callable(self.index_model):
            value = self.index_model(wavelength_um)
        else:
            value = self.index_model

        return complex(value)

    def epsilon(self, wavelength_um: float = 1.55) -> complex:
        return self.n(wavelength_um) ** 2
```

The `index_model` can be:

- a constant real index, e.g. `2.0`
- a constant complex index, e.g. `3.48 + 1e-4j`
- a function `n(wavelength_um) -> float | complex`

All wavelengths are in micrometers.

Example dispersive material:

```python
def n_bcb_cauchy(wavelength_um: float) -> float:
    A = 1.535
    B = 0.003
    C = 0.0
    return A + B / wavelength_um**2 + C / wavelength_um**4

BCB_DISPERSIVE = Material(
    name="Benzocyclobutene dispersive",
    index_model=n_bcb_cauchy,
    aliases=("BCB dispersive",),
)
```

Initial UPVfab materials should include:

- `AIR`
- `SILICON_NITRIDE`
- `THERMAL_SILICON_DIOXIDE`
- `LPCVD_SILICON_DIOXIDE`
- `SILICON`
- `BCB`

Approximate initial constants around 1.55 um are acceptable, but comments should state that they are defaults and should be replaced by process/PDK/measured data when available.

Suggested initial values:

```python
AIR = Material("Air", 1.0, aliases=("air",))
SILICON_NITRIDE = Material("Silicon Nitride", 2.00, aliases=("SiN", "Si3N4", "silicon_nitride"))
THERMAL_SILICON_DIOXIDE = Material("Thermal Silicon Dioxide", 1.444, aliases=("SiO2 thermal", "thermal_sio2", "thermal oxide"))
LPCVD_SILICON_DIOXIDE = Material("LPCVD Silicon Dioxide", 1.46, aliases=("SiO2 LPCVD", "lpcvd_sio2", "LPCVD oxide"))
SILICON = Material("Silicon", 3.476, aliases=("Si", "silicon"))
BCB = Material("Benzocyclobutene", 1.537, aliases=("BCB", "benzocyclobutene"))
```

Provide a material registry and `get_material(name: str) -> Material` helper.

## `core/geometry.py`

This module defines neutral 2D geometry primitives in the x-z cross-section plane.

Coordinate convention:

```text
x -> lateral coordinate of the cross-section
z -> vertical coordinate of the cross-section
```

All dimensions are in micrometers.

Do not implement solver-specific geometry here.

Recommended primitives:

- `Bounds2D`
- `Rectangle`
- `Polygon`
- `Trapezoid`
- `GeometryPrimitive = Rectangle | Polygon | Trapezoid`

`Rectangle` should expose:

- `width`
- `height`
- `x_center`
- `z_center`
- `bounds()`
- `vertices()` returning counter-clockwise `(x, z)` pairs
- `to_polygon()`

`Polygon` should store arbitrary vertices and expose `bounds()`.

`Trapezoid` should support sidewall-angle waveguides, with a constructor:

```python
Trapezoid.from_sidewall_angle(
    *,
    x_center: float,
    z_min: float,
    height: float,
    width: float,
    sidewall_angle_deg: float,
    material: Material,
    width_reference: Literal["top", "bottom"] = "top",
    name: str = "",
)
```

Convention for sidewall angle:

```text
sidewall_angle_deg = 0 -> vertical sidewalls
sidewall_angle_deg > 0 -> bottom wider than top
```

This supports future fabrication variation modeling.

Strip waveguides can be represented as a `Rectangle` or `Trapezoid`.

Rib waveguides can be represented as a `Rectangle` slab plus a `Rectangle` or `Trapezoid` ridge.

## `core/cross_section.py`

A `CrossSection` is a complete 2D simulation domain, not merely `core + cladding`.

It should support simple homogeneous claddings and more complex material regions such as:

```text
background material: SiO2
core: SiN
thin layer above core: BCB
upper region: Air
```

Recommended fields:

```python
@dataclass(frozen=True)
class CrossSection:
    name: str
    background_material: Material
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    structures: tuple[GeometryPrimitive, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Important composition rule:

```text
The order of `structures` matters.
Later structures have priority over earlier structures.
```

This rule allows layered/overlapping structures where later shapes overwrite earlier shapes.

Useful methods:

- `bounds`
- `x_span`
- `z_span`
- `background_region()`
- `all_regions(include_background=True)`
- `polygons(include_background=True)`
- `materials()`
- `with_structures(*structures)`

Example complex cross-section:

```python
from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle
from upvfab_design_tools.core.materials import (
    AIR,
    BCB,
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)

bcb_region = Rectangle(
    x_min=-4.0,
    x_max=4.0,
    z_min=0.3,
    z_max=0.35,
    material=BCB,
    name="BCB",
)

air_region = Rectangle(
    x_min=-4.0,
    x_max=4.0,
    z_min=0.35,
    z_max=2.0,
    material=AIR,
    name="Air",
)

sin_core = Rectangle(
    x_min=-0.5,
    x_max=0.5,
    z_min=0.0,
    z_max=0.3,
    material=SILICON_NITRIDE,
    name="SiN",
)

xs = CrossSection(
    name="sin_bcb_air_cross_section",
    background_material=THERMAL_SILICON_DIOXIDE,
    x_min=-4.0,
    x_max=4.0,
    z_min=-2.0,
    z_max=2.0,
    structures=(
        bcb_region,
        air_region,
        sin_core,
    ),
)
```

This represents:

- SiO2 background throughout the domain.
- SiN core from z=0.0 to z=0.3.
- Thin BCB layer from z=0.3 to z=0.35.
- Air above z=0.35.

## `core/waveguides.py`

This module should provide high-level helpers for common waveguide cross-sections.

Recommended helpers:

- `strip_waveguide(...) -> CrossSection`
- `rib_waveguide(...) -> CrossSection`

They should build `CrossSection` objects using `Rectangle` and/or `Trapezoid`.

The helpers should support:

- rectangular strip waveguides
- trapezoidal strip waveguides
- rectangular rib waveguides
- trapezoidal rib waveguides
- optional surrounding regions through something like `surrounding_regions: Iterable[GeometryPrimitive] = ()`

They must remain backend-independent.

## `core/visualization.py`

This module visualizes physical cross-sections before they are sent to a solver.

Provide:

```python
plot_cross_section(cross_section, wavelength_um=1.55, ...)
save_figure(fig, filename, dpi=300)
```

`plot_cross_section` should color each region by the material refractive index `n(lambda)` at the chosen wavelength.

Do not call it “effective index”; before solving modes this is the material refractive index.

Suggested features:

- Draw background first.
- Draw `structures` in order.
- Later structures appear on top.
- Use a continuous colormap based on `Re(n)`.
- Add a colorbar labelled `Refractive index n`.
- Optionally show region names and indices.
- Hide the background label by default to avoid clutter.
- Preserve aspect ratio.
- Label axes as `x [µm]` and `z [µm]`.

Example use:

```python
from upvfab_design_tools.core.visualization import plot_cross_section, save_figure

fig, ax = plot_cross_section(xs, wavelength_um=1.55, show_names=True, show_indices=True)
save_figure(fig, "cross_section_before_modesolver.png")
```

For visual example scripts during the current development phase, prefer Camilo's
interactive-first pattern: try to display figures with `plt.show()` and save
them only as a fallback when the environment/backend cannot show plots.

## `modesolver/results.py`

The package should expose backend-independent mode/result data types.

Recommended classes:

```python
@dataclass(frozen=True)
class Mode:
    index: int
    neff: complex
    wavelength_um: float
    te_fraction: float | None = None
    tm_fraction: float | None = None
    backend: str = ""
    raw: Any = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
```

`Mode` should expose:

- `beta = 2*pi*neff/wavelength_um`
- `polarization`, based on TE/TM fractions if available
- `overlap(other)`, delegating to backend raw objects when possible

Recommended result class:

```python
@dataclass(frozen=True)
class ModeSolverResult:
    modes: tuple[Mode, ...]
    wavelength_um: float
    backend: str
    cross_section: Any = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
```

`ModeSolverResult` should expose:

- `__len__`
- `__getitem__`
- `neffs`
- `te_modes`
- `tm_modes`
- `raw_modes`

The goal is that EME and other modules consume `Mode` and `ModeSolverResult`, not FEMWELL or Tidy3D objects directly.

## `modesolver/base.py`

Define an abstract solver interface:

```python
class BaseModeSolver(ABC):
    backend: str

    @abstractmethod
    def solve(
        self,
        cross_section: CrossSection,
        wavelength_um: float,
        num_modes: int,
    ) -> ModeSolverResult:
        ...
```

All future backends should implement this interface.

## `modesolver/femwell_solver.py`

The initial backend is `FemwellModeSolver`.

It should convert:

```text
CrossSection -> shapely domains -> FEMWELL mesh -> epsilon -> FEMWELL modes -> ModeSolverResult
```

The implementation is based on an earlier working FEMWELL flow:

- Build geometries with `shapely`.
- Use `femwell.mesh.mesh_from_OrderedDict`.
- Convert using `skfem.io.meshio.from_meshio`.
- Create a FEMWELL basis with `fmwg.Basis(mesh, fmwg.ElementTriP0())`.
- Fill `epsilon` by domain.
- Call `fmwg.compute_modes(...)`.
- Optionally filter guided modes using a reference material index.

Important: `CrossSection` allows overlapping regions with priority by order. Before meshing, convert to non-overlapping FEMWELL domains.

Algorithm:

1. Start with `cross_section.all_regions(include_background=True)`.
2. Convert each region to a Shapely polygon.
3. Clip to the simulation domain.
4. For each region, subtract the union of all later regions.
5. Drop empty domains.
6. Use domain names as FEMWELL subdomain names.
7. Build `domain_materials` mapping from domain name to material.

The conversion helper can be named:

```python
cross_section_to_femwell_domains(cross_section)
```

FEMWELL currently assumes lossless materials in this implementation. Since `Material.n()` returns `complex`, explicitly convert epsilon to real only if the imaginary part is negligible.

Use a helper like:

```python
def _real_epsilon_for_femwell(
    epsilon: float | complex,
    material_name: str,
    tol: float = 1e-12,
) -> float:
    epsilon_complex = complex(epsilon)

    if abs(epsilon_complex.imag) > tol:
        raise NotImplementedError(
            "Complex permittivity is not supported yet in the FEMWELL backend. "
            f"Material '{material_name}' has epsilon={epsilon_complex}."
        )

    return float(epsilon_complex.real)
```

Then use:

```python
for domain_name, material in domain_materials.items():
    eps = material.epsilon(wavelength_um)
    eps = _real_epsilon_for_femwell(eps, material.name)
    epsilon[basis0.get_dofs(elements=domain_name)] = eps
```

This avoids the error:

```text
TypeError: float() argument must be a string or a real number, not 'complex'
```

Mesh resolutions should support automatic defaults and overrides:

```python
solver = FemwellModeSolver(
    resolution_overrides={
        "core": {"resolution": 0.02, "distance": 0.2},
        "BCB": {"resolution": 0.01, "distance": 0.2},
        "Air": {"resolution": 0.2, "distance": 1.0},
        "background": {"resolution": 0.4, "distance": 2.0},
    },
    reference_material=THERMAL_SILICON_DIOXIDE,
)
```

Be careful that override keys match the actual region names.

## `modesolver/tidy3d_solver.py` and `tidy3d_plugin/`

The Tidy3D mode-solver backend must use the same backend-neutral geometry from
`core.CrossSection`; do not rebuild waveguide geometry independently in a way
that can drift from the central API.

Coordinate convention for the Tidy3D conversion:

```text
UPVfab cross-section x          -> Tidy3D x
UPVfab cross-section z vertical -> Tidy3D y
propagation direction           -> Tidy3D z
```

`tidy3d_plugin/conversion.py` owns conversion from `Material`,
`GeometryPrimitive`, and `CrossSection` to Tidy3D objects. `core/` must not
import Tidy3D. Structures should be extruded along Tidy3D `z` for uniform
waveguide mode solving. Preserve the `CrossSection.structures` priority rule by
assigning increasing Tidy3D structure priorities to later structures.

`modesolver/tidy3d_solver.py` should implement `BaseModeSolver` and return the
same generic `ModeSolverResult` / `Mode` objects used by FEMWELL. Store the
Tidy3D `ModeSolverData` needed for field interpolation in a small raw wrapper,
but keep that wrapper behind the generic `Mode.raw` field.

Tidy3D field sampling belongs in `modesolver/postprocessing.py`, not in `eme/`.
For sampled profiles, map UPVfab `z_um` cuts to Tidy3D `y` interpolation and use
the same public `sample_mode_profile()` / `sample_mode_profiles()` API consumed
by EME.

### Tidy3D FDTD helpers

Experimental FDTD simulation builders should live in:

```text
src/upvfab_design_tools/tidy3d_plugin/fdtd.py
```

and examples should live in:

```text
examples/fdtd/
```

Do not promote MMI-specific objects to `components/mmi/` until the FDTD and EME
workflows have been validated. In the current experimental phase, prefer
function-based helpers over component classes.

For FDTD PIC simulations, use the coordinate convention:

```text
Tidy3D x -> propagation direction
Tidy3D y -> lateral coordinate
Tidy3D z -> vertical coordinate
```

MMI FDTD helpers may generate explicit layout polygons in the propagation /
lateral plane, but the vertical stack and materials must come from a
`core.CrossSection`. The initial supported stack model is:

- `cross_section.background_material` as the Tidy3D simulation background.
- One selected core layer, whose material and vertical bounds are used to
  extrude the MMI/access polygons as `PolySlab` structures.
- Optional non-core full-width `Rectangle` layers such as BCB or Air, converted
  to full-domain Tidy3D `Box` structures with their original materials and
  vertical bounds.

If a cross-section contains non-core structures that are not full-width
rectangles, raise a clear `NotImplementedError` rather than silently omitting or
approximating them. The purpose is to guarantee that Tidy3D draws the same
physical stack described by the central `CrossSection`.

Cloud execution with `tidy3d.web.run(...)` must be opt-in in examples, e.g.
behind `RUN_TIDY3D_CLOUD = False`, to avoid accidental paid runs. Examples
should build, validate, and plot simulations locally by default.

Cost estimation should also be opt-in, e.g. behind
`ESTIMATE_TIDY3D_COST = False`. Estimating cost does not run the simulation, but
it does contact Tidy3D cloud and creates an estimation task, so examples should
not do it silently during local-only validation.

MMI FDTD source and monitor planes should be explicitly controlled. For the
current 2x2 MMI examples, the input port labels map as:

```text
input_port="top"    -> source centered at y = +io_y_position_um
input_port="bottom" -> source centered at y = -io_y_position_um
```

The source direction is along positive Tidy3D `x` from left to right. Keep the
source and modal monitor planes centered at the vertical center of the
`CrossSection` domain, and size them with explicit arguments such as
`mode_plane_y_span_um` and `mode_plane_z_span_um`. Avoid making the lateral
plane so wide that the input source overlaps the other access waveguide. For
the current 2 um-wide access guides separated at `y=±2 um`, a useful default is:

```python
MODE_PLANE_Y_SPAN_UM = 2.5 * ACCESS_WIDTH_UM
```

FDTD field-result visualization should follow the same orientation convention
as EME propagation plots:

```text
horizontal axis -> Tidy3D x propagation coordinate, left to right
vertical axis   -> Tidy3D y lateral coordinate
color           -> selected field component/part, e.g. Re(Ey) or |Ey|²
```

For MMI FDTD examples, prefer a planar `FieldMonitor` named `field_core` placed
at the vertical center of the extracted core layer. Use helpers such as
`plot_fdtd_field_xy(..., field_component="Ey")` to avoid accidental transposed
or bottom-to-top propagation plots. Because older downloaded simulation data may
have used the monitor name `field_z0`, plotting helpers should gracefully fall
back from `field_core` to `field_z0` when loading legacy results.

When plotting FDTD fields with geometry overlays, use white geometry lines by
default because dark outlines are hard to see on common field colormaps.

Examples should support plotting previously downloaded Tidy3D results without
rerunning cloud simulations. Prefer an explicit flag such as
`LOAD_TIDY3D_DATA = False` and a shared path such as
`TIDY3D_DATA_PATH = "data/upvfab_mmi_2x2_fdtd.hdf5"`. Loading from disk should
use a small helper around `tidy3d.SimulationData.from_file(...)`, then reuse the
same post-processing/plotting path as freshly returned `web.run(...)` data.

Current FDTD result visualization should support at least:

- `field_component="Ey"` for the MMI field view.
- `field_part="real"`, `"mag"`, `"phase"`, and `"intensity"`.
- Optional polygon overlays from the same layout vertices used to build the
  Tidy3D structures.

### Tidy3D FDTD taper examples

Linear taper FDTD examples should also live in `examples/fdtd/` and use
function helpers from `tidy3d_plugin/fdtd.py`, not component classes yet.

The taper FDTD workflow should be:

```text
input CrossSection + output CrossSection + taper length
    -> centered taper polygons in Tidy3D x/y
    -> Tidy3D FDTD simulation
    -> reflected/transmitted fluxes and optional Ey field plot
```

The input/output cross-sections must share the same background material,
vertical domain, core material, core vertical bounds, and surrounding layer
stack. Their core widths define the taper start/end widths. The helper may add
short straight input/output sections so the mode source and output monitor are
placed on uniform waveguides rather than directly on the taper transition.

For a single centered taper there is no neighboring access guide, so the modal
plane can be wider than in MMI, but it should still remain inside the simulation
domain. A practical default for the taper helper is:

```python
mode_plane_y_span_um = 2.0 * max(input_width_um, output_width_um)
```

Use monitor names:

- `flux_reflection`
- `mode_reflection`
- `flux_transmission`
- `mode_transmission`
- `field_core`

and use `print_taper_transmission(sim_data)` for the initial transmission
report.

### Tidy3D cloud API configuration

Tidy3D cloud authentication should be configured through the official
environment variable:

```bash
export SIMCLOUD_APIKEY="your_api_key"
```

Do not store API keys in Python files, notebooks, examples, `.env` files that
could be committed, or AGENTS.md. The repository `.gitignore` should ignore
local secret/config files such as `.env`.

Authentication helpers belong in:

```text
src/upvfab_design_tools/tidy3d_plugin/config.py
```

Expose small helpers such as:

- `has_tidy3d_api_key()`
- `require_tidy3d_api_key()`
- `test_tidy3d_api()`
- `estimate_tidy3d_cost()`
- `load_tidy3d_simulation_data()`

`test_tidy3d_api()` should check that `SIMCLOUD_APIKEY` exists and then delegate
to `tidy3d.web.test()`. Examples should call it only before cloud execution
branches, not during local-only simulation construction/validation.

Use the dedicated example to validate authentication:

```bash
uv run python examples/fdtd/test_tidy3d_api.py
```

## `modesolver/visualization.py`

This module visualizes mode-solver results.

It should work on generic `Mode` and `ModeSolverResult` objects, but may internally access `mode.raw` for backend-specific details.

Initial functions:

- `plot_mode(mode, field_component="auto", field_part="real", ...)`
- `plot_mode_ex_ey(mode, field_part="real", ...)`
- `plot_modes_grid(result, field_component="auto", field_part="real", max_modes=None, ...)`
- `plot_effective_indices(result, ...)`
- `plot_epsilon(result, ...)`
- `save_figure(fig, filename, dpi=300)`

`field_part` should support:

- `"mag"`: plot the field magnitude.
- `"phase"`: plot the field phase.
- `"real"`: plot the real part of the field.

When `field_component="auto"`, choose the plotted field component from the mode TE/TM fractions:

- TE-like modes should plot `Ex`.
- TM-like modes should plot `Ey`.

Explicit `field_component="Ex"` or `field_component="Ey"` should remain supported for manual inspection.

The initial implementation can be FEMWELL-oriented, using:

```python
(et, et_basis), _ = raw_mode.basis.split(raw_mode.E)
plot_basis = et_basis.with_element(ElementVector(ElementDG(ElementTriP1())))
et_xy = plot_basis.project(et_basis.interpolate(et))
(et_x, et_x_basis), (et_y, et_y_basis) = plot_basis.split(et_xy)
```

The old implementation plotted `Ex` and `Ey` from FEMWELL modes this way. Preserve that behavior but adapt it to the generic `Mode` wrapper through `mode.raw`.

For future Tidy3D support, detect `mode.backend == "tidy3d"` and implement a separate extraction path.

Current Tidy3D mode visualization is supported through the same public plotting
functions used for FEMWELL:

- `plot_mode(...)`
- `plot_mode_ex_ey(...)`
- `plot_modes_grid(...)`

For Tidy3D raw mode data, plot fields from `Mode.raw.data.field_components`.
Use the coordinate mapping from the Tidy3D conversion layer: Tidy3D `x` is the
plot x-axis and Tidy3D `y` is labelled as the UPVfab vertical `z` axis. The
`show_mesh` option draws FEMWELL mesh/domain boundaries for FEMWELL results and
backend-neutral `CrossSection` structure outlines for Tidy3D results. Store the
originating `CrossSection` in Tidy3D raw mode data so these geometry overlays
remain tied to the same central geometry used for solving.
`plot_effective_indices(...)` remains fully backend-neutral.

## Previous FEMWELL utilities worth preserving conceptually

An earlier local implementation included these utilities:

- `guided_modes(modes, mat_cladd, TOL=1e-2)`
- `get_TETM(modes)`
- `slice_mode(mode, num_points=1024, x0=-4, x1=4, polarization="TE")`
- `mode_overlap(mode_single, modes_multiple)`
- `propagate_modes(...)`
- `my_plot_mode(mode, i, xlim=(-2, 2), ylim=(-1, 1))`

These should be reintroduced in the new architecture carefully:

- filtering and TE/TM classification belong in `modesolver/results.py` or solver post-processing
- 1D field slicing could go in `modesolver/postprocessing.py`
- modal overlaps should ideally use the generic `Mode.overlap()` API
- propagation belongs more naturally in `eme/` or a future modal propagation module
- plotting belongs in `modesolver/visualization.py`

## Example workflow

```python
from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle
from upvfab_design_tools.core.materials import (
    AIR,
    BCB,
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.visualization import plot_cross_section, save_figure
from upvfab_design_tools.modesolver import FemwellModeSolver
from upvfab_design_tools.modesolver.visualization import plot_mode_ex_ey, plot_epsilon

bcb_region = Rectangle(
    x_min=-4.0,
    x_max=4.0,
    z_min=0.3,
    z_max=0.35,
    material=BCB,
    name="BCB",
)

air_region = Rectangle(
    x_min=-4.0,
    x_max=4.0,
    z_min=0.35,
    z_max=2.0,
    material=AIR,
    name="Air",
)

sin_core = Rectangle(
    x_min=-0.5,
    x_max=0.5,
    z_min=0.0,
    z_max=0.3,
    material=SILICON_NITRIDE,
    name="core",
)

xs = CrossSection(
    name="sin_bcb_air_cross_section",
    background_material=THERMAL_SILICON_DIOXIDE,
    x_min=-4.0,
    x_max=4.0,
    z_min=-2.0,
    z_max=2.0,
    structures=(bcb_region, air_region, sin_core),
)

fig, ax = plot_cross_section(xs, wavelength_um=1.55)
save_figure(fig, "cross_section.png")

solver = FemwellModeSolver(
    resolution_overrides={
        "core": {"resolution": 0.02, "distance": 0.2},
        "BCB": {"resolution": 0.01, "distance": 0.2},
        "Air": {"resolution": 0.2, "distance": 1.0},
        "background": {"resolution": 0.4, "distance": 2.0},
    },
    reference_material=THERMAL_SILICON_DIOXIDE,
    filter_guided=True,
)

result = solver.solve(xs, wavelength_um=1.55, num_modes=4)

for mode in result.modes:
    print(mode.index, mode.neff, mode.polarization, mode.te_fraction, mode.tm_fraction)

fig, ax = plot_epsilon(result)
save_figure(fig, "epsilon.png")

fig, axs = plot_mode_ex_ey(result[0], xlim=(-2, 2), zlim=(-1, 1))
save_figure(fig, "mode_0_ex_ey.png")
```

## Handling Matplotlib in VSCode/headless environments

During local visual validation, examples may try to show figures first:

```python
try:
    plt.show()
except Exception:
    save_figure(fig, "output.png")
```

This keeps the workflow convenient in VSCode or other GUI-capable sessions,
while still allowing scripts to produce files in headless environments.

Be aware that `plt.show()` can fail or warn with:

```text
FigureCanvasAgg is non-interactive, and thus cannot be shown
```

Only use an interactive backend if the environment has a GUI and the corresponding packages are installed.

For optional interactive plotting:

```bash
uv add PyQt6 --optional plot
uv sync --extra plot
```

Then in local scripts only:

```python
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
```

Do not make Qt a required dependency of the package.

## Current project decisions

These are recent working conventions that should be preserved unless Camilo explicitly changes them.

### Example-first validation

Examples are educational material for external users, maintained and curated by
Camilo. Existing scripts are transitional and may be reset after a few weeks.
Keep examples organized by module area:

- `examples/core/` for materials, geometry, cross-section, and waveguide checks.
- `examples/modesolver/` for mode-solver examples such as
  `examples/modesolver/test_modesolver.py`.
- `examples/eme/` for eigenmode-expansion examples.

Formal tests in `tests/` can be added later, but do not assume new validation code should go there unless requested. When asked to add a quick check or visual smoke test during the current development phase, prefer updating the relevant example script.

### Visualization layers to preserve

Keep both cross-section and mode-result visualizations because they answer different questions:

- `core.visualization.plot_cross_section()` shows the physical/material cross-section before solving. It colors regions by material refractive index.
- `modesolver.visualization.plot_effective_indices()` shows solved modal effective indices after the mode solver has run.

Do not merge or rename these as if they represented the same concept.

### Current EME architecture

The first EME implementation handles propagation through one uniform section.
It consumes a generic `ModeSolverResult`, explicit complex initial modal
amplitudes, a propagation length `length_um`, and a longitudinal sampling step
`dz_um`. The EME engine must not assume a particular input port, polarization,
or modesolver backend.

Preserve the propagation convention:

```text
a_m(z) = a_m(0) * exp(-1j * beta_m * z)
```

where `beta_m` comes from the generic `Mode.beta` property. The final
longitudinal sample should be placed exactly at `length_um`, even when the
length is not an integer multiple of `dz_um`.

Current module responsibilities are:

- `eme/solver.py`: uniform-section modal propagation.
- `eme/results.py`: immutable propagation results and field reconstruction.
- `eme/overlap.py`: normalized scalar 1D profile overlaps, excitation
  coefficients, lateral profile shifts, and modal overlap matrices between
  sampled bases.
- `eme/visualization.py`: reconstructed propagation plots of `|E(x, z)|^2`.
- `modesolver/postprocessing.py`: backend-specific extraction of sampled mode
  profiles, initially for FEMWELL.

The backend-neutral contract between `modesolver/` and `eme/` is:

```text
ModeSolverResult + modesolver.postprocessing.sample_mode_profiles(...)
    -> complex sampled profiles
    -> eme.overlap / eme.solver / eme.results / eme.visualization
```

`eme/` must not import FEMWELL, Tidy3D, or any future solver backend. New
backends should return `ModeSolverResult` objects and add only their field
sampling adapter inside `modesolver/postprocessing.py`. Unsupported backends
should raise a clear `NotImplementedError` from profile sampling rather than
failing later inside EME.

Currently supported profile-sampling backends are exposed through
`modesolver.SUPPORTED_PROFILE_SAMPLING_BACKENDS`. At this stage the list is
expected to contain FEMWELL only. When Tidy3D mode solving is added, the next
step is to implement the Tidy3D raw-field interpreter for
`sample_mode_profile()` / `sample_mode_profiles()`.

For scalar 1D excitation, use the normalized complex inner product:

```text
a_m(0) = integral(conj(E_m(x)) * E_input(x), x)
```

Input and modal profiles must be sampled on the same strictly increasing
`x_um` grid. Normalize profiles using their integrated intensity before both
overlap calculation and propagated-field reconstruction. This scalar overlap
is the current approximation; a full vector electromagnetic power overlap may
be added later without changing the backend-independent propagation engine.

For translating an input excitation laterally in a homogeneous surrounding
medium, prefer shifting the sampled input profile instead of re-solving an
equivalent translated input waveguide. Use `eme.shift_profile(profile, x_um,
shift_um)`, where positive `shift_um` moves the profile toward positive `x`.
The shift should be implemented by interpolation with zero/default fill outside
the sampled window. Do not use `np.roll`, because periodic wrap-around is
unphysical for optical field profiles. Re-solve the input waveguide only when
the translated geometry or surrounding material environment is not physically
equivalent.

For coupling between two different sampled modal bases, use
`eme.modal_overlap_matrix(source_profiles, target_profiles, x_um)`. The
convention is:

```text
target_amplitudes = C @ source_amplitudes
C[target_index, source_index] = integral(conj(E_target) * E_source, x)
```

Both source and target profiles should be sampled on the same strictly
increasing `x_um` grid and normalized row-by-row before the overlap calculation.
This scalar basis-change approximation is the current building block for
stepwise EME examples such as tapers.

For non-uniform devices such as tapers, the current recommended example-first
approach is to discretize the geometry into a list of uniform sections, solve
the modes of each section, propagate inside each section, and project the
amplitudes to the next section basis using `modal_overlap_matrix`. Keep this in
`examples/eme/` until the API feels stable enough to promote into a reusable
multi-section EME module.

Coherent field reconstruction must combine the same field component for every
mode. For a TE family use `Ex`; for a TM family use `Ey`. Automatic component
selection may use TE -> Ex and TM -> Ey for individual modes, but it must raise
a clear error if it would mix Ex and Ey in one reconstructed scalar field.

The initial visualization follows the earlier EME implementation: propagation
distance `z` is horizontal, transverse position `x` is vertical, and color
shows `|E(x, z)|^2`. Keep the colormap configurable. Component extraction and
FEMWELL field probing belong outside `eme/` so that future Tidy3D modes can use
the same EME solver and result objects.

## Future `components/mmi/` direction

MMI design tools should live in:

```text
src/upvfab_design_tools/components/mmi/
```

This module should use `core`, `modesolver`, and later `eme`, but should not duplicate solver logic.

Possible files:

- `geometry.py`: build MMI regions, access waveguides, ports and tapers
- `design.py`: design formulas/heuristics for MMI length, width, access positions
- `simulation.py`: connect MMI geometry to EME or Tidy3D simulation workflows
- `sweep.py`: parameter sweeps
- `visualization.py`: component-level plots

Example future high-level API:

```python
from upvfab_design_tools.components.mmi import MMIDesign

mmi = MMIDesign(
    wavelength=1.55,
    width=6.0,
    length=120.0,
    input_width=0.5,
    output_width=0.5,
    n_inputs=1,
    n_outputs=2,
)

mmi.simulate(method="eme")
mmi.plot()
```

## Style guidelines for code agents

- Do not make large architecture changes without preserving the separation described here.
- Keep public APIs simple and documented.
- Add docstrings to new public classes/functions.
- Use type hints.
- Prefer small, testable functions.
- Avoid solver imports in `core/`.
- Avoid hard-coded process-specific constants unless labelled as defaults.
- Use explicit wavelength units: `wavelength_um`.
- Keep examples runnable from the repository root.
- For visual examples, prefer the interactive-first pattern:
  `try: plt.show()` and save figures in the fallback path for headless runs.
- When adding optional dependencies, place them in the right optional dependency group in `pyproject.toml`.
- When implementing a new backend, make it return `ModeSolverResult`.
- If the backend cannot support a requested feature, raise a clear `NotImplementedError` rather than silently ignoring it.

## Author/contact note

Do not assume that `dpastor@dcom.upv.es` is the project contact or maintainer. The project is currently being developed with Camilo Cano. Use neutral package metadata such as:

```toml
authors = [
    { name = "UPVfab" }
]
```

Additional authors/maintainers can be added explicitly later.
