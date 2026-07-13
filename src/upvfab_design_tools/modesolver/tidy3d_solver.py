from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.materials import Material
from upvfab_design_tools.tidy3d_plugin.conversion import (
    cross_section_to_tidy3d_simulation,
    mode_solver_plane,
)

from .base import BaseModeSolver
from .results import Mode, ModeSolverResult


@dataclass(frozen=True)
class Tidy3DModeRaw:
    """Raw Tidy3D mode data needed for backend-specific post-processing."""

    data: Any
    mode_index: int
    frequency_index: int = 0
    cross_section: CrossSection | None = None


@dataclass(frozen=True)
class Tidy3DModeSolver(BaseModeSolver):
    """Tidy3D mode-solver backend for UPVfab cross-sections.

    The input geometry is always a backend-neutral
    :class:`upvfab_design_tools.core.cross_section.CrossSection`. It is
    converted to a Tidy3D simulation only inside this backend.
    """

    backend: str = "tidy3d"

    min_steps_per_wvl: int = 20
    propagation_length_um: float = 1.0
    precision: str = "double"
    filter_guided: bool = True
    guided_tolerance: float = 1e-2
    reference_material: Material | None = None
    target_neff: float | None = None
    filter_pol: str | None = None
    group_index_step: bool | float = False
    run_time_s: float = 1e-12

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

        td, ModeSolver = _import_tidy3d_mode_solver()
        simulation = cross_section_to_tidy3d_simulation(
            cross_section,
            wavelength_um=wavelength_um,
            propagation_length_um=self.propagation_length_um,
            min_steps_per_wvl=self.min_steps_per_wvl,
            run_time_s=self.run_time_s,
        )
        mode_spec = td.ModeSpec(
            num_modes=num_modes,
            target_neff=self._target_neff(cross_section, wavelength_um),
            precision=self.precision,
            filter_pol=self.filter_pol,
            group_index_step=self.group_index_step,
        )
        mode_solver = ModeSolver(
            simulation=simulation,
            plane=mode_solver_plane(cross_section),
            mode_spec=mode_spec,
            freqs=[td.C_0 / wavelength_um],
        )
        data = mode_solver.solve()
        modes = tuple(
            Mode(
                index=mode_index,
                neff=complex(_mode_neff(data, mode_index)),
                wavelength_um=wavelength_um,
                te_fraction=_mode_te_fraction(data, mode_index),
                tm_fraction=_mode_tm_fraction(data, mode_index),
                backend=self.backend,
                raw=Tidy3DModeRaw(
                    data=data,
                    mode_index=mode_index,
                    cross_section=cross_section,
                ),
            )
            for mode_index in range(num_modes)
        )

        if self.filter_guided:
            reference_material = (
                self.reference_material or cross_section.background_material
            )
            reference_index = float(np.real(reference_material.n(wavelength_um)))
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
                "simulation": simulation,
                "mode_solver": mode_solver,
                "mode_solver_data": data,
                "mode_spec": mode_spec,
                "coordinate_mapping": {
                    "upvfab_x": "tidy3d_x",
                    "upvfab_z_vertical": "tidy3d_y",
                    "propagation": "tidy3d_z",
                },
            },
        )

    def _target_neff(
        self,
        cross_section: CrossSection,
        wavelength_um: float,
    ) -> float | None:
        if self.target_neff is not None:
            return self.target_neff

        material_indices = [
            float(np.real(material.n(wavelength_um)))
            for material in cross_section.materials().values()
        ]

        return max(material_indices)


def _mode_neff(data, mode_index: int) -> complex:
    return complex(np.asarray(data.n_eff.values)[0, mode_index])


def _mode_te_fraction(data, mode_index: int) -> float:
    ex_energy = _field_energy(data, "Ex", mode_index)
    ey_energy = _field_energy(data, "Ey", mode_index)
    total = ex_energy + ey_energy

    if total <= 0:
        return 0.0

    return float(ex_energy / total)


def _mode_tm_fraction(data, mode_index: int) -> float:
    ex_energy = _field_energy(data, "Ex", mode_index)
    ey_energy = _field_energy(data, "Ey", mode_index)
    total = ex_energy + ey_energy

    if total <= 0:
        return 0.0

    return float(ey_energy / total)


def _field_energy(data, field_component: str, mode_index: int) -> float:
    field = data.field_components[field_component].isel(
        f=0,
        mode_index=mode_index,
    )

    return float(np.sum(np.abs(field.values) ** 2))


def _import_tidy3d_mode_solver():
    try:
        import tidy3d as td
        from tidy3d.plugins.mode import ModeSolver
    except ImportError as exc:
        raise ImportError(
            "The Tidy3D mode solver requires the optional 'tidy3d' dependency. "
            "Install it with `uv sync --extra tidy3d`."
        ) from exc

    return td, ModeSolver
