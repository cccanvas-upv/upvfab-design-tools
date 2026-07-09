from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from upvfab_design_tools.modesolver.results import ModeSolverResult


@dataclass(frozen=True)
class EMEPropagationResult:
    """Modal amplitudes propagated through one uniform section.

    ``modal_amplitudes[i, m]`` is the complex amplitude of mode ``m`` at
    ``z_um[i]``. Modal powers are meaningful when the supplied modes and
    initial amplitudes use a consistent power normalization.
    """

    mode_result: ModeSolverResult
    z_um: np.ndarray
    modal_amplitudes: np.ndarray

    def __post_init__(self) -> None:
        z_um = np.array(self.z_um, dtype=float, copy=True)
        modal_amplitudes = np.array(
            self.modal_amplitudes,
            dtype=np.complex128,
            copy=True,
        )

        if z_um.ndim != 1:
            raise ValueError("z_um must be a one-dimensional array.")

        if z_um.size == 0:
            raise ValueError("z_um must contain at least one position.")

        if not np.all(np.isfinite(z_um)):
            raise ValueError("z_um must contain only finite values.")

        if np.any(np.diff(z_um) < 0):
            raise ValueError("z_um must be ordered from low to high values.")

        expected_shape = (z_um.size, len(self.mode_result))
        if modal_amplitudes.shape != expected_shape:
            raise ValueError(
                "modal_amplitudes must have shape "
                f"{expected_shape}, got {modal_amplitudes.shape}."
            )

        z_um.setflags(write=False)
        modal_amplitudes.setflags(write=False)
        object.__setattr__(self, "z_um", z_um)
        object.__setattr__(self, "modal_amplitudes", modal_amplitudes)

    @property
    def final_amplitudes(self) -> np.ndarray:
        """Complex modal amplitudes at the end of the section."""

        return self.modal_amplitudes[-1]

    @property
    def modal_powers(self) -> np.ndarray:
        """Relative modal powers along the section."""

        return np.abs(self.modal_amplitudes) ** 2

    @property
    def total_modal_power(self) -> np.ndarray:
        """Sum of relative modal powers at each longitudinal position."""

        return np.sum(self.modal_powers, axis=1)

    def reconstruct_field(self, field_profiles: np.ndarray) -> np.ndarray:
        """Reconstruct the sampled transverse field along the section.

        Parameters
        ----------
        field_profiles:
            Complex array with shape ``(num_modes, num_x_points)``. Every row
            must contain the same field component and use the same transverse
            sampling positions.

        Returns
        -------
        np.ndarray
            Complex field with shape ``(num_z_points, num_x_points)``.
        """

        profiles = np.asarray(field_profiles, dtype=np.complex128)

        if profiles.ndim != 2:
            raise ValueError("field_profiles must be a two-dimensional array.")

        if profiles.shape[0] != len(self.mode_result):
            raise ValueError(
                "field_profiles must contain one row per mode: "
                f"expected {len(self.mode_result)}, got {profiles.shape[0]}."
            )

        if profiles.shape[1] == 0:
            raise ValueError("field_profiles must contain at least one x point.")

        if not np.all(np.isfinite(profiles)):
            raise ValueError("field_profiles must contain only finite values.")

        return self.modal_amplitudes @ profiles

    def field_intensity(self, field_profiles: np.ndarray) -> np.ndarray:
        """Return ``abs(E)**2`` for reconstructed transverse field profiles."""

        return np.abs(self.reconstruct_field(field_profiles)) ** 2
