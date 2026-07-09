from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from .results import EMEPropagationResult

if TYPE_CHECKING:
    from upvfab_design_tools.modesolver.results import ModeSolverResult


def propagate_modes(
    mode_result: ModeSolverResult,
    *,
    initial_amplitudes: Sequence[complex] | np.ndarray,
    length_um: float,
    dz_um: float,
) -> EMEPropagationResult:
    """Propagate a modal superposition through a uniform section.

    Each mode evolves independently according to
    ``a_m(z) = a_m(0) * exp(-1j * beta_m * z)``. The caller defines the
    excitation through ``initial_amplitudes``; no input mode or polarization
    is assumed by this function.

    Parameters
    ----------
    mode_result:
        Generic modes of the uniform propagation section.
    initial_amplitudes:
        Complex excitation coefficient for each mode in ``mode_result``.
    length_um:
        Propagation length in micrometers. Zero is allowed.
    dz_um:
        Maximum longitudinal sampling step in micrometers. The final sample
        is always placed exactly at ``length_um``.
    """

    if len(mode_result) == 0:
        raise ValueError("mode_result must contain at least one mode.")

    if not np.isfinite(length_um) or length_um < 0:
        raise ValueError("length_um must be finite and non-negative.")

    if not np.isfinite(dz_um) or dz_um <= 0:
        raise ValueError("dz_um must be finite and positive.")

    amplitudes_0 = np.asarray(initial_amplitudes, dtype=np.complex128)
    if amplitudes_0.ndim != 1:
        raise ValueError("initial_amplitudes must be one-dimensional.")

    if amplitudes_0.size != len(mode_result):
        raise ValueError(
            "initial_amplitudes must contain one value per mode: "
            f"expected {len(mode_result)}, got {amplitudes_0.size}."
        )

    if not np.all(np.isfinite(amplitudes_0)):
        raise ValueError("initial_amplitudes must contain only finite values.")

    z_um = _longitudinal_positions(length_um=length_um, dz_um=dz_um)
    beta = np.array([mode.beta for mode in mode_result.modes], dtype=np.complex128)
    if not np.all(np.isfinite(beta)):
        raise ValueError("All modal propagation constants must be finite.")

    phase = np.exp(-1j * z_um[:, np.newaxis] * beta[np.newaxis, :])
    modal_amplitudes = amplitudes_0[np.newaxis, :] * phase

    return EMEPropagationResult(
        mode_result=mode_result,
        z_um=z_um,
        modal_amplitudes=modal_amplitudes,
    )


def _longitudinal_positions(*, length_um: float, dz_um: float) -> np.ndarray:
    if length_um == 0:
        return np.array([0.0])

    full_steps = int(np.floor(length_um / dz_um))
    z_um = np.arange(full_steps + 1, dtype=float) * dz_um

    endpoint_tolerance = (
        10
        * np.finfo(float).eps
        * max(abs(length_um), abs(z_um[-1]), np.finfo(float).tiny)
    )
    if abs(z_um[-1] - length_um) <= endpoint_tolerance:
        z_um[-1] = length_um
        return z_um

    return np.append(z_um, length_um)
