from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Mode:
    """
    Generic optical mode.

    This class is backend-independent at the API level, but it can keep
    the original backend-specific mode object in `raw`.
    """

    index: int
    neff: complex
    wavelength_um: float
    te_fraction: float | None = None
    tm_fraction: float | None = None
    backend: str = ""
    raw: Any = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def beta(self) -> complex:
        """
        Propagation constant in rad/um.
        """

        return 2 * np.pi * self.neff / self.wavelength_um

    @property
    def polarization(self) -> str:
        """
        Return a simple TE/TM/mixed label based on modal fractions.
        """

        if self.te_fraction is None or self.tm_fraction is None:
            return "unknown"

        if self.te_fraction > 0.5:
            return "TE"

        if self.tm_fraction > 0.5:
            return "TM"

        return "mixed"

    def overlap(self, other: Mode) -> complex:
        """
        Calculate modal overlap with another mode.

        For now this delegates to the backend object if available.
        This works directly for FEMWELL modes because they expose
        calculate_overlap().
        """

        if self.raw is None:
            raise ValueError("This mode does not contain a raw backend mode.")

        if other.raw is None:
            raise ValueError("The other mode does not contain a raw backend mode.")

        if not hasattr(self.raw, "calculate_overlap"):
            raise NotImplementedError(
                f"Overlap is not implemented for backend '{self.backend}'."
            )

        return self.raw.calculate_overlap(other.raw)


@dataclass(frozen=True)
class ModeSolverResult:
    """
    Generic result returned by a mode solver.
    """

    modes: tuple[Mode, ...]
    wavelength_um: float
    backend: str
    cross_section: Any = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __len__(self) -> int:
        return len(self.modes)

    def __getitem__(self, item: int) -> Mode:
        return self.modes[item]

    @property
    def neffs(self) -> np.ndarray:
        return np.array([mode.neff for mode in self.modes], dtype=np.complex128)

    @property
    def te_modes(self) -> tuple[Mode, ...]:
        return tuple(mode for mode in self.modes if mode.polarization == "TE")

    @property
    def tm_modes(self) -> tuple[Mode, ...]:
        return tuple(mode for mode in self.modes if mode.polarization == "TM")

    @property
    def raw_modes(self) -> tuple[Any, ...]:
        return tuple(mode.raw for mode in self.modes)