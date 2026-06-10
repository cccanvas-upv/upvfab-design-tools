from __future__ import annotations

from abc import ABC, abstractmethod

from upvfab_design_tools.core.cross_section import CrossSection

from .results import ModeSolverResult


class BaseModeSolver(ABC):
    """
    Abstract base class for mode solvers.
    """

    backend: str

    @abstractmethod
    def solve(
        self,
        cross_section: CrossSection,
        wavelength_um: float,
        num_modes: int,
    ) -> ModeSolverResult:
        """
        Solve modes for a given cross-section.
        """