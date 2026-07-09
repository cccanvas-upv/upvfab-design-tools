"""Eigenmode-expansion propagation tools."""

from .overlap import (
    modal_excitation_coefficients,
    normalize_profile,
    normalize_profiles,
    profile_overlap,
)
from .results import EMEPropagationResult
from .solver import propagate_modes
from .visualization import plot_propagation

__all__ = [
    "EMEPropagationResult",
    "modal_excitation_coefficients",
    "normalize_profile",
    "normalize_profiles",
    "plot_propagation",
    "profile_overlap",
    "propagate_modes",
]
