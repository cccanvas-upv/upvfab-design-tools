"""Eigenmode-expansion propagation tools."""

from .overlap import (
    modal_excitation_coefficients,
    modal_overlap_matrix,
    normalize_profile,
    normalize_profiles,
    profile_overlap,
    shift_profile,
)
from .results import EMEPropagationResult
from .solver import propagate_modes
from .visualization import plot_propagation

__all__ = [
    "EMEPropagationResult",
    "modal_excitation_coefficients",
    "modal_overlap_matrix",
    "normalize_profile",
    "normalize_profiles",
    "plot_propagation",
    "profile_overlap",
    "propagate_modes",
    "shift_profile",
]
