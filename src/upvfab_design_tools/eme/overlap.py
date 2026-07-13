from __future__ import annotations

import numpy as np


def normalize_profile(profile: np.ndarray, x_um: np.ndarray) -> np.ndarray:
    """Normalize a sampled scalar field profile to unit integrated intensity."""

    x_um, profile = _validated_profile(profile=profile, x_um=x_um)
    norm_squared = float(np.real(np.trapezoid(np.abs(profile) ** 2, x_um)))

    if norm_squared <= 0:
        raise ValueError("Cannot normalize a profile with zero integrated intensity.")

    return profile / np.sqrt(norm_squared)


def normalize_profiles(profiles: np.ndarray, x_um: np.ndarray) -> np.ndarray:
    """Normalize each row of a modal-profile array independently."""

    profiles = np.asarray(profiles, dtype=np.complex128)
    if profiles.ndim != 2:
        raise ValueError("profiles must be a two-dimensional array.")

    if profiles.shape[0] == 0:
        raise ValueError("profiles must contain at least one mode.")

    return np.array(
        [normalize_profile(profile, x_um) for profile in profiles],
        dtype=np.complex128,
    )


def shift_profile(
    profile: np.ndarray,
    x_um: np.ndarray,
    shift_um: float,
    *,
    fill_value: complex = 0.0,
) -> np.ndarray:
    """Return a laterally shifted copy of a sampled scalar field profile.

    Positive ``shift_um`` moves the profile toward positive ``x`` by evaluating
    ``profile(x - shift_um)`` on the original grid. Values shifted outside the
    sampled window are filled with ``fill_value``.
    """

    x_um, profile = _validated_profile(profile=profile, x_um=x_um)
    shift_um = float(shift_um)
    fill_value = complex(fill_value)

    if not np.isfinite(shift_um):
        raise ValueError("shift_um must be finite.")

    if not np.isfinite(fill_value):
        raise ValueError("fill_value must be finite.")

    shifted_x_um = x_um - shift_um
    shifted_real = np.interp(
        shifted_x_um,
        x_um,
        profile.real,
        left=fill_value.real,
        right=fill_value.real,
    )
    shifted_imag = np.interp(
        shifted_x_um,
        x_um,
        profile.imag,
        left=fill_value.imag,
        right=fill_value.imag,
    )

    return shifted_real + 1j * shifted_imag


def profile_overlap(
    input_profile: np.ndarray,
    mode_profile: np.ndarray,
    x_um: np.ndarray,
) -> complex:
    """Return the normalized scalar overlap ``<mode|input>``.

    This is a one-dimensional scalar approximation. Both profiles are
    independently normalized before evaluating the complex inner product.
    """

    input_normalized = normalize_profile(input_profile, x_um)
    mode_normalized = normalize_profile(mode_profile, x_um)
    return complex(
        np.trapezoid(np.conjugate(mode_normalized) * input_normalized, x_um)
    )


def modal_excitation_coefficients(
    input_profile: np.ndarray,
    mode_profiles: np.ndarray,
    x_um: np.ndarray,
) -> np.ndarray:
    """Project one input profile onto several sampled modal profiles.

    Parameters
    ----------
    input_profile:
        Complex input field sampled at ``x_um``.
    mode_profiles:
        Complex array with shape ``(num_modes, len(x_um))``.
    x_um:
        Strictly increasing transverse coordinates in micrometers.
    """

    x_um, input_profile = _validated_profile(
        profile=input_profile,
        x_um=x_um,
    )
    profiles = np.asarray(mode_profiles, dtype=np.complex128)

    if profiles.ndim != 2 or profiles.shape[1] != x_um.size:
        raise ValueError(
            "mode_profiles must have shape "
            f"(num_modes, {x_um.size}), got {profiles.shape}."
        )

    if profiles.shape[0] == 0:
        raise ValueError("mode_profiles must contain at least one mode.")

    if not np.all(np.isfinite(profiles)):
        raise ValueError("mode_profiles must contain only finite values.")

    return np.array(
        [
            profile_overlap(input_profile, mode_profile, x_um)
            for mode_profile in profiles
        ],
        dtype=np.complex128,
    )


def modal_overlap_matrix(
    source_profiles: np.ndarray,
    target_profiles: np.ndarray,
    x_um: np.ndarray,
) -> np.ndarray:
    """Return the normalized scalar overlap matrix between two modal bases.

    The convention is:

    ``target_amplitudes = C @ source_amplitudes``

    where ``C[target_index, source_index] = <target_mode|source_mode>``.
    Source and target profiles are normalized row-by-row before computing
    overlaps.
    """

    x_um = _validated_x_grid(x_um)
    source_profiles = _validated_profile_matrix(
        profiles=source_profiles,
        x_um=x_um,
        name="source_profiles",
    )
    target_profiles = _validated_profile_matrix(
        profiles=target_profiles,
        x_um=x_um,
        name="target_profiles",
    )

    source_normalized = normalize_profiles(source_profiles, x_um)
    target_normalized = normalize_profiles(target_profiles, x_um)
    integration_weights = _trapezoid_weights(x_um)

    return target_normalized.conj() @ (source_normalized * integration_weights).T


def _validated_profile(
    *,
    profile: np.ndarray,
    x_um: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_um = _validated_x_grid(x_um)
    profile = np.asarray(profile, dtype=np.complex128)

    if profile.ndim != 1 or profile.size != x_um.size:
        raise ValueError(
            f"profile must have shape ({x_um.size},), got {profile.shape}."
        )

    if not np.all(np.isfinite(profile)):
        raise ValueError("profile must contain only finite values.")

    return x_um, profile


def _validated_profile_matrix(
    *,
    profiles: np.ndarray,
    x_um: np.ndarray,
    name: str,
) -> np.ndarray:
    profiles = np.asarray(profiles, dtype=np.complex128)

    if profiles.ndim != 2 or profiles.shape[1] != x_um.size:
        raise ValueError(
            f"{name} must have shape (num_modes, {x_um.size}), "
            f"got {profiles.shape}."
        )

    if profiles.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one mode.")

    if not np.all(np.isfinite(profiles)):
        raise ValueError(f"{name} must contain only finite values.")

    return profiles


def _validated_x_grid(x_um: np.ndarray) -> np.ndarray:
    x_um = np.asarray(x_um, dtype=float)

    if x_um.ndim != 1 or x_um.size < 2:
        raise ValueError("x_um must contain at least two positions.")

    if not np.all(np.isfinite(x_um)) or np.any(np.diff(x_um) <= 0):
        raise ValueError("x_um must be finite and strictly increasing.")

    return x_um


def _trapezoid_weights(x_um: np.ndarray) -> np.ndarray:
    weights = np.empty_like(x_um, dtype=float)
    dx = np.diff(x_um)

    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]

    if x_um.size > 2:
        weights[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    return weights
