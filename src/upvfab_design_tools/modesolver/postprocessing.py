from __future__ import annotations

from typing import Literal

import numpy as np

from .results import Mode, ModeSolverResult


FieldComponent = Literal["auto", "Ex", "Ey"]
SUPPORTED_PROFILE_SAMPLING_BACKENDS = ("femwell", "tidy3d")


def sample_mode_profile(
    mode: Mode,
    *,
    x_um: np.ndarray,
    z_um: float = 0.0,
    field_component: FieldComponent = "auto",
) -> np.ndarray:
    """Sample one mode field profile along a horizontal x cut.

    ``field_component="auto"`` selects Ex for a TE-dominant mode and Ey for
    a TM-dominant mode. The returned array is backend-independent and is the
    scalar profile consumed by EME.

    Backend-specific raw field interpretation is intentionally confined to this
    module. Future mode-solver backends, such as Tidy3D, should add their
    sampling adapter here without changing EME.
    """

    x_um = _validate_sample_positions(x_um)
    if not np.isfinite(z_um):
        raise ValueError("z_um must be finite.")

    backend = _mode_backend(mode)
    selected_component = select_field_component(mode, field_component)

    if backend == "femwell":
        return _sample_femwell_mode_profile(
            mode,
            x_um=x_um,
            z_um=z_um,
            field_component=selected_component,
        )

    if backend == "tidy3d":
        return _sample_tidy3d_mode_profile(
            mode,
            x_um=x_um,
            z_um=z_um,
            field_component=selected_component,
        )

    raise NotImplementedError(
        "Field-profile sampling is not implemented for backend "
        f"'{backend}'. Supported backends: "
        f"{', '.join(SUPPORTED_PROFILE_SAMPLING_BACKENDS)}. "
        "EME only requires sampled scalar profiles, so add the backend-specific "
        "field extraction adapter in modesolver.postprocessing."
    )


def _sample_femwell_mode_profile(
    mode: Mode,
    *,
    x_um: np.ndarray,
    z_um: float,
    field_component: Literal["Ex", "Ey"],
) -> np.ndarray:
    """Sample one FEMWELL transverse mode along a horizontal x cut."""

    raw_mode = get_raw_mode(mode)
    et_x, et_x_basis, et_y, et_y_basis = get_transverse_fields(raw_mode)

    if field_component == "Ex":
        field = et_x
        basis = et_x_basis
    else:
        field = et_y
        basis = et_y_basis

    query_points = np.vstack([x_um, np.full(x_um.size, z_um)])
    return np.asarray(basis.probes(query_points) @ field, dtype=np.complex128)


def _sample_tidy3d_mode_profile(
    mode: Mode,
    *,
    x_um: np.ndarray,
    z_um: float,
    field_component: Literal["Ex", "Ey"],
) -> np.ndarray:
    """Sample one Tidy3D mode along a horizontal UPVfab x cut.

    UPVfab's vertical coordinate is named ``z_um``. In the Tidy3D mode-solver
    representation used by this package, that coordinate maps to Tidy3D ``y``.
    """

    raw = mode.raw

    if raw is None:
        raise ValueError("Mode does not contain raw Tidy3D mode data.")

    if not all(hasattr(raw, attribute) for attribute in ("data", "mode_index")):
        raise TypeError(
            "Expected raw Tidy3D mode data with 'data' and 'mode_index' "
            "attributes."
        )

    frequency_index = getattr(raw, "frequency_index", 0)
    data_array = raw.data.field_components[field_component].isel(
        f=frequency_index,
        mode_index=raw.mode_index,
    )

    if "z" in data_array.dims:
        data_array = data_array.isel(z=0)

    sampled = data_array.interp(x=x_um, y=z_um)
    values = np.asarray(sampled.values, dtype=np.complex128)

    if values.shape != (x_um.size,):
        values = np.ravel(values)

    if values.shape != (x_um.size,):
        raise ValueError(
            "Unexpected Tidy3D sampled profile shape: "
            f"expected ({x_um.size},), got {values.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Tidy3D field interpolation produced non-finite values. Check that "
            "x_um and z_um lie inside the computed mode-solver plane."
        )

    return values


def sample_mode_profiles(
    result: ModeSolverResult,
    *,
    x_um: np.ndarray,
    z_um: float = 0.0,
    field_component: FieldComponent = "auto",
) -> np.ndarray:
    """Sample several modes on one common transverse coordinate grid.

    Automatic selection is accepted only when it resolves to the same field
    component for every mode. Coherent reconstruction cannot mix Ex and Ey as
    if they represented one scalar field.

    This function is the intended generic bridge between mode-solver results
    and EME: it converts backend-specific raw modes into a plain complex array
    with shape ``(num_modes, len(x_um))``.
    """

    if len(result) == 0:
        raise ValueError("ModeSolverResult contains no modes to sample.")

    mode_backends = {_mode_backend(mode) for mode in result.modes}
    if len(mode_backends) != 1:
        raise ValueError(
            "All modes in a ModeSolverResult must use the same backend for "
            f"profile sampling, got {sorted(mode_backends)}."
        )

    selected_components = tuple(
        select_field_component(mode, field_component) for mode in result.modes
    )
    if len(set(selected_components)) != 1:
        raise ValueError(
            "Automatic selection produced a mixture of Ex and Ey profiles. "
            "Select one polarization family before EME propagation or set "
            "field_component explicitly."
        )

    return np.array(
        [
            sample_mode_profile(
                mode,
                x_um=x_um,
                z_um=z_um,
                field_component=selected_components[0],
            )
            for mode in result.modes
        ],
        dtype=np.complex128,
    )


def get_raw_mode(mode: Mode):
    """Return the raw backend object wrapped by a generic FEMWELL mode."""

    backend = _mode_backend(mode)
    if backend != "femwell":
        raise NotImplementedError(
            f"FEMWELL raw-mode extraction cannot handle backend '{backend}'."
        )

    if mode.raw is None:
        raise ValueError("Mode does not contain a raw FEMWELL backend mode.")

    return mode.raw


def get_transverse_fields(raw_mode):
    """Extract FEMWELL Ex and Ey fields and their plotting bases."""

    try:
        from skfem import ElementDG, ElementTriP1, ElementVector
    except ImportError as exc:
        raise ImportError(
            "Sampling FEMWELL mode fields requires scikit-fem. Install the "
            "modesolver optional dependencies before sampling FEMWELL modes."
        ) from exc

    if not hasattr(raw_mode, "basis") or not hasattr(raw_mode, "E"):
        raise TypeError(
            "Expected a FEMWELL raw mode with 'basis' and 'E' attributes."
        )

    (et, et_basis), _ = raw_mode.basis.split(raw_mode.E)
    plot_basis = et_basis.with_element(
        ElementVector(ElementDG(ElementTriP1()))
    )
    et_xy = plot_basis.project(
        et_basis.interpolate(et),
        dtype=np.complex128,
    )
    (et_x, et_x_basis), (et_y, et_y_basis) = plot_basis.split(et_xy)
    return et_x, et_x_basis, et_y, et_y_basis


def select_field_component(
    mode: Mode,
    field_component: FieldComponent,
) -> Literal["Ex", "Ey"]:
    """Resolve an explicit or polarization-dependent field component."""

    if field_component in {"Ex", "Ey"}:
        return field_component

    if field_component != "auto":
        raise ValueError("field_component must be 'auto', 'Ex', or 'Ey'.")

    if mode.te_fraction is None or mode.tm_fraction is None:
        raise ValueError(
            "Automatic field-component selection requires both TE and TM "
            "fractions. Set field_component explicitly to 'Ex' or 'Ey'."
        )

    if mode.te_fraction > mode.tm_fraction:
        return "Ex"

    if mode.tm_fraction > mode.te_fraction:
        return "Ey"

    raise ValueError(
        "Automatic field-component selection is ambiguous because the TE and "
        "TM fractions are equal. Set field_component explicitly to 'Ex' or 'Ey'."
    )


def _mode_backend(mode: Mode) -> str:
    backend = mode.backend.strip().lower()

    if not backend:
        raise ValueError(
            "Mode.backend is empty. Field-profile sampling requires modes "
            "created by a backend-aware mode solver."
        )

    return backend


def _validate_sample_positions(x_um: np.ndarray) -> np.ndarray:
    x_um = np.asarray(x_um, dtype=float)

    if x_um.ndim != 1 or x_um.size == 0:
        raise ValueError("x_um must be a non-empty one-dimensional array.")

    if not np.all(np.isfinite(x_um)) or np.any(np.diff(x_um) <= 0):
        raise ValueError("x_um must be finite and strictly increasing.")

    return x_um
