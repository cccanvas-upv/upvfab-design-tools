from __future__ import annotations

from typing import Literal

import numpy as np
from skfem import ElementDG, ElementTriP1, ElementVector

from .results import Mode, ModeSolverResult


FieldComponent = Literal["auto", "Ex", "Ey"]


def sample_mode_profile(
    mode: Mode,
    *,
    x_um: np.ndarray,
    z_um: float = 0.0,
    field_component: FieldComponent = "auto",
) -> np.ndarray:
    """Sample one FEMWELL transverse mode along a horizontal x cut.

    ``field_component="auto"`` selects Ex for a TE-dominant mode and Ey for
    a TM-dominant mode.
    """

    x_um = _validate_sample_positions(x_um)
    if not np.isfinite(z_um):
        raise ValueError("z_um must be finite.")

    raw_mode = get_raw_mode(mode)
    et_x, et_x_basis, et_y, et_y_basis = get_transverse_fields(raw_mode)
    selected_component = select_field_component(mode, field_component)

    if selected_component == "Ex":
        field = et_x
        basis = et_x_basis
    else:
        field = et_y
        basis = et_y_basis

    query_points = np.vstack([x_um, np.full(x_um.size, z_um)])
    return np.asarray(basis.probes(query_points) @ field, dtype=np.complex128)


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
    """

    if len(result) == 0:
        raise ValueError("ModeSolverResult contains no modes to sample.")

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
    """Return the FEMWELL object wrapped by a generic mode."""

    if mode.backend and mode.backend != "femwell":
        raise NotImplementedError(
            f"Field sampling is not implemented for backend '{mode.backend}'."
        )

    if mode.raw is None:
        raise ValueError("Mode does not contain a raw backend mode.")

    return mode.raw


def get_transverse_fields(raw_mode):
    """Extract FEMWELL Ex and Ey fields and their plotting bases."""

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


def _validate_sample_positions(x_um: np.ndarray) -> np.ndarray:
    x_um = np.asarray(x_um, dtype=float)

    if x_um.ndim != 1 or x_um.size == 0:
        raise ValueError("x_um must be a non-empty one-dimensional array.")

    if not np.all(np.isfinite(x_um)) or np.any(np.diff(x_um) <= 0):
        raise ValueError("x_um must be finite and strictly increasing.")

    return x_um
