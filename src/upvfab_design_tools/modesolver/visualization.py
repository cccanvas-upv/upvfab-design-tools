from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .postprocessing import (
    FieldComponent,
    get_raw_mode,
    get_transverse_fields,
    select_field_component,
)
from .results import Mode, ModeSolverResult


FieldPart = Literal["mag", "phase", "real"]


def plot_mode(
    mode: Mode,
    *,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    field_component: FieldComponent = "auto",
    field_part: FieldPart = "real",
    show_mesh: bool = True,
    ax=None,
):
    """
    Plot one transverse field component of a mode.

    Parameters
    ----------
    mode:
        Generic Mode object returned by the modesolver.
    xlim:
        Optional x-axis limits in micrometers.
    zlim:
        Optional z-axis limits in micrometers.
    field_component:
        Field component to plot. ``"auto"`` selects Ex for a TE-dominant
        mode and Ey for a TM-dominant mode using the modal fractions.
    field_part:
        Field quantity to display: magnitude, phase, or real part.
    show_mesh:
        If True, draw FEMWELL mesh boundaries.
    ax:
        Optional Matplotlib axis.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    raw_mode = get_raw_mode(mode)

    et_x, et_x_basis, et_y, et_y_basis = get_transverse_fields(raw_mode)

    selected_component = select_field_component(mode, field_component)

    if selected_component == "Ex":
        field = et_x
        basis = et_x_basis
    else:
        field = et_y
        basis = et_y_basis

    plot_field, cmap, vmin, vmax = _prepare_field_for_plot(field, field_part)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if show_mesh:
        raw_mode.basis.mesh.draw(ax=ax, boundaries_only=True)

        for subdomain in raw_mode.basis.mesh.subdomains.keys() - {"gmsh:bounding_entities"}:
            raw_mode.basis.mesh.restrict(subdomain).draw(
                ax=ax,
                boundaries_only=True,
            )

    basis.plot(
        plot_field,
        shading="gouraud",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        colorbar=True,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("z [µm]")

    title = _mode_title(mode, selected_component, field_part)
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if zlim is not None:
        ax.set_ylim(zlim)

    return fig, ax


def plot_mode_ex_ey(
    mode: Mode,
    *,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    field_part: FieldPart = "real",
    show_mesh: bool = True,
):
    """
    Plot Ex and Ey transverse field components of a mode.

    ``field_part`` selects whether magnitude, phase, or real part is shown.
    """

    raw_mode = get_raw_mode(mode)

    et_x, et_x_basis, et_y, et_y_basis = get_transverse_fields(raw_mode)

    fig, axs = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        sharex=True,
        sharey=True,
    )

    fields = [
        ("Ex", et_x, et_x_basis),
        ("Ey", et_y, et_y_basis),
    ]

    for ax, (component_name, field, basis) in zip(axs, fields):
        plot_field, cmap, vmin, vmax = _prepare_field_for_plot(field, field_part)

        if show_mesh:
            raw_mode.basis.mesh.draw(ax=ax, boundaries_only=True)

            for subdomain in raw_mode.basis.mesh.subdomains.keys() - {"gmsh:bounding_entities"}:
                raw_mode.basis.mesh.restrict(subdomain).draw(
                    ax=ax,
                    boundaries_only=True,
                )

        basis.plot(
            plot_field,
            shading="gouraud",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar=True,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [µm]")
        ax.set_title(_mode_title(mode, component_name, field_part))

        if xlim is not None:
            ax.set_xlim(xlim)

        if zlim is not None:
            ax.set_ylim(zlim)

    axs[0].set_ylabel("z [µm]")

    fig.tight_layout()

    return fig, axs


def plot_modes_grid(
    result: ModeSolverResult,
    *,
    field_component: FieldComponent = "auto",
    field_part: FieldPart = "real",
    max_modes: int | None = None,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    show_mesh: bool = True,
):
    """
    Plot several modes in a vertical grid.

    This is useful for quickly checking all modes returned by the solver.
    By default, each mode uses Ex when it is TE-dominant and Ey when it is
    TM-dominant.
    """

    modes = result.modes

    if max_modes is not None:
        modes = modes[:max_modes]

    if len(modes) == 0:
        raise ValueError("ModeSolverResult contains no modes to plot.")

    fig, axs = plt.subplots(
        len(modes),
        1,
        figsize=(6, 3.2 * len(modes)),
        sharex=True,
        sharey=True,
    )

    if len(modes) == 1:
        axs = [axs]

    for ax, mode in zip(axs, modes):
        plot_mode(
            mode,
            field_component=field_component,
            field_part=field_part,
            xlim=xlim,
            zlim=zlim,
            show_mesh=show_mesh,
            ax=ax,
        )

    fig.tight_layout()

    return fig, axs


def plot_effective_indices(result: ModeSolverResult, ax=None):
    """
    Plot effective index of the modes returned by a solver.
    """

    if len(result.modes) == 0:
        raise ValueError("ModeSolverResult contains no modes.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.figure

    mode_indices = [mode.index for mode in result.modes]
    neffs = [np.real(mode.neff) for mode in result.modes]

    ax.plot(mode_indices, neffs, marker="o")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Re(n_eff)")
    ax.set_title(f"Effective indices - {result.backend}")
    ax.grid(True)

    return fig, ax


def plot_epsilon(
    result: ModeSolverResult,
    *,
    ax=None,
):
    """
    Plot the relative permittivity distribution used by FEMWELL.

    Requires the result metadata to contain 'basis' and 'epsilon'.
    """

    basis = result.metadata.get("basis")
    epsilon = result.metadata.get("epsilon")

    if basis is None or epsilon is None:
        raise ValueError(
            "Result metadata does not contain 'basis' and 'epsilon'. "
            "This plot is currently available for FEMWELL results."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    basis.plot(
        epsilon,
        ax=ax,
        colorbar=True,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("z [µm]")
    ax.set_title("Relative permittivity")

    return fig, ax


def save_figure(fig, filename: str | Path, *, dpi: int = 300) -> Path:
    """
    Save a Matplotlib figure and return the output path.
    """

    path = Path(filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _prepare_field_for_plot(
    field,
    field_part: FieldPart,
) -> tuple[np.ndarray, str, float | None, float | None]:
    """Transform a complex field and choose suitable plotting limits."""

    field_array = np.asarray(field)

    if field_part == "mag":
        plot_field = np.abs(field_array)
        maximum = _finite_absolute_max(plot_field)
        return plot_field, "viridis", 0.0, maximum

    if field_part == "phase":
        return np.angle(field_array), "twilight", -np.pi, np.pi

    if field_part == "real":
        plot_field = np.real(field_array)
        maximum = _finite_absolute_max(plot_field)
        minimum = -maximum if maximum is not None else None
        return plot_field, "bwr", minimum, maximum

    raise ValueError("field_part must be 'mag', 'phase', or 'real'.")


def _finite_absolute_max(field: np.ndarray) -> float | None:
    finite_values = np.abs(field[np.isfinite(field)])

    if finite_values.size == 0:
        return None

    maximum = float(np.max(finite_values))
    return maximum if maximum > 0 else None


def _mode_title(mode: Mode, component: str, field_part: FieldPart) -> str:
    neff = np.real(mode.neff)
    field_label = {
        "mag": "magnitude",
        "phase": "phase",
        "real": "real",
    }[field_part]

    if mode.te_fraction is None or mode.tm_fraction is None:
        return (
            f"Mode {mode.index} | {component} {field_label} | "
            f"n_eff = {neff:.6f}"
        )

    return (
        f"Mode {mode.index} | {component} {field_label} | "
        f"n_eff = {neff:.6f} | "
        f"TE = {mode.te_fraction:.3f}, TM = {mode.tm_fraction:.3f}"
    )
