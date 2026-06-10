from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from skfem import ElementDG, ElementTriP1, ElementVector

from .results import Mode, ModeSolverResult


FieldComponent = Literal["Ex", "Ey"]


def _get_raw_mode(mode: Mode):
    """
    Return the backend-specific mode object.

    For FemwellModeSolver, this is the original FEMWELL mode object.
    """

    if isinstance(mode, Mode):
        if mode.raw is None:
            raise ValueError("Mode does not contain a raw backend mode.")

        return mode.raw

    return mode


def _get_transverse_fields(raw_mode):
    """
    Extract transverse electric field components from a FEMWELL mode.

    Returns
    -------
    et_x:
        x component of the transverse electric field.
    et_x_basis:
        basis associated with et_x.
    et_y:
        y component of the transverse electric field.
    et_y_basis:
        basis associated with et_y.
    """

    (et, et_basis), _ = raw_mode.basis.split(raw_mode.E)

    plot_basis = et_basis.with_element(
        ElementVector(
            ElementDG(
                ElementTriP1()
            )
        )
    )

    et_xy = plot_basis.project(et_basis.interpolate(et))
    (et_x, et_x_basis), (et_y, et_y_basis) = plot_basis.split(et_xy)

    return et_x, et_x_basis, et_y, et_y_basis


def plot_mode(
    mode: Mode,
    *,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    field_component: FieldComponent = "Ex",
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
        Field component to plot. Either "Ex" or "Ey".
    show_mesh:
        If True, draw FEMWELL mesh boundaries.
    ax:
        Optional Matplotlib axis.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    raw_mode = _get_raw_mode(mode)

    et_x, et_x_basis, et_y, et_y_basis = _get_transverse_fields(raw_mode)

    if field_component == "Ex":
        field = et_x
        basis = et_x_basis
    elif field_component == "Ey":
        field = et_y
        basis = et_y_basis
    else:
        raise ValueError("field_component must be either 'Ex' or 'Ey'.")

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

    vabs = np.nanmax(np.abs(field))

    if vabs == 0 or np.isnan(vabs):
        vmin = None
        vmax = None
    else:
        vmin = -vabs
        vmax = vabs

    basis.plot(
        field,
        shading="gouraud",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        cmap="bwr",
        colorbar=True,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("z [µm]")

    title = _mode_title(mode, field_component)
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
    show_mesh: bool = True,
):
    """
    Plot Ex and Ey transverse field components of a mode.
    """

    raw_mode = _get_raw_mode(mode)

    et_x, et_x_basis, et_y, et_y_basis = _get_transverse_fields(raw_mode)

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
        if show_mesh:
            raw_mode.basis.mesh.draw(ax=ax, boundaries_only=True)

            for subdomain in raw_mode.basis.mesh.subdomains.keys() - {"gmsh:bounding_entities"}:
                raw_mode.basis.mesh.restrict(subdomain).draw(
                    ax=ax,
                    boundaries_only=True,
                )

        vabs = np.nanmax(np.abs(field))

        if vabs == 0 or np.isnan(vabs):
            vmin = None
            vmax = None
        else:
            vmin = -vabs
            vmax = vabs

        basis.plot(
            field,
            shading="gouraud",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap="bwr",
            colorbar=True,
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [µm]")
        ax.set_title(_mode_title(mode, component_name))

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
    field_component: FieldComponent = "Ex",
    max_modes: int | None = None,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    show_mesh: bool = True,
):
    """
    Plot several modes in a vertical grid.

    This is useful for quickly checking all modes returned by the solver.
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


def _mode_title(mode: Mode, component: str) -> str:
    neff = np.real(mode.neff)

    if mode.te_fraction is None or mode.tm_fraction is None:
        return f"Mode {mode.index} | {component} | n_eff = {neff:.6f}"

    return (
        f"Mode {mode.index} | {component} | "
        f"n_eff = {neff:.6f} | "
        f"TE = {mode.te_fraction:.3f}, TM = {mode.tm_fraction:.3f}"
    )