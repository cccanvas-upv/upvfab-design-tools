from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

from .results import EMEPropagationResult


PlotAspect = Literal["auto", "equal"]


def plot_propagation(
    propagation: EMEPropagationResult,
    *,
    x_um: np.ndarray,
    field_profiles: np.ndarray,
    ax=None,
    cmap: str = "inferno",
    aspect: PlotAspect = "auto",
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    show_colorbar: bool = True,
):
    """Plot reconstructed field intensity over transverse position and z.

    This preserves the propagation view from the original EME implementation:
    the horizontal axis is propagation distance, the vertical axis is the
    sampled transverse coordinate, and color represents ``abs(E(x, z))**2``.

    Parameters
    ----------
    propagation:
        Result returned by :func:`propagate_modes`.
    x_um:
        One-dimensional transverse sampling positions in micrometers.
    field_profiles:
        Complex modal profiles with shape ``(num_modes, len(x_um))``. All
        profiles must represent the same electric-field component.
    ax:
        Optional Matplotlib axis.
    cmap:
        Matplotlib colormap name.
    aspect:
        ``"auto"`` fills the available axes. ``"equal"`` uses the same scale
        for one micrometer along x and z, matching the old ``AspectRatioOne``
        option.
    xlim:
        Optional limits for the transverse x coordinate.
    zlim:
        Optional limits for the propagation coordinate.
    show_colorbar:
        Add an intensity colorbar when True.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    x_um = np.asarray(x_um, dtype=float)
    if x_um.ndim != 1:
        raise ValueError("x_um must be a one-dimensional array.")

    if x_um.size == 0:
        raise ValueError("x_um must contain at least one position.")

    if not np.all(np.isfinite(x_um)):
        raise ValueError("x_um must contain only finite values.")

    if np.any(np.diff(x_um) <= 0):
        raise ValueError("x_um must be strictly increasing.")

    profiles = np.asarray(field_profiles, dtype=np.complex128)
    if profiles.ndim != 2 or profiles.shape[1] != x_um.size:
        raise ValueError(
            "field_profiles must have shape "
            f"(num_modes, {x_um.size}), got {profiles.shape}."
        )

    if aspect not in ("auto", "equal"):
        raise ValueError("aspect must be 'auto' or 'equal'.")

    intensity = propagation.field_intensity(profiles)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    image = ax.pcolormesh(
        propagation.z_um,
        x_um,
        intensity.T,
        shading="auto",
        cmap=cmap,
    )

    ax.set_aspect(aspect, adjustable="box")
    ax.set_xlabel("z [µm]")
    ax.set_ylabel("x [µm]")
    ax.set_title("EME propagation |E(x, z)|²")

    if xlim is not None:
        ax.set_ylim(xlim)

    if zlim is not None:
        ax.set_xlim(zlim)

    if show_colorbar:
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Field intensity |E|²")

    return fig, ax
