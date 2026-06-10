from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from .cross_section import CrossSection


def plot_cross_section(
    cross_section: CrossSection,
    *,
    wavelength_um: float = 1.55,
    ax=None,
    cmap: str = "viridis",
    show_names: bool = True,
    show_indices: bool = True,
    show_background_label: bool = False,
    edgecolor: str = "black",
    linewidth: float = 0.8,
    alpha: float = 1.0,
):
    """
    Plot a CrossSection using material refractive index as color.

    Parameters
    ----------
    cross_section:
        CrossSection object to visualize.
    wavelength_um:
        Wavelength in micrometers used to evaluate material index n(lambda).
    ax:
        Optional Matplotlib axis.
    cmap:
        Matplotlib colormap name.
    show_names:
        If True, show region names.
    show_indices:
        If True, show material refractive indices in the labels.
    show_background_label:
        If True, label the background region. Usually False to avoid clutter.
    edgecolor:
        Polygon edge color.
    linewidth:
        Polygon edge linewidth.
    alpha:
        Polygon transparency.

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """

    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    regions = cross_section.all_regions(include_background=True)

    refractive_indices = [
        _real_refractive_index(region.material, wavelength_um)
        for region in regions
    ]

    n_min = min(refractive_indices)
    n_max = max(refractive_indices)

    if np.isclose(n_min, n_max):
        n_min -= 0.05
        n_max += 0.05

    norm = mpl.colors.Normalize(vmin=n_min, vmax=n_max)
    colormap = mpl.colormaps[cmap]

    for index, region in enumerate(regions):
        is_background = index == 0

        vertices = _get_vertices(region)
        n_real = _real_refractive_index(region.material, wavelength_um)

        patch = MplPolygon(
            vertices,
            closed=True,
            facecolor=colormap(norm(n_real)),
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=alpha,
            zorder=index,
        )

        ax.add_patch(patch)

        should_label = (
            show_names
            and region.name
            and (not is_background or show_background_label)
        )

        if should_label:
            x_min, x_max, z_min, z_max = _bounds_from_vertices(vertices)

            label = region.name

            if show_indices:
                n_complex = region.material.n(wavelength_um)
                label += f"\nn={_format_index(n_complex)}"

            ax.text(
                0.5 * (x_min + x_max),
                0.5 * (z_min + z_max),
                label,
                ha="center",
                va="center",
                fontsize=8,
                color=_text_color_for_background(n_real, n_min, n_max),
                zorder=index + 0.5,
            )

    ax.set_xlim(cross_section.x_min, cross_section.x_max)
    ax.set_ylim(cross_section.z_min, cross_section.z_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("z [µm]")
    ax.set_title(
        f"{cross_section.name} | λ = {wavelength_um:g} µm"
    )
    ax.grid(True, linewidth=0.3, alpha=0.5)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Refractive index n")

    return fig, ax


def save_figure(fig, filename: str | Path, *, dpi: int = 300) -> Path:
    """
    Save a Matplotlib figure.
    """

    path = Path(filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _get_vertices(region):
    vertices = region.vertices

    if callable(vertices):
        return vertices()

    return vertices


def _bounds_from_vertices(vertices):
    xs = [point[0] for point in vertices]
    zs = [point[1] for point in vertices]

    return min(xs), max(xs), min(zs), max(zs)


def _real_refractive_index(material, wavelength_um: float) -> float:
    n = complex(material.n(wavelength_um))

    return float(np.real(n))


def _format_index(n: complex, tol: float = 1e-12) -> str:
    n = complex(n)

    if abs(n.imag) < tol:
        return f"{n.real:.3f}"

    sign = "+" if n.imag >= 0 else "-"
    return f"{n.real:.3f}{sign}{abs(n.imag):.1e}j"


def _text_color_for_background(
    n: float,
    n_min: float,
    n_max: float,
) -> str:
    """
    Choose black or white text depending on normalized refractive index.

    This is only for label readability.
    """

    if np.isclose(n_min, n_max):
        return "black"

    normalized = (n - n_min) / (n_max - n_min)

    if normalized > 0.55:
        return "white"

    return "black"