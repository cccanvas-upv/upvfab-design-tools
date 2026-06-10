import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle, Polygon, Trapezoid
from upvfab_design_tools.core.materials import (
    AIR,
    BCB,
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)


def get_vertices(geometry):
    """
    Return vertices from Rectangle, Trapezoid or Polygon.
    """

    vertices = geometry.vertices

    if callable(vertices):
        return vertices()

    return vertices


def get_bounds_from_vertices(vertices):
    xs = [point[0] for point in vertices]
    zs = [point[1] for point in vertices]

    return min(xs), max(xs), min(zs), max(zs)


def material_color(material_name):
    """
    Simple color map for checking geometry visually.
    """

    colors = {
        "Air": "#ffffff",
        "Benzocyclobutene": "#d9ead3",
        "Thermal Silicon Dioxide": "#dddddd",
        "Silicon Nitride": "#7030a0",
    }

    return colors.get(material_name, "#cccccc")


def plot_cross_section(cross_section: CrossSection, ax=None, show_names=True):
    """
    Plot a CrossSection object.

    Later structures are drawn on top of previous structures,
    following the same priority rule that the solver should use.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure

    regions = cross_section.all_regions(include_background=True)

    for region in regions:
        vertices = get_vertices(region)

        patch = MplPolygon(
            vertices,
            closed=True,
            facecolor=material_color(region.material.name),
            edgecolor="black",
            linewidth=1.0,
            alpha=1.0,
        )

        ax.add_patch(patch)

        if show_names and region.name:
            x_min, x_max, z_min, z_max = get_bounds_from_vertices(vertices)

            ax.text(
                0.5 * (x_min + x_max),
                0.5 * (z_min + z_max),
                region.name,
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
            )

    ax.set_xlim(cross_section.x_min, cross_section.x_max)
    ax.set_ylim(cross_section.z_min, cross_section.z_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("z [µm]")
    ax.set_title(cross_section.name)
    ax.grid(True, linewidth=0.3)

    return fig, ax


# ----------------------------------------------------------------------
# Complex cross-section example
# ----------------------------------------------------------------------
# Interpretation:
#
# 1. Whole domain is SiO2.
# 2. Upper region is BCB.
# 3. Upper-left window is Air.
# 4. SiN core is drawn on top.
# ----------------------------------------------------------------------

bcb_region = Rectangle(
    x_min=-4.0,
    x_max=4.0,
    z_min=0.3,
    z_max=0.35,
    material=BCB,
    name="BCB",
)

air_region = Rectangle(
    x_min=-4.0,
    x_max=4.0,
    z_min=0.35,
    z_max=2.0,
    material=AIR,
    name="Air",
)

sin_core = Rectangle(
    x_min=-0.5,
    x_max=0.5,
    z_min=0.0,
    z_max=0.3,
    material=SILICON_NITRIDE,
    name="SiN",
)

xs = CrossSection(
    name="complex_sin_cross_section",
    background_material=THERMAL_SILICON_DIOXIDE,
    x_min=-4.0,
    x_max=4.0,
    z_min=-2.0,
    z_max=2.0,
    structures=(
        bcb_region,
        air_region,
        sin_core,
    ),
)

fig, ax = plot_cross_section(xs)
try:
    plt.show()
except UserWarning:
    pass
output_file = "complex_cross_section.png"
fig.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Saved figure to {output_file}")