import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import upvfab_design_tools as udt
from upvfab_design_tools.core import get_material
from upvfab_design_tools.core.geometry import Rectangle, Trapezoid, Polygon
from upvfab_design_tools.core.materials import SILICON_NITRIDE

print(udt.__version__)

core_material = get_material("sin")
cladding_material = get_material("thermal_sio2")

print(core_material.n(1.55))
print(cladding_material.n(1.55))


def get_vertices(geometry):
    """
    Return the vertices of a geometry primitive.

    Rectangle and Trapezoid expose a vertices() method.
    Polygon stores the vertices directly as an attribute.
    """

    if hasattr(geometry, "vertices"):
        vertices = geometry.vertices

        if callable(vertices):
            return vertices()

        return vertices

    raise TypeError(f"Unsupported geometry type: {type(geometry)}")


def plot_geometry(geometries, ax=None, show_names=True):
    """
    Plot a list of 2D geometry primitives in the x-z plane.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    for geometry in geometries:
        vertices = get_vertices(geometry)

        patch = MplPolygon(
            vertices,
            closed=True,
            fill=True,
            alpha=0.4,
            edgecolor="black",
        )

        ax.add_patch(patch)

        if show_names and getattr(geometry, "name", ""):
            xs = [point[0] for point in vertices]
            zs = [point[1] for point in vertices]

            x_center = 0.5 * (min(xs) + max(xs))
            z_center = 0.5 * (min(zs) + max(zs))

            ax.text(
                x_center,
                z_center,
                geometry.name,
                ha="center",
                va="center",
            )

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("z [µm]")
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale_view()
    ax.grid(True)

    return fig, ax


# ----------------------------------------------------------------------
# Example 1: rectangular strip waveguide
# ----------------------------------------------------------------------

strip_rectangular = Rectangle(
    x_min=-0.5,
    x_max=0.5,
    z_min=0.0,
    z_max=0.3,
    material=SILICON_NITRIDE,
    name="strip rectangular",
)

# ----------------------------------------------------------------------
# Example 2: strip waveguide with angled sidewalls
# ----------------------------------------------------------------------

strip_trapezoid = Trapezoid.from_sidewall_angle(
    x_center=0.0,
    z_min=0.6,
    height=0.3,
    width=1.0,
    width_reference="top",
    sidewall_angle_deg=30,
    material=SILICON_NITRIDE,
    name="strip angled",
)

# ----------------------------------------------------------------------
# Example 3: rib waveguide
# ----------------------------------------------------------------------

rib_slab = Rectangle(
    x_min=-3.0,
    x_max=3.0,
    z_min=1.2,
    z_max=1.35,
    material=SILICON_NITRIDE,
    name="rib slab",
)

rib_ridge = Trapezoid.from_sidewall_angle(
    x_center=0.0,
    z_min=1.35,
    height=0.25,
    width=1.0,
    width_reference="top",
    sidewall_angle_deg=6.0,
    material=SILICON_NITRIDE,
    name="rib ridge",
)

# ----------------------------------------------------------------------
# Example 4: arbitrary polygon
# ----------------------------------------------------------------------

custom_polygon = Polygon(
    vertices=(
        (-0.6, 1.9),
        (0.6, 1.9),
        (0.4, 2.2),
        (-0.4, 2.2),
    ),
    material=SILICON_NITRIDE,
    name="custom polygon",
)

fig, ax = plot_geometry(
    [
        strip_rectangular,
        strip_trapezoid,
        rib_slab,
        rib_ridge,
        custom_polygon,
    ]
)

fig.savefig("geometry_primitives.png", dpi=300, bbox_inches="tight")

try:
    plt.show()
except UserWarning:
    pass
