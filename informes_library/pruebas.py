from upvfab_design_tools.core.visualization import plot_cross_section
from upvfab_design_tools.core.waveguides import strip_waveguide
import matplotlib.pyplot as plt

from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)

DOMAIN_X_SPAN_UM = 18.0

cross_section = strip_waveguide(
    width=1,
    height=0.3,
    core_material=SILICON_NITRIDE,
    background_material=THERMAL_SILICON_DIOXIDE,
    x_center=0.0,
    z_min=0.0,
    x_span=DOMAIN_X_SPAN_UM,
    bottom_margin=1.5,
    top_margin=1.5,
    name="sin_strip_fdtd_stack",
)

fig, ax = plot_cross_section(cross_section)
plt.show()