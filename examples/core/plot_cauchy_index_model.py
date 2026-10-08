"""Plot an experimentally fitted Cauchy refractive-index model."""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from upvfab_design_tools.core import CauchyIndexModel, Material


WAVELENGTH_MIN_UM = 1.5
WAVELENGTH_MAX_UM = 1.6

index_model = CauchyIndexModel(
    A=1.961864,
    B=2.257584,
    C=0.0,
    wavelength_min_um=WAVELENGTH_MIN_UM,
    wavelength_max_um=WAVELENGTH_MAX_UM,
)

material = Material(
    name="Experimental Cauchy material",
    index_model=index_model,
    description="Material defined from experimentally fitted Cauchy coefficients.",
)

wavelength_um = np.linspace(WAVELENGTH_MIN_UM, WAVELENGTH_MAX_UM, 201)
refractive_index = np.array(
    [material.n(wavelength).real for wavelength in wavelength_um]
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(wavelength_um, refractive_index, linewidth=2.0)
ax.set_xlabel("Wavelength [\N{MICRO SIGN}m]")
ax.set_ylabel("Refractive index n")
ax.set_title("Experimental Cauchy index model")
ax.grid(True, alpha=0.3)
fig.tight_layout()

print(f"n({WAVELENGTH_MIN_UM} \N{MICRO SIGN}m) = {refractive_index[0]:.9f}")
print(f"n({WAVELENGTH_MAX_UM} \N{MICRO SIGN}m) = {refractive_index[-1]:.9f}")

try:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plt.show()
except (Exception, UserWarning):
    output_file = "cauchy_index_model.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
