"""Fit an experimental Cauchy material to a Tidy3D PoleResidue model."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td

from upvfab_design_tools.core import CauchyIndexModel, Material


WAVELENGTH_MIN_UM = 1.5
WAVELENGTH_MAX_UM = 1.6

cauchy_model = CauchyIndexModel(
    A=1.961864,
    B=2.257584,
    C=0.0,
    wavelength_min_um=WAVELENGTH_MIN_UM,
    wavelength_max_um=WAVELENGTH_MAX_UM,
)

material = Material(
    name="Experimental Cauchy Material",
    index_model=cauchy_model,
    description="Material defined from experimentally fitted Cauchy coefficients.",
)

tidy_medium, rms_error = material.fit_tidy(
    num_samples=21,
    wavelength_min_um=WAVELENGTH_MIN_UM,
    wavelength_max_um=WAVELENGTH_MAX_UM,
    max_num_poles=2,
)

wavelengths_um = np.linspace(WAVELENGTH_MIN_UM, WAVELENGTH_MAX_UM, 600)
original_n_values = np.array(
    [material.n(wavelength_um).real for wavelength_um in wavelengths_um]
)

n_values = []
k_values = []

for wavelength_um in wavelengths_um:
    frequency_hz = td.C_0 / wavelength_um
    n, k = tidy_medium.nk_model(frequency_hz)
    n_values.append(n)
    k_values.append(k)

n_values = np.asarray(n_values)
k_values = np.asarray(k_values)

print(f"Pole-residue RMS error: {rms_error:.9g}")
print(f"Number of fitted poles: {len(tidy_medium.poles)}")
print("n values:")
print(n_values)
print("k values:")
print(k_values)

fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

axes[0].plot(wavelengths_um, original_n_values, label="Cauchy input")
axes[0].plot(wavelengths_um, n_values, "--", label="Tidy3D PoleResidue")
axes[0].set_ylabel("Refractive index n")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(wavelengths_um, k_values)
axes[1].set_xlabel("Wavelength [\N{MICRO SIGN}m]")
axes[1].set_ylabel("Extinction coefficient k")
axes[1].grid(True, alpha=0.3)

fig.suptitle(f"Cauchy to Tidy3D fit (RMS = {rms_error:.3g})")
fig.tight_layout()

try:
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plt.show()
except (Exception, UserWarning):
    output_file = "cauchy_tidy_fit.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_file}")
