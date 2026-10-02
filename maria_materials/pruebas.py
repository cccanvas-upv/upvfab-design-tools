import numpy as np
import matplotlib.pyplot as plt

from upvfab_design_tools.core.materials import SILICON_NITRIDE


wavelengths_um = np.linspace(1.2, 1.8, 200)

n_values = np.array([
    SILICON_NITRIDE.n(wavelength_um)
    for wavelength_um in wavelengths_um
])

plt.figure()

plt.plot(
    wavelengths_um,
    np.real(n_values),
    label="n",
)

plt.xlabel("Wavelength (µm)")
plt.ylabel("Refractive index")
plt.title(SILICON_NITRIDE.name)
plt.grid()
plt.legend()

plt.show()
