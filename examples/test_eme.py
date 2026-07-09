"""Minimal check of uniform-section eigenmode propagation."""

import numpy as np
import matplotlib.pyplot as plt
from upvfab_design_tools.eme import (
    modal_excitation_coefficients,
    normalize_profiles,
    plot_propagation,
    propagate_modes,
)
from upvfab_design_tools.modesolver.results import Mode, ModeSolverResult


wavelength_um = 1.55
modes = (
    Mode(index=0, neff=2.0, wavelength_um=wavelength_um),
    Mode(index=1, neff=1.8, wavelength_um=wavelength_um),
)
mode_result = ModeSolverResult(
    modes=modes,
    wavelength_um=wavelength_um,
    backend="synthetic",
)

x_um = np.linspace(-3.0, 3.0, 401)
field_profiles = normalize_profiles(
    np.array(
        [
            np.exp(-(x_um / 0.8) ** 2),
            (x_um / 0.8) * np.exp(-(x_um / 0.8) ** 2),
        ]
    ),
    x_um,
)

input_profile = field_profiles[0] + 0.5j * field_profiles[1]
initial_amplitudes = modal_excitation_coefficients(
    input_profile,
    field_profiles,
    x_um,
)
propagation = propagate_modes(
    mode_result,
    initial_amplitudes=initial_amplitudes,
    length_um=10.0,
    dz_um=0.3,
)

expected_final = initial_amplitudes * np.exp(
    -1j * np.array([mode.beta for mode in modes]) * propagation.z_um[-1]
)

assert propagation.z_um[-1] == 10.0
assert np.allclose(propagation.modal_amplitudes[0], initial_amplitudes)
assert np.allclose(propagation.final_amplitudes, expected_final)
assert np.allclose(
    propagation.total_modal_power,
    np.sum(np.abs(initial_amplitudes) ** 2),
)
assert np.allclose(np.sum(np.abs(initial_amplitudes) ** 2), 1.0)
field = propagation.reconstruct_field(field_profiles)

assert field.shape == (propagation.z_um.size, x_um.size)

fig, ax = plot_propagation(
    propagation,
    x_um=x_um,
    field_profiles=field_profiles,
    aspect="auto",
)

plt.show()
# fig.savefig("eme_propagation.png", dpi=300, bbox_inches="tight")

print("Uniform EME propagation check passed.")
print("Final modal amplitudes:", propagation.final_amplitudes)
print("Propagation plot generated.")
