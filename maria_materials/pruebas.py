## EJEMPLO DE USO DEL MATERIAL DISPERSIVO

import tidy3d as td

fname_sin = "/home/maria_romero/upvfab-design-tools/maria_materials/fits/sin_cband.json"
sin = td.PoleResidue.from_file(fname_sin)

fname_sio2 = "/home/maria_romero/upvfab-design-tools/maria_materials/fits/sio2_cband.json"
sio2 = td.PoleResidue.from_file(fname_sio2)


import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td


wavelengths_um = np.linspace(1.5, 1.6, 600)

n_values = []
k_values = []

for wavelength_um in wavelengths_um:

    freq = td.C_0 / wavelength_um

    n, k = sin.nk_model(freq)

    n_values.append(n)
    k_values.append(k)


n_values = np.asarray(n_values)
k_values = np.asarray(k_values)

print(n_values)
print(k_values)


# lo probamos en una guía -> funciona
# 


##################### SIMULACIÓN GUIA ONDA

# import numpy as np
# import matplotlib.pyplot as plt
# import tidy3d as td

# from tidy3d.plugins.mode import ModeSolver


# # ============================================================
# # 1. LONGITUD DE ONDA
# # ============================================================

# wvl = np.linspace(1.5, 1.6, 101)  # um
# freqs = td.C_0 / wvl


# # ============================================================
# # 3. GEOMETRÍA DE LA GUÍA
# # ============================================================

# wg_width = 1  # um
# wg_height = 0.3  # um

# waveguide = td.Structure(
#     geometry=td.Box(
#         center=(0, 0, 0),
#         size=(td.inf, wg_width, wg_height),
#     ),
#     medium=sin,
# )


# # ============================================================
# # 4. SIMULACIÓN
# # ============================================================

# sim = td.Simulation(
#     size=(2, 3, 3),

#     # Todo lo que no sea el core será nuestro SiO2 dispersivo
#     medium=sio2,

#     structures=[waveguide],

#     grid_spec=td.GridSpec.auto(
#         wavelength=wvl.min(),
#         min_steps_per_wvl=20,
#     ),

#     boundary_spec=td.BoundarySpec.all_sides(
#         boundary=td.PML()
#     ),

#     run_time=1e-12,
# )


# # ============================================================
# # 5. PLANO TRANSVERSAL
# # ============================================================

# mode_plane = td.Box(
#     center=(0, 0, 0),
#     size=(0, 3, 3),
# )


# # ============================================================
# # 6. MODE SOLVER
# # ============================================================

# mode_solver = ModeSolver(
#     simulation=sim,
#     plane=mode_plane,
#     mode_spec=td.ModeSpec(
#         num_modes=4,
#         target_neff=1.8,
#     ),
#     freqs=freqs,
# )

# mode_data = mode_solver.solve()

# # Índice correspondiente a lambda = 1.55 um
# idx_1550 = np.argmin(np.abs(wvl - 1.55))

# # Índices efectivos de los dos primeros modos
# neff_0 = mode_data.n_eff.values[idx_1550, 0]
# neff_1 = mode_data.n_eff.values[idx_1550, 1]

# print(f"λ = {wvl[idx_1550]:.4f} µm")
# print(f"Mode 0: neff = {neff_0}")
# print(f"Mode 1: neff = {neff_1}")


# # ============================================================
# # 7. RESOLVER
# # ============================================================

# mode_data = mode_solver.solve()

# print(mode_data.to_dataframe())

# lambda0 = 1.55
# freq0 = td.C_0 / lambda0

# for mode_index in range(2):

#     mode_solver.plot_field(
#         "E",
#         val="abs",
#         mode_index=mode_index,
#         f=freq0,
#     )

#     plt.title(f"Mode {mode_index} - λ = {lambda0} µm")
#     plt.show()

# n_eff = mode_data.n_eff.values

# plt.figure()

# for mode_index in range(2):
#     plt.plot(
#         wvl,
#         np.real(n_eff[:, mode_index]),
#         label=f"Mode {mode_index}",
#     )

# plt.xlabel("Wavelength (µm)")
# plt.ylabel("Effective index")
# plt.legend()
# plt.grid()
# plt.show()




############ PLOT DEL SIN DE LA LIBRERÍA

# import numpy as np
# import matplotlib.pyplot as plt

# from upvfab_design_tools.core.materials import SILICON_NITRIDE


# wavelengths_um = np.linspace(1.2, 1.8, 200)

# n_values = np.array([
#     SILICON_NITRIDE.n(wavelength_um)
#     for wavelength_um in wavelengths_um
# ])

# plt.figure()

# plt.plot(
#     wavelengths_um,
#     np.real(n_values),
#     label="n",
# )

# plt.xlabel("Wavelength (µm)")
# plt.ylabel("Refractive index")
# plt.title(SILICON_NITRIDE.name)
# plt.grid()
# plt.legend()

# plt.show()
