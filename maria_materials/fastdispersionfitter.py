import matplotlib.pylab as plt
import numpy as np
import tidy3d as td

from tidy3d.plugins.dispersion import FastDispersionFitter, AdvancedFastFitterParam

## HACEMOS EL FITTING SOBRE LOS DATOS DE n, wvl, k DEL ELIPSO USANDO UNA HERRAMIENTA DE TIDY.

file_name_sio2 = "/home/maria_romero/upvfab-design-tools/maria_materials/dioxido_fastdispersionfitter_friendly.csv"
fitter_sio2 = FastDispersionFitter.from_file(file_name_sio2, skiprows= 1, delimiter = ";")

file_name_sin = "/home/maria_romero/upvfab-design-tools/maria_materials/sin_fastdispersionfitter_friendly.csv"
fitter_sin = FastDispersionFitter.from_file(file_name_sin, skiprows= 1, delimiter = ";")

# fitter_sin.plot()
# plt.show()

# advanced_param = AdvancedFastFitterParam(weights=(1, 1))
# medium, rms_error = fitter_sio2.fit(max_num_poles=2, advanced_param=advanced_param, tolerance_rms=1e-4)
# #fitter.plot(medium)
# #plt.show()

advanced_param = AdvancedFastFitterParam(weights=(1, 1))
fitter_sio2_c = fitter_sio2.copy(update={"wvl_range": (1.5, 1.6)})
sio2, rms_error = fitter_sio2_c.fit(max_num_poles=2, advanced_param=advanced_param, tolerance_rms=1e-4)
# fitter_sio2_c.plot(sio2)
# plt.show()

# advanced_param = AdvancedFastFitterParam(weights=(1, 1))
# medium, rms_error = fitter_sin.fit(max_num_poles=10, advanced_param=advanced_param, tolerance_rms=1e-4)
# fitter_sin.plot(medium)
# plt.show()

advanced_param = AdvancedFastFitterParam(weights=(1, 1))
fitter_sin_c = fitter_sin.copy(update={"wvl_range": (1.5, 1.6)})
sin, rms_error = fitter_sin_c.fit(max_num_poles=2, advanced_param=advanced_param, tolerance_rms=1e-4)
# fitter_sin_c.plot(sin)
# plt.show()

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

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td


wavelengths_um = np.linspace(1.5, 1.6, 200)

n_values = []
k_values = []

for wavelength_um in wavelengths_um:

    freq = td.C_0 / wavelength_um

    n, k = sin.nk_model(freq)

    n_values.append(n)
    k_values.append(k)


n_values = np.asarray(n_values)
k_values = np.asarray(k_values)


plt.figure()

plt.plot(
    wavelengths_um,
    n_values,
    label="n",
)

# plt.plot(
#     wavelengths_um,
#     k_values,
#     label="k",
# )

plt.xlabel("Wavelength (µm)")
plt.ylabel("Optical constants")
plt.title("SiN - laboratory fit")

plt.grid()
plt.legend()

plt.show()