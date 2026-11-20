
# 4 MODES TARDA PROU
#importem tot lo necessari: cross_sections, materials, solvers, ferramentes de plot
import numpy as np
import matplotlib.pyplot as plt
from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle, Trapezoid
from upvfab_design_tools.core.materials import (
    AIR,
    BCB,
    SILICON_NITRIDE_O_BAND,
    THERMAL_SILICON_DIOXIDE_O_BAND,
)
from upvfab_design_tools.modesolver import FemwellModeSolver, Tidy3DModeSolver
from upvfab_design_tools.modesolver import plot_modes_grid
from upvfab_design_tools.core.visualization import (
    plot_cross_section,
    save_figure,
)
from upvfab_design_tools.core.waveguides import strip_waveguide, strip_waveguide_bcb

# #waveguide parameters
# wvg_width = 1.0
# wvg_thickness = 0.3
# bcb_thickness = 0.4

# #modesolver parameters
# window_width = 10.0
# bottom_margin = 3.0
# top_margin = 3.0

# bcb_region = Rectangle(
#     x_min = -0.5*window_width, #per a tindre el centre en 0
#     x_max = 0.5*window_width, 
#     z_min = 0.0, 
#     z_max = bcb_thickness, 
#     material = BCB, 
#     name = "BCB",
# )

# air_region = Rectangle(
#     x_min = -0.5*window_width, 
#     x_max = 0.5*window_width, 
#     z_min = bcb_thickness, 
#     z_max = bcb_thickness + bottom_margin, 
#     material = AIR, 
#     name = "Air",
# )

# sin_core = Rectangle(
#     x_min = -0.5*wvg_width,
#     x_max = 0.5*wvg_width,
#     z_min = 0, 
#     z_max = wvg_thickness, #aixina tens bcb de cladding
#     material = SILICON_NITRIDE, 
#     name = "Core",
# )

# xs = CrossSection(
#     name = "sin_cross_section", 
#     background_material = THERMAL_SILICON_DIOXIDE, 
#     x_min = -0.5*window_width, 
#     x_max = 0.5*window_width, 
#     z_min = -bottom_margin, 
#     z_max = wvg_thickness + top_margin,
#     structures = (
#         bcb_region, 
#         air_region,
#         sin_core, 
#     ),
# )

# fig, ax = plot_cross_section(
#     xs,
#     wavelength_um = 1.55,
#     show_names = True,
#     show_indices = True,
# )
# plt.show() #aixina revises que la xs está bé

# # solver configuration

# solver = FemwellModeSolver(
#     default_resolution=0.3, #default mesh resolution in um
#     min_resolution=0.01, #minimum automatically generated resolution in um
#     resolution_factor=10.0, #automatic resolution is approximately min_feature / resolution_factor
#     filter_guided=True,#if True, only keeps modes with n_eff above the reference material
#     reference_material=THERMAL_SILICON_DIOXIDE, #if None, xs background material is used
#     guided_tolerance=1e-2, #minimum index difference for guided mode filtering
#     enable_plots=False, #if True, plot mesh, domains and epsilon
# )

# result = solver.solve(
#     cross_section = xs,
#     wavelength_um = 1.55,
#     num_modes = 4, 
# )

# # comentaris camilo:
# # Need to adjust because in a grid I might need to switch from one field component to another. TE/TM
# # Also need to add the option to plot "abs" or "real"

# fig, axs = plot_modes_grid(
#     result,
#     field_component = "auto",
#     field_part = "mag", 
#     max_modes = 4,
#     xlim = (-2,2),
#     zlim = (-1,1),
# )

# # fig_neff, ax_neff = plot_effective_indices(result)

# try:
#     plt.show()
# except Exception:
#     save_figure(fig, "modes_ex.png")
#     # save_figure(fig_neff, "effective_indices.png")
#     pass

# for mode in result.modes:
#     print(
#         mode.index,
#         mode.neff, 
#         mode.polarization, 
#         mode.te_fraction, 
#         mode.tm_fraction,
#     )

# xs = strip_waveguide_bcb(
#     width = 1, 
#     height = 0.3,
#     core_material= SILICON_NITRIDE,
#     top_material = BCB,
#     top_material_thickness = 0.4,
#     background_material =  THERMAL_SILICON_DIOXIDE,
#     air = AIR, 
#     x_center= 0.0,
#     z_min= 0.0,
#     sidewall_angle_deg = 0.0,
#     x_span = 6.0,
#     bottom_margin = 2.0,
#     top_margin =  2.0,
#     name= "strip_waveguide_bcb",
#     )

xs = strip_waveguide(
    width = 1, 
    height = 0.3,
    core_material = SILICON_NITRIDE_O_BAND,
    background_material= THERMAL_SILICON_DIOXIDE_O_BAND, 
    name = "strip_waveguide-o-band"
)

fig, ax = plot_cross_section(
    xs,
    wavelength_um = 1.31,
    show_names = True,
    show_indices = True,
)

# solver configuration

# solver = FemwellModeSolver(
#     default_resolution=0.3, #default mesh resolution in um
#     min_resolution=0.03, #minimum automatically generated resolution in um
#     resolution_factor=6.0, #automatic resolution is approximately min_feature / resolution_factor
#     filter_guided=True,#if True, only keeps modes with n_eff above the reference material
#     reference_material=THERMAL_SILICON_DIOXIDE_O_BAND, #if None, xs background material is used
#     guided_tolerance=1e-2, #minimum index difference for guided mode filtering
#     enable_plots=False, #if True, plot mesh, domains and epsilon
# )

solver = Tidy3DModeSolver(
    min_steps_per_wvl = 45, 
    reference_material = THERMAL_SILICON_DIOXIDE_O_BAND, 
    filter_guided = True, 
    guided_tolerance = 1e-2, 
)

wl = np.linspace(1.29, 1.32, 60)
neff = []

for wavelength in wl:
    result = solver.solve(
        cross_section=xs, 
        wavelength_um=float(wavelength), 
        num_modes = 1,
    )

    te_modes = [mode for mode in result.modes if mode.te_fraction>0.5]
    mode = max(te_modes, key=lambda m: np.real(m.neff))
    neff.append(np.real(mode.neff))

neff = np.array(neff)
ng = neff - wl * np.gradient(neff, wl, edge_order=2)

print("Effective indices:", neff)
print("Group indices:", ng)

# Group index at 1310 nm
idx = np.argmin(np.abs(wl - 1.31))

print(f"\nAt {wl[idx]*1000:.1f} nm:")
print(f"neff = {neff[idx]:.8f}")
print(f"ng = {ng[idx]:.8f}")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(wl * 1000, neff, "o-")
axs[0].set_xlabel("Wavelength (nm)")
axs[0].set_ylabel("Effective index")
axs[0].set_title("Effective index vs wavelength")
axs[0].grid(True)

axs[1].plot(wl * 1000, ng, "o-")
axs[1].set_xlabel("Wavelength (nm)")
axs[1].set_ylabel("Group index")
axs[1].set_title("Group index vs wavelength")
axs[1].grid(True)

plt.tight_layout()
plt.show()


# fig, axs = plot_modes_grid(
#     result,
#     field_component = "auto",
#     field_part = "mag", 
#     max_modes = 4,
#     xlim = (-2,2),
#     zlim = (-1,1),
# )

# # fig_neff, ax_neff = plot_effective_indices(result)

# try:
#     plt.show()
# except Exception:
#     save_figure(fig, "modes_ex.png")
#     # save_figure(fig_neff, "effective_indices.png")
#     pass

# for mode in result.modes:
#     print(
#         mode.index,
#         mode.neff, 
#         mode.polarization, 
#         mode.te_fraction, 
#         mode.tm_fraction,
#     )

# plt.show()
    