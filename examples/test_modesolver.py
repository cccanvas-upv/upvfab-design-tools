from upvfab_design_tools.core.cross_section import CrossSection
from upvfab_design_tools.core.geometry import Rectangle
from upvfab_design_tools.core.materials import (AIR, BCB, SILICON_NITRIDE, THERMAL_SILICON_DIOXIDE)
from upvfab_design_tools.modesolver import FemwellModeSolver
from upvfab_design_tools.modesolver import plot_effective_indices,plot_modes_grid,plot_mode_ex_ey, save_figure
import matplotlib.pyplot as plt
from upvfab_design_tools.core.visualization import (plot_cross_section, save_figure,)
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning) 
from femwell.mesh import mesh_from_Dict

width_array = 1.0 + np.arange(0, 9)*0.05
width_array = [1.0]
wavelength = np.arange(1.50, 1.63, 0.01)
wavelength = [1.55]
folder_name = 'Test'

bcb_region = Rectangle(
        x_min=-4.0,
        x_max=4.0,
        z_min=0.3,
        z_max=0.4,
        material=BCB,
        name="BCB",
    )

air_region = Rectangle(
    x_min=-4.0,
    x_max=4.0,
    z_min=0.4,
    z_max=2.0,
    material=AIR,
    name="Air",
)

tot = len(width_array)
i = 0

for wg_width in width_array:
    print(f'Starting sweep for waveguide width: {wg_width} um')
    folder_name2 = f'results/{folder_name}/{wg_width}um'
    folder_path = Path(folder_name2)
    folder_path.mkdir(parents=True, exist_ok=True)

    output_file_wg = f"{folder_name2}/complex_cross_section.png"
    file_name = f'{folder_name2}/modesolver_results_{wg_width}um.txt'

    sin_core = Rectangle(
        x_min = -wg_width/2,
        x_max = wg_width/2,
        z_min=0.0,
        z_max=0.3,
        material=SILICON_NITRIDE,
        name="core",
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

    fig, ax = plot_cross_section(
        xs,
        wavelength_um=1.55,
        show_names=False,
        show_indices=True,
    )

    fig.savefig(output_file_wg, dpi=300, bbox_inches="tight")

    solver = FemwellModeSolver(
        default_resolution=0.4,
        min_resolution=0.02,
        resolution_factor=5.0,
        filter_guided=True,
        reference_material=THERMAL_SILICON_DIOXIDE,
        guided_tolerance=1e-2,
        enable_plots=False,
    )
    
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(f'{"wavelength [nm]"}\t {"index"}\t {"neff"}\t\t\t\t {"polarization"}\t {"te_fraction"}\t {"tm_fraction"}\n')

        for wl in wavelength:
            print(f'Solving for wavelength: {wl*1e3:.1f} nm')

            result = solver.solve(
                cross_section=xs,
                wavelength_um=wl,
                num_modes=4,
            )

            # Extraer la malla según si Femwell devolvió un dict, Basis o Mesh directamente
            raw_data = result[0].raw

            if hasattr(raw_data, "mesh"):
                mesh = raw_data.mesh
            elif isinstance(raw_data, dict) and "mesh" in raw_data:
                mesh = raw_data["mesh"]
            elif hasattr(raw_data, "basis"):
                mesh = raw_data.basis.mesh
            else:
                # Si raw es una tupla/lista (por ej. (basis, fields))
                mesh = raw_data[0].mesh if hasattr(raw_data[0], "mesh") else raw_data[0]

            # 3. Extraer coordenadas y nodos para graficar
            if hasattr(mesh, "points"):
                x, y = mesh.points[:, 0], mesh.points[:, 1]
                triangles = mesh.cells_dict["triangle"]
            else: # Estructura skfem.Mesh2D (p, t)
                x, y = mesh.p[0], mesh.p[1]
                triangles = mesh.t.T

            # 4. Graficar y guardar la imagen
            plt.figure(figsize=(8, 6))
            plt.triplot(x, y, triangles, lw=0.3, color="navy")
            plt.gca().set_aspect("equal")
            plt.xlabel("x [µm]", fontsize=12)
            plt.ylabel("y [µm]", fontsize=12)
            #plt.title(f"Mallado FEM ({len(x)} nodos)")
            plt.grid(alpha=0.5)

            # Guardar la imagen en disco
            plt.savefig("mesh_femwell.png", dpi=300, bbox_inches="tight")
            print(f"¡ÉXITO! Malla generada con {len(x)} nodos y guardada en 'malla_femwell.png'")
            plt.show()
            # Need to adjust because in a grid I might need to switch from one field component to another. TE/TM
            # Also need to add the option to plot "abs" or "real"
            fig, axs = plot_modes_grid(
                result,
                field_component="Ex",
                max_modes=4,
                xlim=(-2, 2),
                zlim=(-1, 1),
            )

            output_file_mode = f"{folder_name2}/modes_ex_{wl*1e3:.1f}nm.png"
            save_figure(fig, output_file_mode)

            for mode in result.modes:
                idx = mode.index
                neff = mode.neff
                pol = mode.polarization
                te = mode.te_fraction
                tm = mode.tm_fraction
            
                f.write(f"{wl*1e3:.4f}\t\t {idx}\t\t {neff:.6f}\t {str(pol)}\t\t\t\t {te:.4f}\t\t\t {tm:.4f}\n")
                f.flush()
    i += 1
    print(f'Sweep {i}/{tot} Completed!')