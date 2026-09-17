import numpy as np
from matplotlib import pyplot as plt
from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.waveguides import strip_waveguide
from upvfab_design_tools.core.visualization import (
    plot_cross_section,
    save_figure,
)
from upvfab_design_tools.eme import (
    modal_excitation_coefficients,
    normalize_profiles,
    plot_propagation,
    propagate_modes,
    shift_profile,
)
from upvfab_design_tools.modesolver import (
    FemwellModeSolver,
    ModeSolverResult,
    Tidy3DModeSolver,
    plot_modes_grid,
    sample_mode_profile,
    sample_mode_profiles,
)

#gráficas que pretendo obtener (ambos solvers): neff (width), neff(lambda), neff(sw_ang), neff(height)
#datos: neff, ng

#parámetros cross_section

LAMBDA_0_UM = 1.55

NOMINAL_CORE_HEIGHT_UM = 0.3
CORE_HEIGHT_DEV_UM = 0.0
CORE_HEIGHT_UM = NOMINAL_CORE_HEIGHT_UM + CORE_HEIGHT_DEV_UM

NOMINAL_WIDTH_UM = 1.0
WIDTH_DEV_UM = 0.0
WIDTH_UM = NOMINAL_WIDTH_UM + WIDTH_DEV_UM

NOMINAL_SIDEWALL_ANGLE_DEG = 20 #no estandarizado, G me ha dicho que se han obtenido entre 70-80º (se traducen a 30-20º en la convencción de la xs creo)
SIDEWALL_ANGLE_DEV_DEG = 0.0
SIDEWALL_ANGLE_DEG = NOMINAL_SIDEWALL_ANGLE_DEG + SIDEWALL_ANGLE_DEV_DEG



#parámetros simulación
NUM_MODES = 4
DOMAIN_X_SPAN_UM = 6.0
FIELD_CUT_Z_UM = 0.5 * CORE_HEIGHT_UM

SOLVER = 0 #0 = femwell, 1 = tidy

#barrido en lambda
LAMBDA_MIN_UM = 1.50
LAMBDA_MAX_UM = 1.60
N_LAMBDA = 21

#barrido en anchura
WIDTH_MIN_UM = 1
WIDTH_MAX_UM = 1.4
N_WIDTH = 4

#barrido en sidewall_angle
SIDEWALL_ANGLE_MIN_DEG = 15
SIDEWALL_ANGLE_MAX_DEG = 30
N_SIDEWALL_ANGLE = 16

#barrido en altura
HEIGHT_MIN_UM = 0.25
HEIGHT_MAX_UM = 0.35
N_HEIGHT = 21


#gráficas: para elegir las que queremos en cada momento

PLOTS = {
    "cross_section": False,
    "modes": False,
    "neff_lambda": False,
    "ng_lambda": False,
    "neff_width": False,
    "neff_sidewall_angle": False,
    "neff_height": True,
}

# cross-section

def make_xs(
    width=WIDTH_UM,
    height=CORE_HEIGHT_UM,
    sidewall_angle=SIDEWALL_ANGLE_DEG,
):

    return strip_waveguide(
        width=width,
        height=height,
        sidewall_angle_deg=sidewall_angle,
        core_material=SILICON_NITRIDE,
        background_material=THERMAL_SILICON_DIOXIDE,
        x_center=0.0,
        z_min=0.0,
        x_span=DOMAIN_X_SPAN_UM,
        bottom_margin=1.5,
        top_margin=1.5,
        name="-",
    )

access_wvg_xs = make_xs()

if PLOTS["cross_section"]:
    plot_cross_section(
        access_wvg_xs,
        wavelength_um=LAMBDA_0_UM,
        show_names=False,
        show_indices=True,
    )

# solvers

########## IMPORTANTE AVERIGUAR VALORES RESOLUCIONES

solver_femwell = FemwellModeSolver(
    default_resolution = 0.3, 
        min_resolution = 0.01, 
        resolution_factor = 10.0,
        filter_guided = True, 
        reference_material = THERMAL_SILICON_DIOXIDE, 
        guided_tolerance = 1e-2, 
        enable_plots = False, 
)

solver_tidy = solver = Tidy3DModeSolver(
    min_steps_per_wvl=30,
    reference_material=THERMAL_SILICON_DIOXIDE,
    filter_guided=True,
    guided_tolerance=1e-2,
)

if SOLVER == 0:
    SOLVER = solver_femwell
    SOLVER_NAME = "Femwell"
else:
    SOLVER = solver_tidy
    SOLVER_NAME = "Tidy3D"



spectral_simulation = (
    PLOTS["neff_lambda"]
    or PLOTS["ng_lambda"]
)

if spectral_simulation:

    wavelengths = np.unique(np.append(np.linspace(LAMBDA_MIN_UM, LAMBDA_MAX_UM, N_LAMBDA), LAMBDA_0_UM))

    results = [
        SOLVER.solve(
            access_wvg_xs,
            wavelength_um=wl,
            num_modes=NUM_MODES, 
        )
        for wl in wavelengths
    ]

    n_modes_found = min(len(result.modes) for result in results) #aqui puede haber error porque coge el min número de modos comunes a todas las lambdas

    print(f"Requested modes: {NUM_MODES}")
    print(f"Guided modes found across wavelength sweep: {n_modes_found}")

    neff = np.array([
        [mode.neff for mode in result.modes[:n_modes_found]]
        for result in results
    ])

    MODES_TO_PLOT = range(n_modes_found)

    lambda0_index = np.argmin(
        np.abs(wavelengths - LAMBDA_0_UM)
    )

    result_lambda0 = results[lambda0_index]

else:

    result_lambda0 = SOLVER.solve(
        access_wvg_xs,
        wavelength_um=LAMBDA_0_UM,
        num_modes=NUM_MODES,
    ) 

# visualización modos

if PLOTS["modes"]:

    fig, axs = plot_modes_grid(
        result_lambda0,
        field_component="auto",
        field_part="mag",
        max_modes=len(result_lambda0.modes),
        xlim=(-2, 2),
        zlim=(-1, 1),
    )

    fig.suptitle(
        f"Modes at λ = {LAMBDA_0_UM:.3f} µm - {SOLVER_NAME}"
    )

# neff vs lambda

if PLOTS["neff_lambda"]:

    plt.figure()

    for mode in MODES_TO_PLOT:

        polarization = result_lambda0.modes[mode].polarization

        plt.plot(
            wavelengths,
            np.real(neff[:, mode]),
            label=f"Mode {mode} ({polarization})",
        )

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("$n_{eff}$")
    plt.title(f"$n_{{eff}}$ vs wavelength - {SOLVER_NAME}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

# ng vs lambda

if PLOTS["ng_lambda"]:

    neff_real = np.real(neff)

    dneff_dlambda = np.gradient(
        neff_real,
        wavelengths,
        axis=0,
    )

    ng = neff_real - wavelengths[:, None] * dneff_dlambda

    plt.figure()

    for mode in MODES_TO_PLOT:

        polarization = result_lambda0.modes[mode].polarization

        plt.plot(
            wavelengths,
            ng[:, mode],
            label=f"Mode {mode} ({polarization})",
        )

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("$n_g$")
    plt.title(f"Group index vs wavelength - {SOLVER_NAME}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    print(f"ng at lambda_0 = {ng[lambda0_index]}")

# neff vs width
if PLOTS["neff_width"]:

    widths = np.linspace(WIDTH_MIN_UM,WIDTH_MAX_UM,N_WIDTH)

    width_results = [
        SOLVER.solve(
            make_xs(width=width),
            wavelength_um=LAMBDA_0_UM,
            num_modes=NUM_MODES,
        )
        for width in widths
    ]

    n_modes_width = min(len(result.modes) for result in width_results)
    print(f"Guided modes across width sweep: {n_modes_width}")

    neff_width = np.array([
        [mode.neff for mode in result.modes[:n_modes_width]]
        for result in width_results
    ])

    nominal_width_index = np.argmin(np.abs(widths - WIDTH_UM))
    plt.figure()

    for mode in range(n_modes_width):

        polarization = (width_results[nominal_width_index].modes[mode].polarization)

        plt.plot(
            widths,
            np.real(neff_width[:, mode]),
            label=f"Mode {mode} ({polarization})",
        )

    plt.xlabel("Width (µm)")
    plt.ylabel("$n_{eff}$")
    plt.title(
        f"$n_{{eff}}$ vs width at λ = {LAMBDA_0_UM:.3f} µm - {SOLVER_NAME}"
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()

# neff vs sidewall_angle

if PLOTS["neff_sidewall_angle"]:

    sidewall_angles = np.linspace(SIDEWALL_ANGLE_MIN_DEG, SIDEWALL_ANGLE_MAX_DEG, N_SIDEWALL_ANGLE)

    sidewall_results = [
        SOLVER.solve(
            make_xs(sidewall_angle=angle),
            wavelength_um=LAMBDA_0_UM,
            num_modes=NUM_MODES,
        )
        for angle in sidewall_angles
    ]

    n_modes_sidewall = min(len(result.modes) for result in sidewall_results)

    print(f"Guided modes across sidewall-angle sweep: {n_modes_sidewall}")

    neff_sidewall = np.array([
        [mode.neff for mode in result.modes[:n_modes_sidewall]]
        for result in sidewall_results
    ])

    nominal_sidewall_index = np.argmin(np.abs(sidewall_angles - SIDEWALL_ANGLE_DEG))

    plt.figure()

    for mode in range(n_modes_sidewall):

        polarization = (sidewall_results[nominal_sidewall_index].modes[mode].polarization)

        plt.plot(
            sidewall_angles,
            np.real(neff_sidewall[:, mode]),
            label=f"Mode {mode} ({polarization})",
        )

    plt.xlabel("Sidewall angle (deg)")
    plt.ylabel("$n_{eff}$")
    plt.title(
        f"$n_{{eff}}$ vs sidewall angle at λ = "
        f"{LAMBDA_0_UM:.3f} µm - {SOLVER_NAME}"
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()

#neff vs altura

if PLOTS["neff_height"]:

    heights = np.linspace(HEIGHT_MIN_UM, HEIGHT_MAX_UM, N_HEIGHT)

    height_results = [
        SOLVER.solve(
            make_xs(height=height),
            wavelength_um=LAMBDA_0_UM,
            num_modes=NUM_MODES,
        )
        for height in heights
    ]

    n_modes_height = min(len(result.modes) for result in height_results)

    print(f"Guided modes across height sweep: {n_modes_height}")

    neff_height = np.array([
        [mode.neff for mode in result.modes[:n_modes_height]]
        for result in height_results
    ])

    nominal_height_index = np.argmin(np.abs(heights - CORE_HEIGHT_UM))

    plt.figure()

    for mode in range(n_modes_height):

        polarization = (height_results[nominal_height_index].modes[mode].polarization)

        plt.plot(
            heights,
            np.real(neff_height[:, mode]),
            label=f"Mode {mode} ({polarization})",
        )

    plt.xlabel("Core height (µm)")
    plt.ylabel("$n_{eff}$")
    plt.title(
        f"$n_{{eff}}$ vs core height at λ = "
        f"{LAMBDA_0_UM:.3f} µm - {SOLVER_NAME}"
    )
    plt.legend()
    plt.grid()
    plt.tight_layout()

#datos en lambda_0

print(f"\nModes at lambda = {LAMBDA_0_UM:.3f} µm\n")

for mode in result_lambda0.modes:
    print(
        f"Mode {mode.index}: "
        f"neff = {mode.neff}, "
        f"polarization = {mode.polarization}, "
        f"TE = {mode.te_fraction:.3f}, "
        f"TM = {mode.tm_fraction:.3f}"
    )

plt.show()