import numpy as np
from matplotlib import pyplot as plt

from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.waveguides import strip_waveguide
from upvfab_design_tools.core.visualization import plot_cross_section

from upvfab_design_tools.modesolver import (
    FemwellModeSolver,
    ModeSolverResult,
    Tidy3DModeSolver,
    plot_modes_grid,
)

#parámetros cross_section

LAMBDA_0_UM = 1.55

NOMINAL_MMI_WIDTH_UM = 12.0
MMI_WIDTH_DEV_UM = 0.0
MMI_WIDTH_UM = NOMINAL_MMI_WIDTH_UM + MMI_WIDTH_DEV_UM

NOMINAL_CORE_HEIGHT_UM = 0.3
CORE_HEIGHT_DEV_UM = 0.0
CORE_HEIGHT_UM = NOMINAL_CORE_HEIGHT_UM + CORE_HEIGHT_DEV_UM

NOMINAL_SIDEWALL_ANGLE_DEG = 20.0 #no estandarizado, G me ha dicho que se han obtenido entre 70-80º (se traducen a 30-20º en la convencción de la xs creo)
SIDEWALL_ANGLE_DEV_DEG = 0.0
SIDEWALL_ANGLE_DEG = NOMINAL_SIDEWALL_ANGLE_DEG + SIDEWALL_ANGLE_DEV_DEG

# parámetros simulación

X_MARGIN_UM = 2.0
DOMAIN_X_SPAN_UM = MMI_WIDTH_UM + 2 * X_MARGIN_UM

BOTTOM_MARGIN_UM = 1.5
TOP_MARGIN_UM = 1.5

#solver 
SOLVER_TYPE = 1 # 0 = femwell, 1 = tidy
NUM_MODES = 20 # provisional, revisar convergencia
MAX_MODES_PLOT = 6

#gráficas: para elegir las que queremos en cada momento

PLOTS = {
    "cross_section": True,
    "te_modes": False,
}

# cross-section

def make_mmi_xs(
    width=MMI_WIDTH_UM,
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
        x_span=width + 2 * X_MARGIN_UM,
        bottom_margin=BOTTOM_MARGIN_UM,
        top_margin=TOP_MARGIN_UM,
        name="MMI",
    )

mmi_xs = make_mmi_xs()


if PLOTS["cross_section"]:

    fig, ax = plot_cross_section(
        mmi_xs,
        wavelength_um=LAMBDA_0_UM,
        show_names=False,
        show_indices=True,
    )

    ax.set_title(
        f"MMI cross-section - W = {MMI_WIDTH_UM:.2f} µm"
    )

#cálculo l_pi, de momento asumo que buscamos l_pi_te

def te_mode_result(result: ModeSolverResult) -> ModeSolverResult:
    """Return a result containing only TE-like modes."""

    if not result.te_modes:
        raise RuntimeError("The mode solve returned no guided TE-like modes.")

    return ModeSolverResult(
        modes=result.te_modes,
        wavelength_um=result.wavelength_um,
        backend=result.backend,
        cross_section=result.cross_section,
        metadata=result.metadata,
    )


def lpi_from_first_two_modes(result: ModeSolverResult) -> float:
    """Calculate MMI beat length from the first two modes."""

    if len(result) < 2:
        raise RuntimeError("At least two modes are required to calculate Lpi.")

    beta0 = complex(result[0].beta)
    beta1 = complex(result[1].beta)

    delta_beta = abs(np.real(beta0 - beta1))

    if delta_beta <= 0:
        raise RuntimeError("Could not calculate Lpi because beta0 = beta1.")

    return float(np.pi / delta_beta)

#solvers

solver_femwell = FemwellModeSolver(
    default_resolution=0.3,
    min_resolution=0.01,
    resolution_factor=10.0,
    filter_guided=True,
    reference_material=THERMAL_SILICON_DIOXIDE,
    guided_tolerance=1e-2,
    enable_plots=False,
)

solver_tidy = Tidy3DModeSolver(
    min_steps_per_wvl=30,
    reference_material=THERMAL_SILICON_DIOXIDE,
    filter_guided=True,
    guided_tolerance=1e-2,
)

if SOLVER_TYPE == 0:
    SOLVER = solver_femwell
    SOLVER_NAME = "Femwell"
else:
    SOLVER = solver_tidy
    SOLVER_NAME = "Tidy3D"

result = SOLVER.solve(
    mmi_xs,
    wavelength_um=LAMBDA_0_UM,
    num_modes=NUM_MODES,
)

result_te = te_mode_result(result)
L_PI_UM = lpi_from_first_two_modes(result_te)

print(f"\nSolver: {SOLVER_NAME}")
print(f"MMI width: {MMI_WIDTH_UM:.3f} µm")
print(f"Wavelength: {LAMBDA_0_UM:.3f} µm")
print(f"Guided modes: {len(result.modes)}")
print(f"TE-like modes: {len(result_te.modes)}")
for mode in result_te.modes[:2]:
    print(
        f"TE mode {mode.index}: "
        f"neff = {np.real(mode.neff):.6f}, "
        f"beta = {np.real(mode.beta):.6f}"
    )
print(f"Lpi (TE): {L_PI_UM:.3f} µm")

if PLOTS["te_modes"]:

    fig, axs = plot_modes_grid(
        result_te,
        field_component="auto",
        field_part="mag",
        max_modes=min(MAX_MODES_PLOT, len(result_te.modes)),
        xlim=(-MMI_WIDTH_UM / 2 - 1, MMI_WIDTH_UM / 2 + 1),
        zlim=(-1, 1),
    )

    fig.suptitle(
        f"TE modes - MMI W = {MMI_WIDTH_UM:.1f} µm - {SOLVER_NAME}"
    )

plt.show()