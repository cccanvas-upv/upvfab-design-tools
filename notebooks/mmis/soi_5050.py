"""Diseño y simulación FDTD de un MMI SOI 2x2 50/50.

Flujo implementado:

1. Construir la sección transversal del cuerpo multimodo.
2. Usar directamente la longitud guardada del cuerpo o, opcionalmente,
   recalcularla con FEMWELL a partir de TE0 y TE1.
3. Crear el cuerpo, cuatro tapers y cuatro rectas de puerto.
4. Construir y validar localmente la simulación Tidy3D FDTD.
5. Opcionalmente estimar el coste, ejecutar en la nube o cargar resultados.

Todas las longitudes geométricas están expresadas en micrómetros.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from upvfab_design_tools.core.materials import (
    SILICON,
    THERMAL_SILICON_DIOXIDE,
)
from upvfab_design_tools.core.visualization import plot_cross_section
from upvfab_design_tools.core.waveguides import strip_waveguide
from upvfab_design_tools.tidy3d_plugin import (
    build_mmi_2x2_fdtd_simulation,
    estimate_tidy3d_cost,
    load_tidy3d_simulation_data,
    mmi_2x2_vertices,
    plot_fdtd_field_xy,
    plot_mmi_2x2_vertices,
    test_tidy3d_api,
)

if TYPE_CHECKING:
    from upvfab_design_tools.modesolver import ModeSolverResult


# Parámetros físicos aprobados.
WAVELENGTH_UM = 1.55
WAVELENGTH_MIN_UM = 1.50
WAVELENGTH_MAX_UM = 1.60
WAVELENGTH_STEP_UM = 0.01
CORE_HEIGHT_UM = 0.220
MMI_WIDTH_UM = 4.5
MMI_LENGTH_FACTOR = 0.5
MMI_LENGTH_UM = 24.98 - 2*1.5

ACCESS_WIDTH_UM = 0.5
TAPER_WIDTH_UM = 0.5
TAPER_LENGTH_UM = 5.0
PORT_STRAIGHT_LENGTH_UM = 5.0
IO_Y_POSITION_UM = MMI_WIDTH_UM / 6.0
INPUT_PORT = "bottom"

# Dominio transversal usado tanto por FEMWELL como por la pila vertical FDTD.
DOMAIN_X_SPAN_UM = 8.5
BOTTOM_MARGIN_UM = 2.0
TOP_MARGIN_UM = 2.0
NUM_MODES = 30

# Perfil FDTD aprobado.
FDTD_BANDWIDTH_UM = WAVELENGTH_MAX_UM - WAVELENGTH_MIN_UM
FDTD_MIN_STEPS_PER_WVL = 20
FDTD_RUN_TIME_S = 5e-12
FDTD_PAD_X_UM = 2.0
FDTD_PAD_Y_UM = 2.0
FDTD_PAD_Z_UM = 0.0
MODE_PLANE_Y_SPAN_UM = 1.5
SOURCE_OFFSET_UM = 3.0
MONITOR_OFFSET_UM = 1.0

TIDY3D_DATA_PATH = Path("data/soi_5050_fdtd.hdf5")
TIDY3D_TASK_NAME = "upvfab_soi_mmi_2x2_5050_notap_v2"


def wavelength_grid_um() -> np.ndarray:
    """Devuelve la rejilla aprobada de 1.50 a 1.60 um, extremos incluidos."""

    num_wavelengths = (
        int(
            np.round(
                (WAVELENGTH_MAX_UM - WAVELENGTH_MIN_UM) / WAVELENGTH_STEP_UM
            )
        )
        + 1
    )
    return np.linspace(WAVELENGTH_MIN_UM, WAVELENGTH_MAX_UM, num_wavelengths)


def lpi_from_first_two_te_modes(result: ModeSolverResult) -> float:
    """Calcula Lpi usando los dos primeros modos TE guiados."""

    te_modes = result.te_modes

    if len(te_modes) < 2:
        raise RuntimeError(
            "Se necesitan al menos dos modos TE guiados para calcular Lpi."
        )

    te0, te1 = te_modes[:2]
    delta_beta = abs(float(np.real(te0.beta - te1.beta)))

    if delta_beta <= 0.0:
        raise RuntimeError("TE0 y TE1 tienen la misma constante de propagación.")

    return float(np.pi / delta_beta)


def parse_args() -> argparse.Namespace:
    """Procesa las acciones cloud, que permanecen desactivadas por defecto."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recalculate-modes",
        action="store_true",
        help=(
            "Ejecuta FEMWELL, recalcula Lpi con TE0 y TE1 y usa Lpi/2 "
            "como longitud del cuerpo."
        ),

    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Crea una tarea de estimación en Tidy3D Cloud sin ejecutar FDTD.",
    )
    parser.add_argument(
        "--run-cloud",
        action="store_true",
        help=(
            "Estima el coste, pide confirmación y ejecuta la simulación "
            "en Tidy3D Cloud."
        ),
    )
    parser.add_argument(
        "--load-data",
        action="store_true",
        help=f"Carga y representa resultados guardados en {TIDY3D_DATA_PATH}.",
    )
    return parser.parse_args()


def output_power_metrics(
    transmitted_cross: np.ndarray,
    transmitted_bar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula balance cross/bar y exceso de pérdidas.

    ``balance_cross`` y ``balance_bar`` están normalizados por la potencia total
    recogida en ambos puertos. ``excess_loss_db`` se calcula respecto a la
    potencia incidente normalizada como ``-10*log10(T_cross + T_bar)``.
    """

    transmitted_cross = np.asarray(transmitted_cross, dtype=float)
    transmitted_bar = np.asarray(transmitted_bar, dtype=float)
    transmitted_total = transmitted_cross + transmitted_bar
    valid = transmitted_total > np.finfo(float).eps

    balance_cross = np.divide(
        transmitted_cross,
        transmitted_total,
        out=np.full_like(transmitted_total, np.nan),
        where=valid,
    )
    balance_bar = np.divide(
        transmitted_bar,
        transmitted_total,
        out=np.full_like(transmitted_total, np.nan),
        where=valid,
    )
    excess_loss_db = np.full_like(transmitted_total, np.nan)
    excess_loss_db[valid] = -10.0 * np.log10(transmitted_total[valid])

    return balance_cross, balance_bar, excess_loss_db


def plot_flux_spectrum(sim_data):
    """Dibuja transmisión, reflexión y balance de potencia de las salidas."""

    import tidy3d as td

    frequencies_hz = np.asarray(sim_data["flux_top"].flux.coords["f"].values)
    wavelengths_um = td.C_0 / frequencies_hz
    reflected = -np.asarray(sim_data["flux_reflection"].flux.values, dtype=float)
    transmitted_top = np.asarray(sim_data["flux_top"].flux.values, dtype=float)
    transmitted_bottom = np.asarray(
        sim_data["flux_bottom"].flux.values,
        dtype=float,
    )
    if INPUT_PORT == "bottom":
        transmitted_cross = transmitted_top
        transmitted_bar = transmitted_bottom
    else:
        transmitted_cross = transmitted_bottom
        transmitted_bar = transmitted_top

    transmitted_total = transmitted_cross + transmitted_bar
    balance_cross, balance_bar, excess_loss_db = output_power_metrics(
        transmitted_cross,
        transmitted_bar,
    )

    order = np.argsort(wavelengths_um)
    wavelengths_um = wavelengths_um[order]
    reflected = reflected[order]
    transmitted_cross = transmitted_cross[order]
    transmitted_bar = transmitted_bar[order]
    transmitted_total = transmitted_total[order]
    balance_cross = balance_cross[order]
    balance_bar = balance_bar[order]
    excess_loss_db = excess_loss_db[order]

    fig, (ax_flux, ax_balance, ax_excess_loss) = plt.subplots(
        3,
        1,
        figsize=(8, 10),
        sharex=True,
    )
    ax_flux.plot(wavelengths_um, transmitted_cross, label="T cross")
    ax_flux.plot(wavelengths_um, transmitted_bar, label="T bar")
    ax_flux.plot(
        wavelengths_um,
        transmitted_total,
        label="T total",
        linestyle="--",
    )
    ax_flux.plot(wavelengths_um, reflected, label="R", linestyle=":")
    ax_flux.set_ylabel("Flujo normalizado")
    ax_flux.set_title("Espectro FDTD del MMI SOI 2x2")
    ax_flux.grid(True, alpha=0.3)
    ax_flux.legend()

    ax_balance.plot(
        wavelengths_um,
        100.0 * balance_cross,
        label="Balance cross",
    )
    ax_balance.plot(
        wavelengths_um,
        100.0 * balance_bar,
        label="Balance bar",
    )
    ax_balance.axhline(50.0, color="black", linestyle="--", linewidth=0.9)
    ax_balance.set_ylabel("Proporción de salida [%]")
    ax_balance.set_ylim(0.0, 100.0)
    ax_balance.grid(True, alpha=0.3)
    ax_balance.legend()

    ax_excess_loss.plot(wavelengths_um, excess_loss_db, color="tab:red")
    ax_excess_loss.set_xlabel("Longitud de onda [µm]")
    ax_excess_loss.set_ylabel("Excess Loss [dB]")
    ax_excess_loss.grid(True, alpha=0.3)

    central_index = int(np.argmin(np.abs(wavelengths_um - WAVELENGTH_UM)))
    print()
    print(f"Resultados FDTD en {wavelengths_um[central_index]:.6f} um:")
    print(f"R                 = {reflected[central_index]:.6f}")
    print(f"T_cross           = {transmitted_cross[central_index]:.6f}")
    print(f"T_bar             = {transmitted_bar[central_index]:.6f}")
    print(f"T_total           = {transmitted_total[central_index]:.6f}")
    print(f"Balance cross     = {100.0 * balance_cross[central_index]:.3f} %")
    print(f"Balance bar       = {100.0 * balance_bar[central_index]:.3f} %")
    print(f"Excess Loss       = {excess_loss_db[central_index]:.3f} dB")

    return fig, (ax_flux, ax_balance, ax_excess_loss)


def main() -> None:
    """Construye y, opcionalmente, ejecuta la simulación Tidy3D FDTD."""

    args = parse_args()
    wavelengths_um = wavelength_grid_um()

    mmi_cross_section = strip_waveguide(
        width=MMI_WIDTH_UM,
        height=CORE_HEIGHT_UM,
        core_material=SILICON,
        background_material=THERMAL_SILICON_DIOXIDE,
        x_center=0.0,
        z_min=0.0,
        x_span=DOMAIN_X_SPAN_UM,
        bottom_margin=BOTTOM_MARGIN_UM,
        top_margin=TOP_MARGIN_UM,
        name="soi_mmi_body_4p5um",
    )

    fig_cross_section, ax_cross_section = plot_cross_section(
        mmi_cross_section,
        wavelength_um=WAVELENGTH_UM,
        show_names=True,
        show_indices=True,
    )
    ax_cross_section.set_title("Sección transversal del cuerpo MMI SOI")

    if args.recalculate_modes:
        from upvfab_design_tools.modesolver import FemwellModeSolver

        solver = FemwellModeSolver(
            resolution_overrides={
                "core": {"resolution": 0.03, "distance": 0.25},
                "background": {"resolution": 0.30, "distance": 1.0},
            },
            reference_material=THERMAL_SILICON_DIOXIDE,
            filter_guided=True,
            guided_tolerance=1e-2,
        )

        print("Resolviendo los modos del cuerpo MMI con FEMWELL...", flush=True)
        mmi_modes = solver.solve(
            mmi_cross_section,
            wavelength_um=WAVELENGTH_UM,
            num_modes=NUM_MODES,
        )

        te0, te1 = mmi_modes.te_modes[:2]
        lpi_um = lpi_from_first_two_te_modes(mmi_modes)
        mmi_length_um = MMI_LENGTH_FACTOR * lpi_um

        print(f"n_eff(TE0)      = {te0.neff.real:.8f}")
        print(f"n_eff(TE1)      = {te1.neff.real:.8f}")
        print(f"Lpi             = {lpi_um:.6f} um")
        print("Longitud obtenida mediante FEMWELL.")
    else:
        mmi_length_um = MMI_LENGTH_UM
        print("Cálculo modal omitido; se usa la longitud guardada.")

    print(f"Longitud MMI    = {mmi_length_um:.6f} um")
    print(f"Posiciones I/O  = +/-{IO_Y_POSITION_UM:.6f} um")

    polygons = mmi_2x2_vertices(
        length_um=mmi_length_um,
        io_length_um=TAPER_LENGTH_UM,
        io_y_position_um=IO_Y_POSITION_UM,
        access_width_um=ACCESS_WIDTH_UM,
        taper_width_um=TAPER_WIDTH_UM,
        mmi_width_input_um=MMI_WIDTH_UM,
        mmi_width_center_um=MMI_WIDTH_UM,
        port_straight_length_um=PORT_STRAIGHT_LENGTH_UM,
    )

    fig_layout, ax_layout = plot_mmi_2x2_vertices(polygons)
    ax_layout.set_title("Layout del MMI SOI 2x2 usado por Tidy3D")

    simulation = build_mmi_2x2_fdtd_simulation(
        polygons=polygons,
        cross_section=mmi_cross_section,
        wavelength_um=WAVELENGTH_UM,
        input_port=INPUT_PORT,
        io_y_position_um=IO_Y_POSITION_UM,
        access_width_um=ACCESS_WIDTH_UM,
        bandwidth_um=FDTD_BANDWIDTH_UM,
        monitor_wavelengths_um=wavelengths_um,
        pad_x_um=FDTD_PAD_X_UM,
        pad_y_um=FDTD_PAD_Y_UM,
        pad_z_um=FDTD_PAD_Z_UM,
        source_offset_um=SOURCE_OFFSET_UM,
        monitor_offset_um=MONITOR_OFFSET_UM,
        mode_plane_y_span_um=MODE_PLANE_Y_SPAN_UM,
        min_steps_per_wvl=FDTD_MIN_STEPS_PER_WVL,
        run_time_s=FDTD_RUN_TIME_S,
    )
    simulation.validate_pre_upload()

    core_midplane_um = 0.5 * CORE_HEIGHT_UM
    fig_simulation, ax_simulation = plt.subplots(figsize=(12, 4))
    simulation.plot(z=core_midplane_um, ax=ax_simulation)
    ax_simulation.set_aspect("equal", adjustable="box")
    ax_simulation.set_title("Simulación FDTD: geometría, fuente y monitores")

    fig_epsilon, ax_epsilon = plt.subplots(figsize=(12, 4))
    simulation.plot_eps(z=core_midplane_um, ax=ax_epsilon)
    ax_epsilon.set_aspect("equal", adjustable="box")
    ax_epsilon.set_title("Permitividad FDTD en el plano medio del silicio")

    print(f"Puntos espectrales = {len(wavelengths_um)}")
    print(f"Rango espectral    = {wavelengths_um[0]:.4f}-{wavelengths_um[-1]:.4f} um")
    print(f"Centro simulación  = {simulation.center}")
    print(f"Tamaño simulación  = {simulation.size}")
    print(f"Fuente             = {simulation.sources[0].name}")
    print(f"Monitores          = {[monitor.name for monitor in simulation.monitors]}")

    plt.show()

    sim_data = None

    if args.load_data:
        sim_data = load_tidy3d_simulation_data(TIDY3D_DATA_PATH)

    if args.estimate_cost or args.run_cloud:
        test_tidy3d_api()
        estimated_cost = estimate_tidy3d_cost(
            simulation,
            task_name=f"{TIDY3D_TASK_NAME}_cost",
            verbose=True,
        )
        print(f"Coste estimado: {estimated_cost:.6g} FlexCredits")

        if args.run_cloud:
            confirmation = input(
                "Escribe RUN para aceptar este coste y ejecutar Tidy3D Cloud: "
            ).strip()

            if confirmation == "RUN":
                import tidy3d.web as web

                TIDY3D_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                sim_data = web.run(
                    simulation,
                    task_name=TIDY3D_TASK_NAME,
                    path=str(TIDY3D_DATA_PATH),
                    verbose=True,
                )
            else:
                print("Ejecución cloud cancelada; no se consumieron créditos de simulación.")

    if sim_data is not None:
        plot_flux_spectrum(sim_data)
        fig_field, ax_field = plot_fdtd_field_xy(
            sim_data,
            monitor_name="field_core",
            field_component="Ey",
            field_part="mag",
            polygons=polygons,
        )
        ax_field.set_title("FDTD |Ey| en el plano medio del silicio")
        plt.show()


if __name__ == "__main__":
    main()
