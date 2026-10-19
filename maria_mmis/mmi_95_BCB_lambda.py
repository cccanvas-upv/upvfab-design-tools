import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web


from upvfab_design_tools.tidy3d_plugin import (
    estimate_tidy3d_cost,
    load_tidy3d_simulation_data,
    test_tidy3d_api,
)

from upvfab_design_tools.core.materials import (
    SILICON_NITRIDE,
    THERMAL_SILICON_DIOXIDE,
    BCB,
    AIR,
)


# ============================================================
# CONTROLES
# ============================================================

RUN_TIDY3D_CLOUD = True
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False


TIDY3D_TASK_NAME = (
    "mmi95_bcb_wt2.7_Awplus0.08_L0plus7_dy0_sweep"
)

TIDY3D_DATA_PATH = (
    f"data/{TIDY3D_TASK_NAME}.hdf5"
)


# ============================================================
# WAVELENGTH SWEEP
# ============================================================

lambda0 = 1.55

lambda_min = 1.50
lambda_max = 1.60

NUM_WAVELENGTHS = 101


ldas = np.linspace(
    lambda_min,
    lambda_max,
    NUM_WAVELENGTHS,
)


freq0 = (
    td.C_0
    / lambda0
)


freqs = (
    td.C_0
    / ldas
)


# El source original ya tenia un ancho suficiente
# para cubrir aproximadamente esta ventana espectral

fwidth = (
    0.1
    * freq0
)


# ============================================================
# MATERIALES
# ============================================================

_n_sin = SILICON_NITRIDE.n(
    lambda0
)

n_sin = np.real(
    _n_sin
)


sin = td.Medium(
    permittivity=(
        SILICON_NITRIDE
        .n(lambda0)
        .real ** 2
    )
)


sio2 = td.Medium(
    permittivity=(
        THERMAL_SILICON_DIOXIDE
        .n(lambda0)
        .real ** 2
    )
)


bcb = td.Medium(
    permittivity=(
        BCB
        .n(lambda0)
        .real ** 2
    )
)


air = td.Medium(
    permittivity=(
        AIR
        .n(lambda0)
        .real ** 2
    )
)


# ============================================================
# PARAMETROS MMI
# ============================================================

L_pi_mmi = 153.042918

W_mmi = 10.0


dW = (
    0.72
    + 0.08
)


deltaW = (
    1.0
    - 2.0
    * dW
    / W_mmi
)


W_narrow = (
    W_mmi
    - dW
)


h_core = 0.30


L_wg = 10.0


# ============================================================
# TAPERS
# ============================================================

lt = 75.0

Wt = 2.7

wt = 1.0


# ============================================================
# POSICION DE PUERTOS
# ============================================================

dy = 0.0

# dy > 0 -> puertos mas separados
# dy < 0 -> puertos mas cerca del centro


# ============================================================
# BUFFERS
# ============================================================

buffer_x = 2.0
buffer_y = 3.0
buffer_z = 1.5


# ============================================================
# STACK VERTICAL
# ============================================================

h_bcb = 0.4

h_air_top = 2.0

h_sio2 = 2.0


# SiN:
# z = 0 ... 0.30 um

z_core_min = 0.0
z_core_max = h_core


# BCB:
# z = 0 ... 0.40 um

z_bcb_min = 0.0
z_bcb_max = h_bcb


# Dominio total

z_top = (
    h_bcb
    + h_air_top
)

z_bottom = (
    -h_sio2
)


Lz = (
    z_top
    - z_bottom
)


sim_center_z = (
    0.5
    * (
        z_top
        + z_bottom
    )
)


# ============================================================
# CONTROLES NUMERICOS
# ============================================================

min_steps_per_wvl = 12

sim_run_time = 5e-12


# Menor que la separacion entre los dos puertos
# para evitar que los planos se solapen

port_size_y = (
    3.0
    * wt
)


# ============================================================
# GEOMETRIA
# ============================================================

def b_symmetric_body_vertices(
    L_mmi,
    W_mmi,
    W_narrow,
):

    return np.array(
        [
            # Parte inferior

            (
                0.0,
                -W_mmi / 2,
            ),

            (
                L_mmi / 4,
                -W_narrow / 2,
            ),

            (
                L_mmi / 2,
                -W_mmi / 2,
            ),

            (
                3 * L_mmi / 4,
                -W_narrow / 2,
            ),

            (
                L_mmi,
                -W_mmi / 2,
            ),

            # Parte superior

            (
                L_mmi,
                W_mmi / 2,
            ),

            (
                3 * L_mmi / 4,
                W_narrow / 2,
            ),

            (
                L_mmi / 2,
                W_mmi / 2,
            ),

            (
                L_mmi / 4,
                W_narrow / 2,
            ),

            (
                0.0,
                W_mmi / 2,
            ),
        ]
    )


def taper_bottom_left_vertices(
    W_mmi,
    lt,
    wt,
    Wt,
    dy,
):

    yc = -(
        W_mmi / 6
        + dy
    )

    return np.array(
        [
            (
                -lt,
                yc - wt / 2,
            ),

            (
                0.0,
                yc - Wt / 2,
            ),

            (
                0.0,
                yc + Wt / 2,
            ),

            (
                -lt,
                yc + wt / 2,
            ),
        ]
    )


def taper_top_left_vertices(
    W_mmi,
    lt,
    wt,
    Wt,
    dy,
):

    yc = (
        W_mmi / 6
        + dy
    )

    return np.array(
        [
            (
                -lt,
                yc - wt / 2,
            ),

            (
                0.0,
                yc - Wt / 2,
            ),

            (
                0.0,
                yc + Wt / 2,
            ),

            (
                -lt,
                yc + wt / 2,
            ),
        ]
    )


def taper_bottom_right_vertices(
    L_mmi,
    W_mmi,
    lt,
    wt,
    Wt,
    dy,
):

    yc = -(
        W_mmi / 6
        + dy
    )

    return np.array(
        [
            (
                L_mmi,
                yc - Wt / 2,
            ),

            (
                L_mmi + lt,
                yc - wt / 2,
            ),

            (
                L_mmi + lt,
                yc + wt / 2,
            ),

            (
                L_mmi,
                yc + Wt / 2,
            ),
        ]
    )


def taper_top_right_vertices(
    L_mmi,
    W_mmi,
    lt,
    wt,
    Wt,
    dy,
):

    yc = (
        W_mmi / 6
        + dy
    )

    return np.array(
        [
            (
                L_mmi,
                yc - Wt / 2,
            ),

            (
                L_mmi + lt,
                yc - wt / 2,
            ),

            (
                L_mmi + lt,
                yc + wt / 2,
            ),

            (
                L_mmi,
                yc + Wt / 2,
            ),
        ]
    )


def wg_bottom_left_vertices(
    W_mmi,
    L_wg,
    lt,
    wt,
    dy,
):

    yc = -(
        W_mmi / 6
        + dy
    )

    return np.array(
        [
            (
                -lt - L_wg,
                yc - wt / 2,
            ),

            (
                -lt,
                yc - wt / 2,
            ),

            (
                -lt,
                yc + wt / 2,
            ),

            (
                -lt - L_wg,
                yc + wt / 2,
            ),
        ]
    )


def wg_top_left_vertices(
    W_mmi,
    L_wg,
    lt,
    wt,
    dy,
):

    yc = (
        W_mmi / 6
        + dy
    )

    return np.array(
        [
            (
                -lt - L_wg,
                yc - wt / 2,
            ),

            (
                -lt,
                yc - wt / 2,
            ),

            (
                -lt,
                yc + wt / 2,
            ),

            (
                -lt - L_wg,
                yc + wt / 2,
            ),
        ]
    )


def wg_bottom_right_vertices(
    L_mmi,
    W_mmi,
    L_wg,
    lt,
    wt,
    dy,
):

    yc = -(
        W_mmi / 6
        + dy
    )

    return np.array(
        [
            (
                L_mmi + lt,
                yc - wt / 2,
            ),

            (
                L_mmi + lt + L_wg,
                yc - wt / 2,
            ),

            (
                L_mmi + lt + L_wg,
                yc + wt / 2,
            ),

            (
                L_mmi + lt,
                yc + wt / 2,
            ),
        ]
    )


def wg_top_right_vertices(
    L_mmi,
    W_mmi,
    L_wg,
    lt,
    wt,
    dy,
):

    yc = (
        W_mmi / 6
        + dy
    )

    return np.array(
        [
            (
                L_mmi + lt,
                yc - wt / 2,
            ),

            (
                L_mmi + lt + L_wg,
                yc - wt / 2,
            ),

            (
                L_mmi + lt + L_wg,
                yc + wt / 2,
            ),

            (
                L_mmi + lt,
                yc + wt / 2,
            ),
        ]
    )


# ============================================================
# CREAR SIMULACION
# ============================================================


L_0 = (
    deltaW
    * L_pi_mmi
)


L_mmi = (
    L_0
    + 7.0
)

def make_sim(
    L_mmi=L_mmi,
    W_mmi=W_mmi,
    W_narrow=W_narrow,
    dy=dy,
):


    # ========================================================
    # POSICIONES PUERTOS
    # ========================================================

    y_top = (
        W_mmi / 6
        + dy
    )

    y_bottom = -(
        W_mmi / 6
        + dy
    )


    # ========================================================
    # DOMINIO
    # ========================================================

    Lx = (
        L_mmi
        + 2 * lt
        + 2 * buffer_x
        + 2 * L_wg
    )

    Ly = (
        W_mmi
        + 2 * buffer_y
    )


    sim_size = (
        Lx,
        Ly,
        Lz,
    )


    sim_center = (
        L_mmi / 2,
        0,
        sim_center_z,
    )


    # ========================================================
    # SUBSTRATO SiO2
    # ========================================================

    sio2_substrate = td.Structure(
        geometry=td.Box(
            center=(
                L_mmi / 2,
                0,
                -0.5 * h_sio2,
            ),

            size=(
                td.inf,
                td.inf,
                h_sio2,
            ),
        ),

        medium=sio2,

        name="sio2_substrate",
    )


    # ========================================================
    # BCB
    # ========================================================

    bcb_layer = td.Structure(
        geometry=td.Box(
            center=(
                L_mmi / 2,
                0,
                0.5 * h_bcb,
            ),

            size=(
                td.inf,
                td.inf,
                h_bcb,
            ),
        ),

        medium=bcb,

        name="bcb_layer",
    )


    # ========================================================
    # GUIAS
    # ========================================================

    wg_bottom_left = td.Structure(
        geometry=td.PolySlab(
            vertices=(
                wg_bottom_left_vertices(
                    W_mmi,
                    L_wg,
                    lt,
                    wt,
                    dy,
                )
            ),

            axis=2,

            slab_bounds=(
                0.0,
                h_core,
            ),
        ),

        medium=sin,
    )


    wg_top_left = td.Structure(
        geometry=td.PolySlab(
            vertices=(
                wg_top_left_vertices(
                    W_mmi,
                    L_wg,
                    lt,
                    wt,
                    dy,
                )
            ),

            axis=2,

            slab_bounds=(
                0.0,
                h_core,
            ),
        ),

        medium=sin,
    )


    wg_bottom_right = td.Structure(
        geometry=td.PolySlab(
            vertices=(
                wg_bottom_right_vertices(
                    L_mmi,
                    W_mmi,
                    L_wg,
                    lt,
                    wt,
                    dy,
                )
            ),

            axis=2,

            slab_bounds=(
                0.0,
                h_core,
            ),
        ),

        medium=sin,
    )


    wg_top_right = td.Structure(
        geometry=td.PolySlab(
            vertices=(
                wg_top_right_vertices(
                    L_mmi,
                    W_mmi,
                    L_wg,
                    lt,
                    wt,
                    dy,
                )
            ),

            axis=2,

            slab_bounds=(
                0.0,
                h_core,
            ),
        ),

        medium=sin,
    )


    # ========================================================
    # TAPERS
    # ========================================================

    taper_bottom_left = td.Structure(
        geometry=td.PolySlab(
            vertices=(
                taper_bottom_left_vertices(
                    W_mmi,
                    lt,
                    wt,
                    Wt,
                    dy,
                )
            ),

            axis=2,

            slab_bounds=(
                0.0,
                h_core,
            ),
        ),

        medium=sin,
    )


    taper_top_left = td.Structure(
        geometry=td.PolySlab(
            vertices=(
                taper_top_left_vertices(
                    W_mmi,
                    lt,
                    wt,
                    Wt,
                    dy,
                )
            ),

            axis=2,

            slab_bounds=(
                0.0,
                h_core,
            ),
        ),

        medium=sin,
    )


    taper_bottom_right = td.Structure(
        geometry=td.PolySlab(
            vertices=(
                taper_bottom_right_vertices(
                    L_mmi,
                    W_mmi,
                    lt,
                    wt,
                    Wt,
                    dy,
                )
            ),

            axis=2,

            slab_bounds=(
                0.0,
                h_core,
            ),
        ),

        medium=sin,
    )


    taper_top_right = td.Structure(
        geometry=td.PolySlab(
            vertices=(
                taper_top_right_vertices(
                    L_mmi,
                    W_mmi,
                    lt,
                    wt,
                    Wt,
                    dy,
                )
            ),

            axis=2,

            slab_bounds=(
                0.0,
                h_core,
            ),
        ),

        medium=sin,
    )


    # ========================================================
    # MMI
    # ========================================================

    mmi_body = td.Structure(
        geometry=td.PolySlab(
            vertices=(
                b_symmetric_body_vertices(
                    L_mmi,
                    W_mmi,
                    W_narrow,
                )
            ),

            axis=2,

            slab_bounds=(
                0.0,
                h_core,
            ),
        ),

        medium=sin,
    )


    # ========================================================
    # MODE SPEC
    # ========================================================

    mode_spec = td.ModeSpec(
        num_modes=2,

        target_neff=(
            n_sin
        ),
    )


    # ========================================================
    # POSICIONES SOURCE / MONITORS
    # ========================================================

    source_x = (
        -lt
        - L_wg / 2
    )


    monitor_x = (
        L_mmi
        + lt
        + L_wg / 2
    )


    source_y = (
        y_top
    )


    monitor_top_y = (
        y_top
    )


    monitor_bottom_y = (
        y_bottom
    )


    # ========================================================
    # MODE SOURCE
    # ========================================================

    mode_source = td.ModeSource(
        center=(
            source_x,
            source_y,
            sim_center_z,
        ),

        size=(
            0,
            port_size_y,
            Lz,
        ),

        source_time=td.GaussianPulse(
            freq0=freq0,
            fwidth=fwidth,
        ),

        direction="+",

        mode_spec=mode_spec,

        mode_index=0,

        num_freqs=5,
    )


    # ========================================================
    # FLUX MONITORS
    # ========================================================

    flux_monitor_top = td.FluxMonitor(
        center=(
            monitor_x,
            monitor_top_y,
            sim_center_z,
        ),

        size=(
            0,
            port_size_y,
            Lz,
        ),

        freqs=freqs,

        name="flux_top",
    )


    flux_monitor_bottom = td.FluxMonitor(
        center=(
            monitor_x,
            monitor_bottom_y,
            sim_center_z,
        ),

        size=(
            0,
            port_size_y,
            Lz,
        ),

        freqs=freqs,

        name="flux_bottom",
    )


    # ========================================================
    # MODE MONITORS
    # ========================================================

    mode_monitor_top = td.ModeMonitor(
        center=(
            monitor_x,
            monitor_top_y,
            sim_center_z,
        ),

        size=(
            0,
            port_size_y,
            Lz,
        ),

        freqs=freqs,

        mode_spec=mode_spec,

        name="mode_top",
    )


    mode_monitor_bottom = td.ModeMonitor(
        center=(
            monitor_x,
            monitor_bottom_y,
            sim_center_z,
        ),

        size=(
            0,
            port_size_y,
            Lz,
        ),

        freqs=freqs,

        mode_spec=mode_spec,

        name="mode_bottom",
    )


    # ========================================================
    # FIELD MONITOR
    # ========================================================

    # Solo guardamos el field completo a lambda0.

    field_monitor = td.FieldMonitor(
        center=(
            L_mmi / 2,
            0,
            0.5 * h_core,
        ),

        size=(
            L_mmi + 2 * lt,
            W_mmi + 1.0,
            0,
        ),

        freqs=[
            freq0
        ],

        name="field",
    )


    # ========================================================
    # SIMULATION
    # ========================================================

    sim = td.Simulation(
        center=sim_center,

        size=sim_size,

        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=(
                min_steps_per_wvl
            ),

            # Mantengo la misma grid del monocromatico.
            wavelength=lambda0,
        ),

        structures=[
            sio2_substrate,
            bcb_layer,

            wg_bottom_left,
            wg_top_left,

            wg_bottom_right,
            wg_top_right,

            taper_bottom_left,
            taper_bottom_right,

            mmi_body,

            taper_top_right,
            taper_top_left,
        ],

        sources=[
            mode_source
        ],

        monitors=[
            field_monitor,

            flux_monitor_top,
            flux_monitor_bottom,

            mode_monitor_top,
            mode_monitor_bottom,
        ],

        run_time=(
            sim_run_time
        ),

        boundary_spec=(
            td.BoundarySpec.all_sides(
                boundary=td.PML()
            )
        ),

        # Todo lo que no esta ocupado
        # por SiO2, BCB o SiN es aire.

        medium=air,
    )


    return sim


# ============================================================
# LONGITUD MMI
# ============================================================



print(
    f"\nL_pi = "
    f"{L_pi_mmi:.6f} um"
)

print(
    f"L_0 = "
    f"{L_0:.6f} um"
)

print(
    f"L_mmi = "
    f"{L_mmi:.6f} um"
)


# ============================================================
# CREAR SIMULACION
# ============================================================

sim = make_sim(
    L_mmi,
    W_mmi,
    W_narrow,
)


# ============================================================
# VALIDACION
# ============================================================

sim.validate_pre_upload()

print(
    "\nSimulation validated successfully."
)


# ============================================================
# PLOT XY EN EL CENTRO DEL CORE
# ============================================================

fig, ax = plt.subplots(
    figsize=(24, 6)
)

sim.plot(
    z=(
        0.5
        * h_core
    ),

    ax=ax,
)

ax.set_title(
    "MMI geometry at z = 0.15 um"
)

plt.tight_layout()


# ============================================================
# PLOT DE PERMITIVIDAD XY
# ============================================================

fig_eps, ax_eps = plt.subplots(
    figsize=(24, 6)
)

sim.plot_eps(
    z=(
        0.5
        * h_core
    ),

    ax=ax_eps,
)

ax_eps.set_title(
    "Permittivity at z = 0.15 um"
)

plt.tight_layout()


# ============================================================
# PLOT VERTICAL DE LA CROSS SECTION
# ============================================================

SOURCE_X = (
    -lt
    - L_wg / 2
)


fig_cross, ax_cross = plt.subplots(
    figsize=(7, 5)
)

sim.plot_eps(
    x=SOURCE_X,
    ax=ax_cross,
)

ax_cross.set_title(
    "Vertical cross section at input waveguide"
)

plt.tight_layout()


# ============================================================
# COSTE
# ============================================================

if ESTIMATE_TIDY3D_COST:

    test_tidy3d_api()

    estimated_cost = (
        estimate_tidy3d_cost(
            sim,

            task_name=(
                f"{TIDY3D_TASK_NAME}_cost"
            ),

            verbose=True,
        )
    )

    print(
        f"\nEstimated Tidy3D cost: "
        f"{estimated_cost:.6g} FlexCredits"
    )


# ============================================================
# DATA
# ============================================================

sim_data = None


# ============================================================
# CARGAR RESULTADOS EXISTENTES
# ============================================================

if LOAD_TIDY3D_DATA:

    print(
        f"\nLoading Tidy3D data from:"
        f"\n{TIDY3D_DATA_PATH}"
    )

    sim_data = (
        load_tidy3d_simulation_data(
            TIDY3D_DATA_PATH
        )
    )


# ============================================================
# EJECUTAR CLOUD
# ============================================================

if RUN_TIDY3D_CLOUD:

    test_tidy3d_api()

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RUNNING TIDY3D CLOUD SIMULATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Task name: "
        f"{TIDY3D_TASK_NAME}"
    )

    print(
        f"Data path: "
        f"{TIDY3D_DATA_PATH}"
    )


    sim_data = web.run(
        sim,

        task_name=(
            TIDY3D_TASK_NAME
        ),

        path=(
            TIDY3D_DATA_PATH
        ),

        verbose=True,
    )


# ============================================================
# POSTPROCESADO ESPECTRAL
# ============================================================

if sim_data is not None:

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FDTD SPECTRAL RESULTS"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # FLUX
    # ========================================================

    flux_top_data = (
        sim_data["flux_top"].flux
    )


    flux_bottom_data = (
        sim_data["flux_bottom"].flux
    )


    frequencies = np.asarray(
        flux_top_data.f.values,
        dtype=float,
    )


    wavelengths_um = (
        td.C_0
        / frequencies
    )


    T_top = np.real(
        np.asarray(
            flux_top_data.values
        ).squeeze()
    )


    T_bottom = np.real(
        np.asarray(
            flux_bottom_data.values
        ).squeeze()
    )


    # ========================================================
    # ORDENAR POR LONGITUD DE ONDA
    # ========================================================

    order = np.argsort(
        wavelengths_um
    )


    wavelengths_um = (
        wavelengths_um[
            order
        ]
    )


    T_top = (
        T_top[
            order
        ]
    )


    T_bottom = (
        T_bottom[
            order
        ]
    )


    # ========================================================
    # POTENCIA TOTAL
    # ========================================================

    T_total = (
        T_top
        + T_bottom
    )


    # ========================================================
    # SPLITTING RATIO
    # ========================================================

    K_top = np.divide(
        T_top,

        T_total,

        out=np.zeros_like(
            T_top
        ),

        where=(
            T_total > 0
        ),
    )


    K_bottom = np.divide(
        T_bottom,

        T_total,

        out=np.zeros_like(
            T_bottom
        ),

        where=(
            T_total > 0
        ),
    )


    # ========================================================
    # EXCESS LOSS
    # ========================================================

    excess_loss_db = (
        -10.0
        * np.log10(
            np.maximum(
                T_total,
                1e-20,
            )
        )
    )


    # ========================================================
    # FASE TE0
    # ========================================================

    a_top = np.asarray(
        sim_data["mode_top"].amps
        .sel(
            direction="+",
            mode_index=0,
        )
        .values
    ).squeeze()


    a_bottom = np.asarray(
        sim_data["mode_bottom"].amps
        .sel(
            direction="+",
            mode_index=0,
        )
        .values
    ).squeeze()


    a_top = (
        a_top[
            order
        ]
    )


    a_bottom = (
        a_bottom[
            order
        ]
    )


    phase_difference_rad = np.angle(
        a_top
        * np.conj(
            a_bottom
        )
    )


    phase_difference_deg = np.degrees(
        np.unwrap(
            phase_difference_rad
        )
    )


    # ========================================================
    # RESULTADO EN lambda0
    # ========================================================

    index_lambda0 = np.argmin(
        np.abs(
            wavelengths_um
            - lambda0
        )
    )


    print(
        "\n--- RESULTS AT lambda0 ---"
    )


    print(
        f"lambda = "
        f"{wavelengths_um[index_lambda0]:.6f} um"
    )


    print(
        f"T_top = "
        f"{T_top[index_lambda0]:.6f}"
    )


    print(
        f"T_bottom = "
        f"{T_bottom[index_lambda0]:.6f}"
    )


    print(
        f"T_total = "
        f"{T_total[index_lambda0]:.6f}"
    )


    print(
        f"K_top = "
        f"{K_top[index_lambda0]:.6f}"
    )


    print(
        f"K_bottom = "
        f"{K_bottom[index_lambda0]:.6f}"
    )


    print(
        f"Splitting ratio = "
        f"{100 * K_top[index_lambda0]:.2f}:"
        f"{100 * K_bottom[index_lambda0]:.2f}"
    )


    print(
        f"Excess loss = "
        f"{excess_loss_db[index_lambda0]:.4f} dB"
    )


    print(
        f"Phase TOP - BOTTOM = "
        f"{phase_difference_deg[index_lambda0]:+.4f} deg"
    )


    # ========================================================
    # BANDWIDTH
    # ========================================================

    COUPLING_TARGET = 0.95

    COUPLING_TOLERANCE = 0.01


    lower_limit = (
        COUPLING_TARGET
        - COUPLING_TOLERANCE
    )


    upper_limit = (
        COUPLING_TARGET
        + COUPLING_TOLERANCE
    )


    # El puerto dominante en este diseño es BOTTOM.

    valid = (
        (K_bottom >= lower_limit)
        & (K_bottom <= upper_limit)
    )


    if valid[index_lambda0]:

        i_left = (
            index_lambda0
        )

        i_right = (
            index_lambda0
        )


        while (
            i_left > 0
            and valid[
                i_left - 1
            ]
        ):

            i_left -= 1


        while (
            i_right
            < len(valid) - 1

            and valid[
                i_right + 1
            ]
        ):

            i_right += 1


        lambda_bw_min = (
            wavelengths_um[
                i_left
            ]
        )


        lambda_bw_max = (
            wavelengths_um[
                i_right
            ]
        )


        bandwidth_um = (
            lambda_bw_max
            - lambda_bw_min
        )


        bandwidth_nm = (
            1000.0
            * bandwidth_um
        )


        bandwidth_hits_left_edge = (
            i_left == 0
        )


        bandwidth_hits_right_edge = (
            i_right
            == len(valid) - 1
        )


    else:

        lambda_bw_min = np.nan

        lambda_bw_max = np.nan

        bandwidth_um = 0.0

        bandwidth_nm = 0.0

        bandwidth_hits_left_edge = False

        bandwidth_hits_right_edge = False


    # ========================================================
    # 1. BANDWIDTH
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        wavelengths_um,
        K_top,
        label="Top output",
        linewidth=2,
    )


    plt.plot(
        wavelengths_um,
        K_bottom,
        label="Bottom output",
        linewidth=2,
    )


    plt.axhline(
        0.95,

        linestyle="--",

        linewidth=1,

        label="Target 95%",
    )


    plt.axhline(
        0.05,

        linestyle="--",

        linewidth=1,

        label="Target 5%",
    )


    plt.axhline(
        lower_limit,

        linestyle=":",

        linewidth=1,
    )


    plt.axhline(
        upper_limit,

        linestyle=":",

        linewidth=1,
    )


    if bandwidth_um > 0.0:

        plt.axvspan(
            lambda_bw_min,
            lambda_bw_max,

            alpha=0.2,

            label=(
                f"Bandwidth = "
                f"{bandwidth_nm:.1f} nm"
            ),
        )


    plt.axvline(
        lambda0,

        linestyle="--",

        linewidth=1,

        label=r"$\lambda_0$",
    )


    plt.xlabel(
        "Wavelength (µm)"
    )


    plt.ylabel(
        "Coupling ratio"
    )


    plt.title(
        "MMI spectral bandwidth"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()

    plt.tight_layout()


    # ========================================================
    # 2. COUPLING RATIO
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        wavelengths_um,
        K_top,

        label="Top output",

        linewidth=2,
    )


    plt.plot(
        wavelengths_um,
        K_bottom,

        label="Bottom output",

        linewidth=2,
    )


    plt.axhline(
        0.95,

        linestyle="--",

        linewidth=1,
    )


    plt.axhline(
        0.05,

        linestyle="--",

        linewidth=1,
    )


    plt.axvline(
        lambda0,

        linestyle="--",

        linewidth=1,

        label=r"$\lambda_0$",
    )


    plt.xlabel(
        "Wavelength (µm)"
    )


    plt.ylabel(
        "Coupling ratio"
    )


    plt.title(
        "Coupling ratio vs wavelength"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()

    plt.tight_layout()


    # ========================================================
    # 3. TOTAL TRANSMITTED POWER
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        wavelengths_um,
        T_total,

        linewidth=2,
    )


    plt.axvline(
        lambda0,

        linestyle="--",

        linewidth=1,
    )


    plt.xlabel(
        "Wavelength (µm)"
    )


    plt.ylabel(
        "Total transmitted power"
    )


    plt.title(
        "Total transmitted power vs wavelength"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.tight_layout()


    # ========================================================
    # 4. EXCESS LOSS
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        wavelengths_um,
        excess_loss_db,

        linewidth=2,
    )


    plt.axvline(
        lambda0,

        linestyle="--",

        linewidth=1,
    )


    plt.xlabel(
        "Wavelength (µm)"
    )


    plt.ylabel(
        "Excess loss (dB)"
    )


    plt.title(
        "Excess loss vs wavelength"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.tight_layout()


    # ========================================================
    # 5. FASE RELATIVA
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        wavelengths_um,
        phase_difference_deg,

        linewidth=2,
    )


    plt.axvline(
        lambda0,

        linestyle="--",

        linewidth=1,

        label=r"$\lambda_0$",
    )


    plt.xlabel(
        "Wavelength (µm)"
    )


    plt.ylabel(
        "Phase TOP - BOTTOM (deg)"
    )


    plt.title(
        "Relative output phase vs wavelength"
    )


    plt.grid(
        True,
        alpha=0.3,
    )


    plt.legend()

    plt.tight_layout()


    # ========================================================
    # FIELD EN lambda0
    # ========================================================

    fig_field, ax_field = plt.subplots(
        figsize=(24, 6)
    )


    sim_data.plot_field(
        field_monitor_name="field",

        field_name="E",

        val="abs^2",

        f=freq0,

        ax=ax_field,
    )


    ax_field.set_title(
        "|E|² at z = 0.15 um"
    )


    # ========================================================
    # RESUMEN
    # ========================================================

    print(
        "\n--- BANDWIDTH ---"
    )


    print(
        f"Target coupling: "
        f"{100 * COUPLING_TARGET:.2f}%"
    )


    print(
        f"Tolerance: "
        f"±{100 * COUPLING_TOLERANCE:.2f}%"
    )


    print(
        f"lambda_min: "
        f"{lambda_bw_min:.4f} um"
    )


    print(
        f"lambda_max: "
        f"{lambda_bw_max:.4f} um"
    )


    print(
        f"Bandwidth: "
        f"{bandwidth_nm:.2f} nm"
    )


    if (
        bandwidth_hits_left_edge
        or bandwidth_hits_right_edge
    ):

        print(
            "NOTE: the valid bandwidth reaches "
            "the edge of the simulated wavelength range."
        )

        print(
            "The real bandwidth may therefore be larger."
        )


    print(
        f"\nAverage excess loss: "
        f"{excess_loss_db.mean():.4f} dB"
    )


    print(
        f"Minimum excess loss: "
        f"{excess_loss_db.min():.4f} dB"
    )


    print(
        f"Maximum excess loss: "
        f"{excess_loss_db.max():.4f} dB"
    )


plt.show()