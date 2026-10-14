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
)

RUN_TIDY3D_CLOUD = False
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = True

TIDY3D_TASK_NAME = "mmi95_lambda_0"

# Archivo local donde se guardaran los resultados
TIDY3D_DATA_PATH = f"data/{TIDY3D_TASK_NAME}.hdf5"

################ MATERIALES
sin, sin_rms = SILICON_NITRIDE.fit_tidy(
    wavelength_min_um=1.5,
    wavelength_max_um=1.6,
    num_samples=201,
    max_num_poles=2,
    tolerance_rms=1e-4,
)

sio2, sio2_rms = THERMAL_SILICON_DIOXIDE.fit_tidy(
    wavelength_min_um=1.5,
    wavelength_max_um=1.6,
    num_samples=201,
    max_num_poles=2,
    tolerance_rms=1e-4,
)

_n_sin = SILICON_NITRIDE.n(1.55)
n_sin = np.real(_n_sin)

# sin = td.Medium(
#     permittivity=(
#         SILICON_NITRIDE.n(1.55).real ** 2
#     )
# )

# sio2 = td.Medium(
#     permittivity=(
#         THERMAL_SILICON_DIOXIDE.n(1.55).real ** 2
#     )
# )
#######################################################

L_pi_mmi = 153.512432 #recalcular con los modesolvers de designtools si se considera necesario
W_mmi = 10
dW =  0.72 +0.08
deltaW = 1 - 2*dW/W_mmi
L_mmi = deltaW * (1/3)* 3*L_pi_mmi
W_narrow = W_mmi - dW
h_core = 0.30 
L_wg = 10.0
#tapers 
lt = 75.0 # um, taper length
Wt = 2.7 #IMPORTANTE NO SUPERARLO #taper width at mmi interface
wt = 1 # access wvg

dy = -0.0 # um.
                        # dy > 0 -> los puertos se separan más
                        # dy < 0 -> los puertos se acercan al centro

#wavelengths
lambda0 = 1.55
lambda_min = 1.5
lambda_max = 1.6
ldas = np.linspace(lambda_min, lambda_max, 101)
freq0 = td.C_0 / lambda0
freqs = td.C_0 / ldas
fwidth = 0.5 * (np.max(freqs) - np.min(freqs))
#buffers: márgenes entre dispositivo y bordes de la simulación
buffer_x = 2.0
buffer_y = 3.0
buffer_z = 1.5
#simulation height 
h_air_top = 2.5   # um, aire por encima del core
h_sio2 = 2.0      # um, SiO2 por debajo del core
z_top = h_core / 2 + h_air_top
z_bottom = -h_core / 2 - h_sio2
Lz = z_top - z_bottom
sim_center_z = (z_top + z_bottom) / 2

# Cheap simulation controls
min_steps_per_wvl = 12
sim_run_time = 2e-12
port_size_y = 3 * wt

#### GEOMETRIA

def b_symmetric_body_vertices(L_mmi, W_mmi, W_narrow):
    return np.array(
        [
            #body mmi
            # parte abajo: mirar dibujo libreta
            (0.0,          -W_mmi / 2), #p1
            (L_mmi / 4,    -W_narrow / 2), #p2
            (L_mmi / 2,    -W_mmi / 2), #p3
            (3*L_mmi / 4,  -W_narrow / 2),
            (L_mmi,        -W_mmi / 2),

            # parte arriba: mirar dibujo libreta
            (L_mmi,         W_mmi / 2), #p6
            (3*L_mmi / 4,   W_narrow / 2),
            (L_mmi / 2,     W_mmi / 2),
            (L_mmi / 4,     W_narrow / 2),
            (0.0,           W_mmi / 2),
        ]
    )


def taper_bottom_left_vertices(W_mmi, lt, wt, Wt, dy):
    yc = -(W_mmi / 6 + dy)

    return np.array([
        (-lt, yc - wt / 2),
        (0.0, yc - Wt / 2),
        (0.0, yc + Wt / 2),
        (-lt, yc + wt / 2),
    ])


def taper_top_left_vertices(W_mmi, lt, wt, Wt, dy):
    yc = W_mmi / 6 + dy

    return np.array([
        (-lt, yc - wt / 2),
        (0.0, yc - Wt / 2),
        (0.0, yc + Wt / 2),
        (-lt, yc + wt / 2),
    ])


def taper_bottom_right_vertices(L_mmi, W_mmi, lt, wt, Wt, dy):
    yc = -(W_mmi / 6 + dy)

    return np.array([
        (L_mmi,      yc - Wt / 2),
        (L_mmi + lt, yc - wt / 2),
        (L_mmi + lt, yc + wt / 2),
        (L_mmi,      yc + Wt / 2),
    ])


def taper_top_right_vertices(L_mmi, W_mmi, lt, wt, Wt, dy):
    yc = W_mmi / 6 + dy

    return np.array([
        (L_mmi,      yc - Wt / 2),
        (L_mmi + lt, yc - wt / 2),
        (L_mmi + lt, yc + wt / 2),
        (L_mmi,      yc + Wt / 2),
    ])


def wg_bottom_left_vertices(W_mmi, L_wg, lt, wt, dy):
    yc = -(W_mmi / 6 + dy)

    return np.array([
        (-lt - L_wg, yc - wt / 2),
        (-lt,        yc - wt / 2),
        (-lt,        yc + wt / 2),
        (-lt - L_wg, yc + wt / 2),
    ])


def wg_top_left_vertices(W_mmi, L_wg, lt, wt, dy):
    yc = W_mmi / 6 + dy

    return np.array([
        (-lt - L_wg, yc - wt / 2),
        (-lt,        yc - wt / 2),
        (-lt,        yc + wt / 2),
        (-lt - L_wg, yc + wt / 2),
    ])


def wg_bottom_right_vertices(L_mmi, W_mmi, L_wg, lt, wt, dy):
    yc = -(W_mmi / 6 + dy)

    return np.array([
        (L_mmi + lt,        yc - wt / 2),
        (L_mmi + lt + L_wg, yc - wt / 2),
        (L_mmi + lt + L_wg, yc + wt / 2),
        (L_mmi + lt,        yc + wt / 2),
    ])


def wg_top_right_vertices(L_mmi, W_mmi, L_wg, lt, wt, dy):
    yc = W_mmi / 6 + dy

    return np.array([
        (L_mmi + lt,        yc - wt / 2),
        (L_mmi + lt + L_wg, yc - wt / 2),
        (L_mmi + lt + L_wg, yc + wt / 2),
        (L_mmi + lt,        yc + wt / 2),
    ])

# define a function that takes the geometric parameters as input arguments
# and returns a Simulation object
def make_sim(
    L_mmi=L_mmi,
    W_mmi=W_mmi,
    W_narrow=W_narrow,
    dy=dy,
):

    # =========================
    # POSICIONES Y DE LOS PUERTOS
    # =========================

    y_top = W_mmi / 6 + dy
    y_bottom = -(W_mmi / 6 + dy)

    # =========================
    # TAMAÑO DOMINIO SIMULACIÓN
    # =========================

    Lx = L_mmi + 2 * lt + 2 * buffer_x + 2 * L_wg
    Ly = W_mmi + 2 * buffer_y

    sim_size = (Lx, Ly, Lz)
    sim_center = (L_mmi / 2, 0, sim_center_z)

    # =========================
    # SUBSTRATO SiO2
    # =========================

    sio2_substrate = td.Structure(
        geometry=td.Box(
            center=(
                L_mmi / 2,
                0,
                -h_core / 2 - h_sio2 / 2,
            ),
            size=(
                td.inf,
                td.inf,
                h_sio2,
            ),
        ),
        medium=sio2,
    )

    # =========================
    # GUÍAS
    # =========================

    wg_bottom_left = td.Structure(
        geometry=td.PolySlab(
            vertices=wg_bottom_left_vertices(
                W_mmi, L_wg, lt, wt, dy
            ),
            axis=2,
            slab_bounds=(-h_core / 2, h_core / 2),
        ),
        medium=sin,
    )

    wg_top_left = td.Structure(
        geometry=td.PolySlab(
            vertices=wg_top_left_vertices(
                W_mmi, L_wg, lt, wt, dy
            ),
            axis=2,
            slab_bounds=(-h_core / 2, h_core / 2),
        ),
        medium=sin,
    )

    wg_bottom_right = td.Structure(
        geometry=td.PolySlab(
            vertices=wg_bottom_right_vertices(
                L_mmi, W_mmi, L_wg, lt, wt, dy
            ),
            axis=2,
            slab_bounds=(-h_core / 2, h_core / 2),
        ),
        medium=sin,
    )

    wg_top_right = td.Structure(
        geometry=td.PolySlab(
            vertices=wg_top_right_vertices(
                L_mmi, W_mmi, L_wg, lt, wt, dy
            ),
            axis=2,
            slab_bounds=(-h_core / 2, h_core / 2),
        ),
        medium=sin,
    )

    # =========================
    # TAPERS
    # =========================

    taper_bottom_left = td.Structure(
        geometry=td.PolySlab(
            vertices=taper_bottom_left_vertices(
                W_mmi, lt, wt, Wt, dy
            ),
            axis=2,
            slab_bounds=(-h_core / 2, h_core / 2),
        ),
        medium=sin,
    )

    taper_top_left = td.Structure(
        geometry=td.PolySlab(
            vertices=taper_top_left_vertices(
                W_mmi, lt, wt, Wt, dy
            ),
            axis=2,
            slab_bounds=(-h_core / 2, h_core / 2),
        ),
        medium=sin,
    )

    taper_bottom_right = td.Structure(
        geometry=td.PolySlab(
            vertices=taper_bottom_right_vertices(
                L_mmi, W_mmi, lt, wt, Wt, dy
            ),
            axis=2,
            slab_bounds=(-h_core / 2, h_core / 2),
        ),
        medium=sin,
    )

    taper_top_right = td.Structure(
        geometry=td.PolySlab(
            vertices=taper_top_right_vertices(
                L_mmi, W_mmi, lt, wt, Wt, dy
            ),
            axis=2,
            slab_bounds=(-h_core / 2, h_core / 2),
        ),
        medium=sin,
    )

    # =========================
    # GEOMETRÍA MMI
    # =========================

    mmi_body = td.Structure(
        geometry=td.PolySlab(
            vertices=b_symmetric_body_vertices(
                L_mmi, W_mmi, W_narrow
            ),
            axis=2,
            slab_bounds=(-h_core / 2, h_core / 2),
        ),
        medium=sin,
    )

    # =========================
    # MODE SPEC
    # =========================

    mode_spec = td.ModeSpec(
        num_modes=2,
        target_neff=n_sin,
    )

    # =========================
    # POSICIONES SOURCE / MONITORS
    # =========================

    source_x = -lt - L_wg / 2
    monitor_x = L_mmi + lt + L_wg / 2

    source_y = y_top
    monitor_top_y = y_top
    monitor_bottom_y = y_bottom

    # =========================
    # MODE SOURCE
    # =========================

    mode_source = td.ModeSource(
        center=(source_x, source_y, 0),
        size=(0, 5 * wt, Lz),
        source_time=td.GaussianPulse(
            freq0=freq0,
            fwidth=fwidth,
        ),
        direction="+",
        mode_spec=mode_spec,
        mode_index=0,
        num_freqs=5,
    )

    # =========================
    # FLUX MONITORS
    # =========================

    flux_monitor_top = td.FluxMonitor(
        center=(monitor_x, monitor_top_y, 0),
        size=(0, 5 * wt, Lz),
        freqs=freqs,
        name="flux_top",
    )

    flux_monitor_bottom = td.FluxMonitor(
        center=(monitor_x, monitor_bottom_y, 0),
        size=(0, 5 * wt, Lz),
        freqs=freqs,
        name="flux_bottom",
    )

    # =========================
    # MODE MONITORS
    # =========================

    mode_monitor_top = td.ModeMonitor(
        center=(monitor_x, monitor_top_y, 0),
        size=(0, 5 * wt, Lz),
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode_top",
    )

    mode_monitor_bottom = td.ModeMonitor(
        center=(monitor_x, monitor_bottom_y, 0),
        size=(0, 5 * wt, Lz),
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode_bottom",
    )

    # =========================
    # FIELD MONITOR
    # =========================

    field_monitor = td.FieldMonitor(
        center=(L_mmi / 2, 0, 0),
        size=(L_mmi + 2 * lt, W_mmi + 1.0, 0),
        freqs=[freq0],
        name="field",
    )

    # =========================
    # SIMULATION
    # =========================

    sim = td.Simulation(
        center=sim_center,
        size=sim_size,

        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=min_steps_per_wvl,
            wavelength=lambda_min,
        ),

        structures=[
            sio2_substrate,
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

        sources=[mode_source],

        monitors=[
            field_monitor,
            flux_monitor_top,
            flux_monitor_bottom,
            mode_monitor_top,
            mode_monitor_bottom,
        ],

        run_time=5e-12,
        boundary_spec=td.BoundarySpec.all_sides(
            boundary=td.PML()
        ),
        medium=sio2,
    )

    return sim

#########################################################################################################################################################
L_0 = deltaW * (1/3)* 3*L_pi_mmi
L_mmi= L_0 + 8 -1
#############################################################################################################################################################3
fig, ax = plt.subplots(figsize=(24, 6))
sim = make_sim( L_mmi, W_mmi, W_narrow)
sim.plot(z=0, ax=ax)
plt.show()

sim_data = None


if ESTIMATE_TIDY3D_COST:
    test_tidy3d_api()

    estimated_cost = estimate_tidy3d_cost(
        sim,
        task_name="mmi_2x2_no_tapers_cost",
        verbose=True,
    )

    print(
        f"\nEstimated Tidy3D cost: "
        f"{estimated_cost:.6g} FlexCredits"
    )

print("\nSimulation validated successfully.")

if LOAD_TIDY3D_DATA:

    print(
        f"\nLoading Tidy3D data from:"
        f"\n{TIDY3D_DATA_PATH}"
    )

    sim_data = load_tidy3d_simulation_data(
        TIDY3D_DATA_PATH
    )

if RUN_TIDY3D_CLOUD:

    import tidy3d.web as web

    test_tidy3d_api()

    print("\n" + "=" * 60)
    print("RUNNING TIDY3D CLOUD SIMULATION")
    print("=" * 60)

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
        task_name=TIDY3D_TASK_NAME,
        path=TIDY3D_DATA_PATH,
        verbose=True,
    )

# ============================================================
# POSTPROCESADO ESPECTRAL
# ============================================================

if sim_data is not None:

    # --------------------------------------------------------
    # FLUX
    # --------------------------------------------------------

    flux_top_data = sim_data["flux_top"].flux
    flux_bottom_data = sim_data["flux_bottom"].flux

    frequencies = np.asarray(
        flux_top_data.f.values,
        dtype=float,
    )

    wavelengths_um = (
        td.C_0 / frequencies
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


    # --------------------------------------------------------
    # ORDENAR POR LONGITUD DE ONDA
    # --------------------------------------------------------

    order = np.argsort(
        wavelengths_um
    )

    wavelengths_um = (
        wavelengths_um[order]
    )

    T_top = T_top[order]
    T_bottom = T_bottom[order]


    # --------------------------------------------------------
    # TRANSMISION Y SPLITTING
    # --------------------------------------------------------

    T_total = (
        T_top
        + T_bottom
    )

    K_top = np.divide(
        T_top,
        T_total,
        out=np.zeros_like(T_top),
        where=T_total > 0,
    )

    K_bottom = np.divide(
        T_bottom,
        T_total,
        out=np.zeros_like(T_bottom),
        where=T_total > 0,
    )

    excess_loss_db = (
        -10.0
        * np.log10(
            np.maximum(
                T_total,
                1e-20,
            )
        )
    )

    

   
    # ============================================================
    # BANDWIDTH
    # ============================================================

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

    valid = (
        (K_bottom >= lower_limit)
        & (K_bottom <= upper_limit)
    )


    # ------------------------------------------------------------
    # BUSCAR LA REGIÓN CONTIGUA QUE CONTIENE lambda0
    # ------------------------------------------------------------

    index_lambda0 = np.argmin(
        np.abs(
            wavelengths_um
            - lambda0
        )
    )

    if valid[index_lambda0]:

        i_left = index_lambda0
        i_right = index_lambda0

        while (
            i_left > 0
            and valid[i_left - 1]
        ):
            i_left -= 1

        while (
            i_right < len(valid) - 1
            and valid[i_right + 1]
        ):
            i_right += 1

        lambda_bw_min = (
            wavelengths_um[i_left]
        )

        lambda_bw_max = (
            wavelengths_um[i_right]
        )

        bandwidth_um = (
            lambda_bw_max
            - lambda_bw_min
        )

        bandwidth_nm = (
            1000.0
            * bandwidth_um
        )

    else:

        lambda_bw_min = np.nan
        lambda_bw_max = np.nan
        bandwidth_um = 0.0
        bandwidth_nm = 0.0


    # ============================================================
    # GRÁFICA BANDWIDTH
    # ============================================================

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
        COUPLING_TARGET,
        linestyle="--",
        linewidth=1,
        label="Target 95%",
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

    plt.show()


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
        f"λ_min: "
        f"{lambda_bw_min:.4f} µm"
    )

    print(
        f"λ_max: "
        f"{lambda_bw_max:.4f} µm"
    )

    print(
        f"Bandwidth: "
        f"{bandwidth_nm:.2f} nm"
    )


    # --------------------------------------------------------
    # FASE TE0
    # --------------------------------------------------------

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

    a_top = a_top[order]
    a_bottom = a_bottom[order]

    phase_difference_rad = np.angle(
        a_top
        * np.conj(a_bottom)
    )

    phase_difference_deg = np.degrees(
        np.unwrap(
            phase_difference_rad
        )
    )


    # ========================================================
    # 1. COUPLING RATIO
    # ========================================================

    plt.figure(
        figsize=(7, 4)
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

    # Para MMI 95:5
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
    # 2. POTENCIA TOTAL TRANSMITIDA
    # ========================================================

    plt.figure(
        figsize=(7, 4)
    )

    plt.plot(
        wavelengths_um,
        T_total,
        linewidth=2,
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
    # 3. EXCESS LOSS
    # ========================================================

    plt.figure(
        figsize=(7, 4)
    )

    plt.plot(
        wavelengths_um,
        excess_loss_db,
        linewidth=2,
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
    # 4. FASE RELATIVA
    # ========================================================

    plt.figure(
        figsize=(7, 4)
    )

    plt.plot(
        wavelengths_um,
        phase_difference_deg,
        linewidth=2,
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

    plt.tight_layout()


    # ========================================================
    # RESUMEN
    # ========================================================

    print(
        f"Average excess loss: "
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