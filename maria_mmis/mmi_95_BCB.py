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

RUN_TIDY3D_CLOUD = True
ESTIMATE_TIDY3D_COST = False
LOAD_TIDY3D_DATA = False

TIDY3D_TASK_NAME = "mmi95_bcb_wt2.7_Awplus0.08_L0plus7_dyPLUS0"

# Archivo local donde se guardaran los resultados
TIDY3D_DATA_PATH = f"data/{TIDY3D_TASK_NAME}.hdf5"

################ MATERIALES
# sin, sin_rms = SILICON_NITRIDE.fit_tidy(
#     wavelength_min_um=1.5,
#     wavelength_max_um=1.6,
#     num_samples=201,
#     max_num_poles=2,
#     tolerance_rms=1e-4,
# )

# sio2, sio2_rms = THERMAL_SILICON_DIOXIDE.fit_tidy(
#     wavelength_min_um=1.5,
#     wavelength_max_um=1.6,
#     num_samples=201,
#     max_num_poles=2,
#     tolerance_rms=1e-4,
# )

_n_sin = SILICON_NITRIDE.n(1.55)
n_sin = np.real(_n_sin)

sin = td.Medium(
    permittivity=(
        SILICON_NITRIDE.n(1.55).real ** 2
    )
)

sio2 = td.Medium(
    permittivity=(
        THERMAL_SILICON_DIOXIDE.n(1.55).real ** 2
    )
)

bcb = td.Medium(
    permittivity=(
        BCB.n(1.55).real ** 2
    )
)

air = td.Medium(
    permittivity=(
        AIR.n(1.55).real ** 2
    )
)
#######################################################

L_pi_mmi = 153.042918 #recalcular con los modesolvers de designtools si se considera necesario
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

dy = 0 # um.
                        # dy > 0 -> los puertos se separan más
                        # dy < 0 -> los puertos se acercan al centro

#wavelengths
lambda0 = 1.55
ldas = np.array([lambda0])
freq0 = td.C_0 / lambda0
freqs = td.C_0 / ldas
fwidth = 0.1 * freq0
#buffers: márgenes entre dispositivo y bordes de la simulación
buffer_x = 2.0
buffer_y = 3.0
buffer_z = 1.5
#simulation height 
h_bcb= 0.4
h_air_top = 2.0   # um, aire por encima del core
h_sio2 = 2.0      # um, SiO2 por debajo del core
z_core_min = 0.0
z_core_max = h_core
z_bcb_min = 0.0
z_bcb_max = h_bcb
z_top = h_bcb + h_air_top
z_bottom = -h_sio2
Lz = z_top - z_bottom
sim_center_z = (z_top + z_bottom) / 2

# Cheap simulation controls
min_steps_per_wvl = 12
sim_run_time = 2e-12
port_size_y = 4 * wt

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
                -0.5 * h_sio2,
            ),
            size=(
                td.inf,
                td.inf,
                h_sio2,
            ),
        ),
        medium=sio2,
    )

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

    # =========================
    # GUÍAS
    # =========================

    wg_bottom_left = td.Structure(
        geometry=td.PolySlab(
            vertices=wg_bottom_left_vertices(
                W_mmi, L_wg, lt, wt, dy
            ),
            axis=2,
            slab_bounds=(0, h_core ),
        ),
        medium=sin,
    )

    wg_top_left = td.Structure(
        geometry=td.PolySlab(
            vertices=wg_top_left_vertices(
                W_mmi, L_wg, lt, wt, dy
            ),
            axis=2,
            slab_bounds=(0, h_core),
        ),
        medium=sin,
    )

    wg_bottom_right = td.Structure(
        geometry=td.PolySlab(
            vertices=wg_bottom_right_vertices(
                L_mmi, W_mmi, L_wg, lt, wt, dy
            ),
            axis=2,
            slab_bounds=(0, h_core ),
        ),
        medium=sin,
    )

    wg_top_right = td.Structure(
        geometry=td.PolySlab(
            vertices=wg_top_right_vertices(
                L_mmi, W_mmi, L_wg, lt, wt, dy
            ),
            axis=2,
            slab_bounds=(0, h_core ),
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
            slab_bounds=(0, h_core),
        ),
        medium=sin,
    )

    taper_top_left = td.Structure(
        geometry=td.PolySlab(
            vertices=taper_top_left_vertices(
                W_mmi, lt, wt, Wt, dy
            ),
            axis=2,
            slab_bounds=(0, h_core ),
        ),
        medium=sin,
    )

    taper_bottom_right = td.Structure(
        geometry=td.PolySlab(
            vertices=taper_bottom_right_vertices(
                L_mmi, W_mmi, lt, wt, Wt, dy
            ),
            axis=2,
            slab_bounds=(0, h_core ),
        ),
        medium=sin,
    )

    taper_top_right = td.Structure(
        geometry=td.PolySlab(
            vertices=taper_top_right_vertices(
                L_mmi, W_mmi, lt, wt, Wt, dy
            ),
            axis=2,
            slab_bounds=(0, h_core ),
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
            slab_bounds=(0, h_core),
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
        center=(source_x, source_y, sim_center_z),
        size=(0, port_size_y, Lz),
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
        center=(monitor_x, monitor_top_y, sim_center_z),
        size=(0, port_size_y, Lz),
        freqs=freqs,
        name="flux_top",
    )

    flux_monitor_bottom = td.FluxMonitor(
        center=(monitor_x, monitor_bottom_y, sim_center_z),
        size=(0, port_size_y, Lz),
        freqs=freqs,
        name="flux_bottom",
    )

    # =========================
    # MODE MONITORS
    # =========================

    mode_monitor_top = td.ModeMonitor(
        center=(monitor_x, monitor_top_y, sim_center_z),
        size=(0, port_size_y, Lz),
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode_top",
    )

    mode_monitor_bottom = td.ModeMonitor(
        center=(monitor_x, monitor_bottom_y, sim_center_z),
        size=(0, port_size_y, Lz),
        freqs=freqs,
        mode_spec=mode_spec,
        name="mode_bottom",
    )

    # =========================
    # FIELD MONITOR
    # =========================

    field_monitor = td.FieldMonitor(
        center=(L_mmi / 2, 0, 0.5 * h_core),
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
        medium=air,
    )

    return sim

#########################################################################################################################################################
L_0 = deltaW * (1/3)* 3*L_pi_mmi
L_mmi= L_0 + 7
#############################################################################################################################################################3
fig, ax = plt.subplots(figsize=(24, 6))
sim = make_sim( L_mmi, W_mmi, W_narrow)
sim.plot(z= 0.5 * h_core, ax=ax)
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

#POSTPROCESADO
if sim_data is not None:

    T_top = sim_data["flux_top"].flux
    T_bottom = sim_data["flux_bottom"].flux

    K_top = T_top / (T_top + T_bottom)
    K_bottom = T_bottom / (T_top + T_bottom)

    amps_top = sim_data["mode_top"].amps
    amps_bottom = sim_data["mode_bottom"].amps

    a_top = amps_top.sel(
        direction="+",
        mode_index=0
    ).isel(f=0).values

    a_bottom = amps_bottom.sel(
        direction="+",
        mode_index=0
    ).isel(f=0).values

    a_top = np.squeeze(a_top)
    a_bottom = np.squeeze(a_bottom)

    phi_top = np.angle(a_top)
    phi_bottom = np.angle(a_bottom)
    delta_phi = np.angle(a_top / a_bottom)

    fig, ax = plt.subplots(figsize=(24, 6))

    print("L_mmi = ", f"{L_mmi}")
    print("T_top =", float(T_top[0]))
    print("T_bottom =", float(T_bottom[0]))
    print("K_top =", float(K_top[0]))
    print("K_bottom =", float(K_bottom[0]))
    print("Splitting ratio =", f"{100*float(K_top[0]):.2f}:{100*float(K_bottom[0]):.2f}")
    print("Excess loss [dB] =", float(-10 * np.log10(T_top[0] + T_bottom[0])))
    print(f"Δφ top-bottom = {delta_phi/np.pi:.4f} π")

    sim_data.plot_field(
        field_monitor_name="field",
        field_name="E",
        val="abs^2",
        f=freq0,
        ax=ax,
    )

    ax.set_title("|E|² at z = 0")
    plt.show()