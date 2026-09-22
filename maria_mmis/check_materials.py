import tidy3d as td
from tidy3d import material_library


WAVELENGTH_UM = 1.55

freq0 = td.C_0 / WAVELENGTH_UM


print("=" * 60)
print(f"MATERIALS AT {WAVELENGTH_UM:.3f} um")
print("=" * 60)


# ============================================================
# Si3N4
# ============================================================

sin_variants = [
    "Luke2015PMLStable",
    "Luke2015Sellmeier",
]

print("\nSi3N4:")

for variant in sin_variants:

    medium = material_library["Si3N4"][variant]

    n, k = medium.nk_model(freq0)

    print(
        f"{variant:25s} "
        f"n = {n:.6f}, "
        f"k = {k:.6e}"
    )


# ============================================================
# SiO2
# ============================================================

# ============================================================
# SiO2
# ============================================================

print("\nSiO2:")

sio2_library = material_library["SiO2"]

print(
    "Available variants:",
    list(sio2_library.variants.keys()),
)

for variant in sio2_library.variants.keys():

    medium = sio2_library[variant]

    try:
        n, k = medium.nk_model(freq0)

        print(
            f"{variant:30s} "
            f"n = {n:.6f}, "
            f"k = {k:.6e}"
        )

    except Exception as exc:

        print(
            f"{variant:30s} "
            f"not usable at {WAVELENGTH_UM:.3f} um "
            f"({exc})"
        )


# ============================================================
# VALORES ACTUALES DEL REPOSITORIO
# ============================================================

print("\nCurrent UPVfab values:")

print(
    f"SiN:  n = {1.996:.6f}"
)

print(
    f"SiO2: n = {1.444:.6f}"
)

print("=" * 60)
