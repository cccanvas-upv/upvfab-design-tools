from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


IndexModel = float | complex | Callable[[float], float | complex]


@dataclass(frozen=True)
class Material:
    """
    Material description.

    Parameters
    ----------
    name:
        Material name.
    index_model:
        Refractive index or function n(wavelength_um).
    description:
        Optional material description.
    aliases:
        Alternative names for the same material.
    metadata:
        Extra user-defined information.
    """

    name: str
    index_model: IndexModel
    description: str = ""
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def n(self, wavelength_um: float = 1.55) -> complex:
        """
        Return the refractive index at a given wavelength.

        Parameters
        ----------
        wavelength_um:
            Wavelength in micrometers.

        Returns
        -------
        complex
            Refractive index.
        """

        if wavelength_um <= 0:
            raise ValueError("wavelength_um must be positive.")

        if callable(self.index_model):
            value = self.index_model(wavelength_um)
        else:
            value = self.index_model

        return complex(value)

    def epsilon(self, wavelength_um: float = 1.55) -> complex:
        """
        Return the relative permittivity epsilon_r = n^2.
        """

        return self.n(wavelength_um) ** 2


# Initial material library. 
# Mi idea is to connect this with Sergi Pla work with the ellipsometer
# For the moment I will just use the value @RefractiveIndex.info - 1.55um

SILICON_NITRIDE = Material(
    name="Silicon Nitride",
    index_model=1.96106,
    aliases=("SiN", "Si3N4", "silicon_nitride"),
    description="Silicon nitride.",
)

THERMAL_SILICON_DIOXIDE = Material(
    name="Thermal Silicon Dioxide",
    index_model=1.4581, #1.444
    aliases=("SiO2 thermal", "thermal_sio2", "thermal oxide"),
    description="Thermal silicon dioxide.",
)

LPCVD_SILICON_DIOXIDE = Material(
    name="LPCVD Silicon Dioxide",
    index_model=1.454,
    aliases=("SiO2 LPCVD", "lpcvd_sio2", "LPCVD oxide"),
    description="LPCVD silicon dioxide",
)

SILICON = Material(
    name="Silicon",
    index_model=3.476,
    aliases=("Si", "silicon"),
    description="Crystalline silicon. Default value around 1.55 um.",
)

BCB = Material(
    name="Benzocyclobutene",
    index_model=1.57442,
    aliases=("BCB", "benzocyclobutene"),
    description="Benzocyclobutene polymer.",
)

AIR = Material(
    name="Air",
    index_model=1.0,
    aliases=("air",),
    description="Air.",
)

MATERIALS: dict[str, Material] = {
    "air": AIR,
    "sin": SILICON_NITRIDE,
    "si3n4": SILICON_NITRIDE,
    "silicon_nitride": SILICON_NITRIDE,
    "thermal_sio2": THERMAL_SILICON_DIOXIDE,
    "sio2_thermal": THERMAL_SILICON_DIOXIDE,
    "lpcvd_sio2": LPCVD_SILICON_DIOXIDE,
    "sio2_lpcvd": LPCVD_SILICON_DIOXIDE,
    "si": SILICON,
    "silicon": SILICON,
    "bcb": BCB,
    "benzocyclobutene": BCB,
}

def get_material(name: str) -> Material:
    """
    Get a material from the material registry.

    Parameters
    ----------
    name:
        Material name or alias.

    Returns
    -------
    Material
        Requested material.
    """

    key = name.strip().lower()

    if key not in MATERIALS:
        available = ", ".join(sorted(MATERIALS))
        raise KeyError(
            f"Unknown material '{name}'. Available materials are: {available}"
        )

    return MATERIALS[key]