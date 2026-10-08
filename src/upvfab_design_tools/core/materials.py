from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


IndexModel = float | complex | Callable[[float], float | complex]


@dataclass(frozen=True)
class CauchyIndexModel:
    """Wavelength-dependent refractive index fitted with a Cauchy model.

    The public wavelength convention is micrometers. The fitted expression is
    evaluated in nanometers as

    ``n = A + B * 1e4 / wavelength_nm**2 + C * 1e9 / wavelength_nm**4``.

    Optional validity limits are expressed in micrometers. Each limit is
    independent, and the endpoints are included in the valid interval.
    """

    A: float
    B: float
    C: float
    wavelength_min_um: float | None = None
    wavelength_max_um: float | None = None

    def __post_init__(self) -> None:
        if self.wavelength_min_um is not None and self.wavelength_min_um <= 0:
            raise ValueError("wavelength_min_um must be positive when provided.")

        if self.wavelength_max_um is not None and self.wavelength_max_um <= 0:
            raise ValueError("wavelength_max_um must be positive when provided.")

        if (
            self.wavelength_min_um is not None
            and self.wavelength_max_um is not None
            and self.wavelength_max_um <= self.wavelength_min_um
        ):
            raise ValueError(
                "wavelength_max_um must be larger than wavelength_min_um."
            )

    def __call__(self, wavelength_um: float) -> float:
        """Return the fitted refractive index at ``wavelength_um``."""

        if wavelength_um <= 0:
            raise ValueError("wavelength_um must be positive.")

        if (
            self.wavelength_min_um is not None
            and wavelength_um < self.wavelength_min_um
        ):
            raise ValueError(
                f"wavelength_um={wavelength_um} is below the model minimum "
                f"of {self.wavelength_min_um} um."
            )

        if (
            self.wavelength_max_um is not None
            and wavelength_um > self.wavelength_max_um
        ):
            raise ValueError(
                f"wavelength_um={wavelength_um} is above the model maximum "
                f"of {self.wavelength_max_um} um."
            )

        wavelength_nm = 1000.0 * wavelength_um
        return (
            self.A
            + self.B * 1e4 / wavelength_nm**2
            + self.C * 1e9 / wavelength_nm**4
        )


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

    def fit_tidy(
        self,
        *,
        wavelength_min_um: float,
        wavelength_max_um: float,
        num_samples: int = 201,
        max_num_poles: int = 2,
        tolerance_rms: float = 1e-4,
    ) -> tuple[Any, float]:
        """Fit the material index to a Tidy3D pole-residue medium.

        The refractive-index model is sampled over the requested wavelength
        interval and written to ``upvfab_design_tools/_temp`` as a semicolon-
        separated CSV file with ``wavelength_um``, ``n``, and ``k`` columns.
        The file is retained and replaced by the next fit of the same material.

        Parameters
        ----------
        wavelength_min_um, wavelength_max_um:
            Inclusive wavelength interval in micrometers.
        num_samples:
            Number of evenly spaced wavelength samples, including both
            interval endpoints.
        max_num_poles:
            Maximum number of poles used by Tidy3D's fast dispersion fitter.
        tolerance_rms:
            Weighted RMS-error target passed to the fitter.

        Returns
        -------
        tuple[Any, float]
            Fitted ``tidy3d.PoleResidue`` and its weighted RMS error.
        """

        if not np.isfinite(wavelength_min_um) or wavelength_min_um <= 0:
            raise ValueError("wavelength_min_um must be finite and positive.")

        if not np.isfinite(wavelength_max_um) or wavelength_max_um <= 0:
            raise ValueError("wavelength_max_um must be finite and positive.")

        if wavelength_max_um <= wavelength_min_um:
            raise ValueError(
                "wavelength_max_um must be larger than wavelength_min_um."
            )

        if not isinstance(num_samples, int) or num_samples < 2:
            raise ValueError("num_samples must be an integer of at least 2.")

        if not isinstance(max_num_poles, int) or max_num_poles <= 0:
            raise ValueError("max_num_poles must be a positive integer.")

        if not np.isfinite(tolerance_rms) or tolerance_rms < 0:
            raise ValueError("tolerance_rms must be finite and non-negative.")

        FastDispersionFitter, AdvancedFastFitterParam = _import_tidy3d_fitter()

        wavelengths_um = np.linspace(
            wavelength_min_um,
            wavelength_max_um,
            num_samples,
        )
        complex_indices = np.array(
            [self.n(float(wavelength_um)) for wavelength_um in wavelengths_um],
            dtype=np.complex128,
        )

        if not np.all(np.isfinite(complex_indices)):
            raise ValueError(
                "The material index model returned non-finite values over the "
                "requested wavelength interval."
            )

        csv_path = _tidy_fit_csv_path(self.name)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_data = np.column_stack(
            (wavelengths_um, complex_indices.real, complex_indices.imag)
        )
        np.savetxt(
            csv_path,
            csv_data,
            delimiter=";",
            fmt="%.6f",
            header="wavelength_um;n;k",
            comments="",
        )

        fitter = FastDispersionFitter.from_file(
            csv_path,
            skiprows=1,
            delimiter=";",
        )
        advanced_param = AdvancedFastFitterParam(weights=(1, 1))
        fitted_medium, rms_error = fitter.fit(
            max_num_poles=max_num_poles,
            tolerance_rms=tolerance_rms,
            advanced_param=advanced_param,
        )
        pole_residue = self.to_tidy(
            eps_inf=fitted_medium.eps_inf,
            poles=fitted_medium.poles,
            frequency_range=fitted_medium.frequency_range,
        )

        return pole_residue, float(rms_error)

    def to_tidy(
        self,
        *,
        eps_inf: float,
        poles: Any,
        frequency_range: tuple[float, float] | None = None,
    ) -> Any:
        """Create a named ``tidy3d.PoleResidue`` from fitted parameters."""

        td = _import_tidy3d()
        return td.PoleResidue(
            name=self.name,
            eps_inf=eps_inf,
            poles=poles,
            frequency_range=frequency_range,
        )


def _tidy_fit_csv_path(material_name: str) -> Path:
    safe_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        material_name.strip().lower(),
    ).strip("_")
    if not safe_name:
        safe_name = "material"

    package_directory = Path(__file__).resolve().parent.parent
    return package_directory / "_temp" / f"{safe_name}.csv"


def _import_tidy3d():
    try:
        import tidy3d as td
    except ImportError as exc:
        raise ImportError(
            "Tidy3D material conversion requires the optional 'tidy3d' "
            "dependency. Install it with `uv sync --extra tidy3d`."
        ) from exc

    return td


def _import_tidy3d_fitter():
    try:
        from tidy3d.plugins.dispersion import (
            AdvancedFastFitterParam,
            FastDispersionFitter,
        )
    except ImportError as exc:
        raise ImportError(
            "Tidy3D material fitting requires the optional 'tidy3d' "
            "dependency. Install it with `uv sync --extra tidy3d`."
        ) from exc

    return FastDispersionFitter, AdvancedFastFitterParam


# Initial material library. 
# Mi idea is to connect this with Sergi Pla work with the ellipsometer
# For the moment I will just use the value @RefractiveIndex.info - 1.55um

SILICON_NITRIDE = Material(
    name="Silicon Nitride",
    index_model=1.996,
    aliases=("SiN", "Si3N4", "silicon_nitride"),
    description="Silicon nitride.",
)

THERMAL_SILICON_DIOXIDE = Material(
    name="Thermal Silicon Dioxide",
    index_model=1.444,
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
    index_model=1.550,
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
