from __future__ import annotations

import os
from pathlib import Path


TIDY3D_API_KEY_ENV_VAR = "SIMCLOUD_APIKEY"


def has_tidy3d_api_key(env_var: str = TIDY3D_API_KEY_ENV_VAR) -> bool:
    """Return True when a Tidy3D API key is available in the environment."""

    return bool(os.environ.get(env_var))


def require_tidy3d_api_key(env_var: str = TIDY3D_API_KEY_ENV_VAR) -> str:
    """Return the Tidy3D API key from the environment or raise a clear error."""

    api_key = os.environ.get(env_var)

    if not api_key:
        raise RuntimeError(
            "Tidy3D API key not found. Configure it for this shell with:\n\n"
            f'    export {env_var}="your_api_key"\n\n'
            "Then rerun the script. The API key should not be stored in the "
            "repository."
        )

    return api_key


def test_tidy3d_api(env_var: str = TIDY3D_API_KEY_ENV_VAR) -> None:
    """Test that Tidy3D cloud authentication works.

    Tidy3D reads the API key from the official ``SIMCLOUD_APIKEY`` environment
    variable. This helper only checks that the key exists and then delegates to
    ``tidy3d.web.test()``.
    """

    require_tidy3d_api_key(env_var)

    try:
        import tidy3d.web as web
    except ImportError as exc:
        raise ImportError(
            "Tidy3D is not installed. Install it with `uv sync --extra tidy3d`."
        ) from exc

    web.test()


def estimate_tidy3d_cost(
    simulation,
    *,
    task_name: str = "upvfab_tidy3d_cost",
    folder_name: str = "default",
    verbose: bool = True,
    env_var: str = TIDY3D_API_KEY_ENV_VAR,
) -> float:
    """Estimate the FlexCredit cost of a Tidy3D simulation without running it.

    This contacts Tidy3D cloud and creates an estimation task, but it does not
    execute the simulation.
    """

    require_tidy3d_api_key(env_var)

    try:
        import tidy3d.web as web
    except ImportError as exc:
        raise ImportError(
            "Tidy3D is not installed. Install it with `uv sync --extra tidy3d`."
        ) from exc

    job = web.Job(
        simulation=simulation,
        task_name=task_name,
        folder_name=folder_name,
        verbose=verbose,
    )
    return float(job.estimate_cost(verbose=verbose))


def load_tidy3d_simulation_data(
    path: str | Path,
    *,
    lazy: bool = False,
    monitor_names: str | list[str] | tuple[str, ...] | None = None,
):
    """Load previously downloaded Tidy3D ``SimulationData`` from disk."""

    try:
        import tidy3d as td
    except ImportError as exc:
        raise ImportError(
            "Tidy3D is not installed. Install it with `uv sync --extra tidy3d`."
        ) from exc

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Tidy3D simulation data file not found: {data_path}")

    return td.SimulationData.from_file(
        data_path,
        lazy=lazy,
        monitor_names=monitor_names,
    )
