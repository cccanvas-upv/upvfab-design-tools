"""Check Tidy3D cloud API configuration.

Usage:

    export SIMCLOUD_APIKEY="your_api_key"
    uv run python examples/fdtd/test_tidy3d_api.py

The API key is read by Tidy3D from the environment and is not stored in this
repository.
"""

from upvfab_design_tools.tidy3d_plugin import test_tidy3d_api


test_tidy3d_api()
print("Tidy3D API configuration looks OK.")
