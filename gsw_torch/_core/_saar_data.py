"""
SAAR (Absolute Salinity Anomaly Ratio) lookup table data.

This module contains the embedded SAAR data arrays extracted from GSW-C
gsw_saar_data.h file. The data is version 3.0 (15th May 2011).

Grid dimensions:
- nx = 91 (longitude points: 0 to 360 degrees, 4-degree spacing)
- ny = 45 (latitude points: -86 to 90 degrees, 4-degree spacing)
- nz = 45 (pressure levels: 0 to 6131 dbar)

Total saar_ref elements: 91 * 45 * 45 = 184275
"""

import os

import numpy as np
import torch

# Try to load SAAR data from package directory first, then fall back to /tmp
_package_dir = os.path.dirname(__file__)
_package_data_file = os.path.join(_package_dir, "saar_data.npz")
_tmp_data_file = "/tmp/saar_data.npz"

# Prefer package-bundled data, fall back to /tmp for development/testing
if os.path.exists(_package_data_file):
    _data_file = _package_data_file
elif os.path.exists(_tmp_data_file):
    _data_file = _tmp_data_file
else:
    _data_file = None

if _data_file is not None:
    try:
        _data = np.load(_data_file, allow_pickle=True)

        # Grid dimensions
        NX = int(_data["nx"])
        NY = int(_data["ny"])
        NZ = int(_data["nz"])

        # Reference grids
        P_REF = torch.tensor(_data["p_ref"], dtype=torch.float64)
        LATS_REF = torch.tensor(_data["lats_ref"], dtype=torch.float64)
        LONGS_REF = torch.tensor(_data["longs_ref"], dtype=torch.float64)

        # SAAR lookup table (flattened, shape will be (nz, ny, nx) when reshaped)
        SAAR_REF = torch.tensor(_data["saar_ref"], dtype=torch.float64)
        SAAR_REF = SAAR_REF.reshape(NZ, NY, NX)  # Reshape to 3D grid

        # Depth reference (for determining valid ocean points)
        NDEPTH_REF = torch.tensor(_data["ndepth_ref"], dtype=torch.float64)
        NDEPTH_REF = NDEPTH_REF.reshape(NY, NX)  # Reshape to 2D grid

        # Delta SA reference (for deltaSA_atlas function)
        DELTA_SA_REF = torch.tensor(_data["delta_sa_ref"], dtype=torch.float64)
        DELTA_SA_REF = DELTA_SA_REF.reshape(NZ, NY, NX)  # Reshape to 3D grid

        # Panama barrier coordinates
        NPAN = int(_data["npan"])
        LONGS_PAN = torch.tensor(
            [260.00, 272.59, 276.50, 278.65, 280.73, 292.0], dtype=torch.float64
        )
        LATS_PAN = torch.tensor([19.55, 13.97, 9.60, 8.10, 9.33, 3.4], dtype=torch.float64)

        # Grid index offsets for 4 corners
        DELI = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
        DELJ = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

        # Constants
        GSW_ERROR_LIMIT = 1e10
        GSW_INVALID_VALUE = 9e15

    except Exception as err:
        raise RuntimeError(f"Failed to load SAAR data file from {_data_file}: {err}") from err
else:
    raise RuntimeError(
        f"SAAR data file not found. Expected at:\n"
        f"  - {_package_data_file} (package-bundled)\n"
        f"  - {_tmp_data_file} (development/testing)\n"
        f"Please run: python scripts/extract_saar_data.py"
    )

__all__ = [
    "NX",
    "NY",
    "NZ",
    "NPAN",
    "P_REF",
    "LATS_REF",
    "LONGS_REF",
    "SAAR_REF",
    "NDEPTH_REF",
    "DELTA_SA_REF",
    "LONGS_PAN",
    "LATS_PAN",
    "DELI",
    "DELJ",
    "GSW_ERROR_LIMIT",
    "GSW_INVALID_VALUE",
]
