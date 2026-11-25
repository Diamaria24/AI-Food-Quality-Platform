import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(
    prefix="/predict/hsi/tomato",
    tags=["Tomato MATLAB Hyperspectral Research"]
)

EXPECTED_SHAPE = (140, 280, 550)
MAX_UPLOAD_BYTES = 160 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


@router.post("/mat")
async def analyze_tomato_mat(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".mat"):
        raise HTTPException(
            status_code=400,
            detail="Upload a MATLAB .mat file."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mat",
            delete=False
        ) as temp:
            temp_path = temp.name
            total = 0

            while True:
                chunk = await file.read(CHUNK_SIZE)

                if not chunk:
                    break

                total += len(chunk)

                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="MAT file exceeds the 160 MiB limit."
                    )

                temp.write(chunk)

        with open(temp_path, "rb") as handle:
            signature = handle.read(128)

        if b"MATLAB 5.0 MAT-file" not in signature[:40]:
            raise HTTPException(
                status_code=400,
                detail="Expected a MATLAB 5.0 MAT-file."
            )

        try:
            data = loadmat(temp_path)
        except (ValueError, OSError, TypeError) as error:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to read MATLAB file: {error}"
            )

        candidates = [
            (name, value)
            for name, value in data.items()
            if not name.startswith("__")
            and isinstance(value, np.ndarray)
            and value.shape == EXPECTED_SHAPE
            and np.issubdtype(value.dtype, np.number)
        ]

        if len(candidates) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Expected exactly one numeric hyperspectral "
                    "cube with shape 140 x 280 x 550."
                )
            )

        variable_name, cube = candidates[0]

        if not np.isfinite(cube).all():
            raise HTTPException(
                status_code=400,
                detail="Cube contains NaN or infinite values."
            )

        raw_min = float(cube.min())
        raw_max = float(cube.max())
        raw_mean = float(cube.mean())

        if raw_max <= raw_min:
            raise HTTPException(
                status_code=400,
                detail="Cube has no usable signal range."
            )

        # Match the research spectral-signature normalization.
        # Compute the normalized mean spectrum without creating
        # another full-size normalized cube in memory.
        mean_raw = cube.mean(axis=(0, 1))
        mean_normalized = (
            (mean_raw - raw_min) / (raw_max - raw_min)
        )

        return {
            "food": "Tomato",
            "analysis_type": "Exploratory MATLAB HSI cube analysis",
            "filename": Path(file.filename).name,
            "variable_name": variable_name,
            "shape": list(EXPECTED_SHAPE),
            "total_pixels": EXPECTED_SHAPE[0] * EXPECTED_SHAPE[1],
            "spectral_bands": EXPECTED_SHAPE[2],
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
            "normalization": "Per-cube min-max normalization",
            "mean_spectrum": [
                {
                    "band": index + 1,
                    "mean": round(float(value), 6)
                }
                for index, value in enumerate(mean_normalized)
            ],
            "warning": (
                "Research-only spectral analysis. The original "
                "MATLAB dataset does not provide calibrated "
                "wavelength values. Spectral positions are "
                "reported as band indices. This analysis "
                "does not diagnose fungal disease or establish "
                "food safety."
            )
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
