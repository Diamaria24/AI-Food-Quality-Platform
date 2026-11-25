from io import BytesIO
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(
    prefix="/predict/hsi/tomato",
    tags=["Tomato Hyperspectral Research"]
)

MAX_FILE_BYTES = 2 * 1024 * 1024
EXPECTED_BANDS = 550


@lru_cache(maxsize=1)
def load_reference():
    path = (
        "datasets/processed/hyperspectral/tomato_spectral/"
        "Healthy_1_spectral_signature.csv"
    )
    return pd.read_csv(path)


@router.post("/spectral")
async def analyze_tomato_spectral(
    file: UploadFile = File(...)
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Upload a CSV spectral-signature file."
        )

    raw = await file.read(MAX_FILE_BYTES + 1)

    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Maximum Tomato spectral CSV size is 2 MB."
        )

    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Unable to read CSV."
        )

    if "wavelength_nm" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Missing wavelength_nm column."
        )

    if "mean" in df.columns:
        signal_column = "mean"
    elif "mean_raw_response" in df.columns:
        signal_column = "mean_raw_response"
    else:
        raise HTTPException(
            status_code=400,
            detail="Expected mean or mean_raw_response column."
        )

    if len(df) != EXPECTED_BANDS:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {EXPECTED_BANDS} spectral bands."
        )

    try:
        wavelengths = pd.to_numeric(
            df["wavelength_nm"], errors="raise"
        ).to_numpy(dtype=np.float64)

        signal = pd.to_numeric(
            df[signal_column], errors="raise"
        ).to_numpy(dtype=np.float64)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Wavelength and signal columns must be numeric."
        )

    if not (
        np.isfinite(wavelengths).all()
        and np.isfinite(signal).all()
    ):
        raise HTTPException(
            status_code=400,
            detail="CSV contains NaN or infinite spectral values."
        )

    if np.any(np.diff(wavelengths) <= 0):
        raise HTTPException(
            status_code=400,
            detail="Wavelengths must be strictly increasing."
        )

    reference = load_reference()
    expected = reference["wavelength_nm"].to_numpy(dtype=float)

    if not np.allclose(wavelengths, expected, atol=0.05, rtol=0):
        raise HTTPException(
            status_code=400,
            detail="Wavelength grid must match the 550-band Tomato reference."
        )

    # Normalize only for exploratory curve-shape comparison.
    # This is not a calibrated reflectance transformation.
    minimum = float(signal.min())
    maximum = float(signal.max())

    if maximum > minimum:
        normalized = (signal - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(signal)

    reference_signal = reference["mean"].to_numpy(dtype=float)

    ref_min = float(reference_signal.min())
    ref_max = float(reference_signal.max())

    if ref_max > ref_min:
        normalized_reference = (
            reference_signal - ref_min
        ) / (ref_max - ref_min)
    else:
        normalized_reference = np.zeros_like(reference_signal)

    correlation = (
        float(np.corrcoef(normalized, normalized_reference)[0, 1])
        if np.std(normalized) > 0 and np.std(normalized_reference) > 0
        else None
    )

    return {
        "food": "Tomato",
        "analysis_type": "Exploratory spectral signature analysis",
        "filename": file.filename,
        "bands": int(len(signal)),
        "wavelength_min_nm": float(wavelengths.min()),
        "wavelength_max_nm": float(wavelengths.max()),
        "signal_min": minimum,
        "signal_max": maximum,
        "signal_mean": float(signal.mean()),
        "signal_std": float(signal.std()),
        "peak_wavelength_nm": float(wavelengths[np.argmax(signal)]),
        "reference_correlation": correlation,
        "curve": [
            {
                "wavelength_nm": round(float(w), 3),
                "normalized_signal": round(float(s), 6),
                "normalized_reference": round(float(r), 6)
            }
            for w, s, r in zip(
                wavelengths,
                normalized,
                normalized_reference
            )
        ],
        "warning": (
            "Research-only spectral analysis. Min-max normalization "
            "is used for curve visualization, not physical calibration. "
            "Correlation with one Healthy reference is not a disease "
            "classification, freshness prediction, or food-safety result."
        )
    }
