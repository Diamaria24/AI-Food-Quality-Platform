from io import BytesIO
import re

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from sklearn.decomposition import PCA

router = APIRouter(
    prefix="/predict/hsi/papaya",
    tags=["Papaya Hyperspectral Research"]
)

EXPECTED_SHAPE = (32, 32, 150)
EXPECTED_BYTES = 32 * 32 * 150 * 4
MAX_HDR_BYTES = 64 * 1024
MAX_IMG_BYTES = 2 * 1024 * 1024


def parse_header(text):
    fields = {}

    for key in [
        "samples", "lines", "bands",
        "data type", "interleave",
        "byte order", "header offset"
    ]:
        match = re.search(
            rf"(?im)^\s*{re.escape(key)}\s*=\s*([^\r\n]+)",
            text
        )

        if match:
            fields[key] = match.group(1).strip().lower()

    required = [
        "samples", "lines", "bands",
        "data type", "interleave",
        "byte order"
    ]

    if any(key not in fields for key in required):
        raise HTTPException(
            status_code=400,
            detail="Incomplete ENVI header."
        )

    try:
        shape = (
            int(fields["lines"]),
            int(fields["samples"]),
            int(fields["bands"])
        )

        dtype_code = int(fields["data type"])
        byte_order = int(fields["byte order"])
        offset = int(fields.get("header offset", "0"))

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid numeric ENVI header fields."
        )

    if shape != EXPECTED_SHAPE:
        raise HTTPException(
            status_code=400,
            detail=f"Expected cube shape {EXPECTED_SHAPE}."
        )

    if (
        dtype_code != 4
        or fields["interleave"] != "bip"
        or byte_order != 0
        or offset != 0
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Expected Float32, BIP, little-endian "
                "ENVI data with zero header offset."
            )
        )

    return shape


@router.post("/spectral")
async def analyze_papaya_hsi(
    hdr: UploadFile = File(...),
    img: UploadFile = File(...)
):
    if not hdr.filename or not hdr.filename.lower().endswith(".hdr"):
        raise HTTPException(
            status_code=400,
            detail="First file must be an ENVI .hdr file."
        )

    if not img.filename or not img.filename.lower().endswith(".img"):
        raise HTTPException(
            status_code=400,
            detail="Second file must be an ENVI .img file."
        )

    if (
        hdr.filename.rsplit(".", 1)[0].lower()
        != img.filename.rsplit(".", 1)[0].lower()
    ):
        raise HTTPException(
            status_code=400,
            detail="HDR and IMG filenames must have matching stems."
        )

    header_bytes = await hdr.read(MAX_HDR_BYTES + 1)

    if len(header_bytes) > MAX_HDR_BYTES:
        raise HTTPException(
            status_code=413,
            detail="HDR file is too large."
        )

    try:
        header_text = header_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Unable to decode ENVI header."
        )

    shape = parse_header(header_text)

    raw = await img.read(MAX_IMG_BYTES + 1)

    if len(raw) > MAX_IMG_BYTES:
        raise HTTPException(
            status_code=413,
            detail="IMG file exceeds 2 MB."
        )

    if len(raw) != EXPECTED_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {EXPECTED_BYTES} bytes, "
                f"received {len(raw)}."
            )
        )

    cube = np.frombuffer(raw, dtype="<f4").reshape(shape)

    if not np.isfinite(cube).all():
        raise HTTPException(
            status_code=400,
            detail="Cube contains NaN or infinite values."
        )

    pixels = cube.reshape(-1, shape[2])

    mean_spectrum = pixels.mean(axis=0)
    std_spectrum = pixels.std(axis=0)

    # PCA is fitted independently on the uploaded cube.
    # It is exploratory and not the trained classifier PCA.
    pca = PCA(n_components=3, random_state=42)
    transformed = pca.fit_transform(pixels)

    return {
        "food": "Papaya",
        "analysis_type": "Exploratory HSI cube analysis",
        "filename": img.filename,
        "shape": list(shape),
        "total_pixels": int(len(pixels)),
        "spectral_bands": int(shape[2]),
        "signal_min": float(cube.min()),
        "signal_max": float(cube.max()),
        "signal_mean": float(cube.mean()),
        "pca_explained_variance_percent": [
            round(float(value * 100), 4)
            for value in pca.explained_variance_ratio_
        ],
        "mean_spectrum": [
            {
                "band": i + 1,
                "mean": round(float(mean_spectrum[i]), 6),
                "std": round(float(std_spectrum[i]), 6)
            }
            for i in range(shape[2])
        ],
        "pca_preview": [
            {
                "PC1": round(float(row[0]), 6),
                "PC2": round(float(row[1]), 6),
                "PC3": round(float(row[2]), 6)
            }
            for row in transformed[:100]
        ],
        "warning": (
            "Research-only hyperspectral analysis. PCA is fitted "
            "on the uploaded sample and cannot be interpreted as "
            "a validated ripeness, disease, or food-safety prediction. "
            "Wavelength calibration was not available in the header; "
            "the spectrum is displayed by band index."
        )
    }
