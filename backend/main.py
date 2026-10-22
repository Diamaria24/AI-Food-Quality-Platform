
from pathlib import Path
from uuid import uuid4
import json
import subprocess
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from backend.rgb_mold_inference import (
    predict_rgb_mold,
    MODEL_PATH as RGB_MOLD_MODEL_PATH,
)

from backend.rgb_freshness_inference import (
    predict_rgb_freshness,
    MODEL_PATH as RGB_FRESHNESS_MODEL_PATH
)
from backend.hsi_nitrogen_router import router as hsi_nitrogen_router
from backend.hsi_nitrogen_upload_router import router as hsi_nitrogen_upload_router
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INFERENCE_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "hyperspectral"
    / "pistachio_xgboost_inference.py"
)

UPLOAD_ROOT = PROJECT_ROOT / "outputs" / "api_uploads"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "pistachio_inference"

MAX_HDR_BYTES = 2 * 1024 * 1024
MAX_BIL_BYTES = 150 * 1024 * 1024

app = FastAPI(
    title="AI Food Quality Platform",
    description="RGB, hyperspectral and multispectral analysis API",
    version="0.1.0",
)
app.include_router(hsi_nitrogen_router)
app.include_router(hsi_nitrogen_upload_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "project": "AI Food Quality Platform",
        "status": "Backend running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pistachio_inference_script_exists": (
            INFERENCE_SCRIPT.is_file()
        ),
    }


async def save_limited_upload(upload, destination, max_bytes):
    total = 0

    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(1024 * 1024)

            if not chunk:
                break

            total += len(chunk)

            if total > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded file exceeds size limit.",
                )

            output.write(chunk)

    if total == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )


@app.post("/predict/hsi/pistachio")
async def predict_pistachio(
    hdr: UploadFile = File(...),
    bil: UploadFile = File(...),
):
    if not hdr.filename or not bil.filename:
        raise HTTPException(
            status_code=400,
            detail="Both filenames are required.",
        )

    if Path(hdr.filename).suffix.lower() != ".hdr":
        raise HTTPException(
            status_code=400,
            detail="First file must be .hdr",
        )

    if Path(bil.filename).suffix.lower() != ".bil":
        raise HTTPException(
            status_code=400,
            detail="Second file must be .bil",
        )

    if (
        Path(hdr.filename).stem.lower()
        != Path(bil.filename).stem.lower()
    ):
        raise HTTPException(
            status_code=400,
            detail="HDR and BIL filenames must have matching stems.",
        )

    if not INFERENCE_SCRIPT.is_file():
        raise HTTPException(
            status_code=503,
            detail="Pistachio inference script is missing.",
        )

    request_id = uuid4().hex
    upload_dir = UPLOAD_ROOT / request_id
    upload_dir.mkdir(parents=True, exist_ok=False)

    # Unique filenames prevent output collisions.
    hdr_path = upload_dir / f"{request_id}.hdr"
    bil_path = upload_dir / f"{request_id}.bil"

    try:
        await save_limited_upload(
            hdr, hdr_path, MAX_HDR_BYTES
        )
        await save_limited_upload(
            bil, bil_path, MAX_BIL_BYTES
        )

        command = [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--hdr",
            str(hdr_path),
            "--bil",
            str(bil_path),
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Inference exceeded the time limit.",
            )

        if completed.returncode != 0:
            # Detailed errors stay in server logs.
            print("Inference failed:", completed.stderr)
            raise HTTPException(
                status_code=422,
                detail="Inference failed. Check server logs.",
            )

        summary_path = (
            RESULT_ROOT / request_id / "summary.json"
        )

        if not summary_path.is_file():
            raise HTTPException(
                status_code=500,
                detail="Inference completed but summary.json was not found.",
            )

        with summary_path.open(
            "r", encoding="utf-8"
        ) as file:
            summary = json.load(file)

        return JSONResponse({
            "request_id": request_id,
            "module": "hyperspectral",
            "food": "pistachio",
            "status": "completed",
            "result": {
                "prediction": summary.get(
                    "exploratory_majority_vote"
                ),
                "fraction_at_or_above": summary.get(
                    "fraction_predicted_at_or_above"
                ),
                "mean_model_probability_at_or_above": summary.get(
                    "mean_model_probability_at_or_above"
                ),
                "foreground_pixels": summary.get(
                    "foreground_pixels"
                ),
                "sampled_pixels": summary.get(
                    "sampled_pixels"
                ),
                "threshold_ug_per_kg": 8.0,
            },
            "warning": (
                "Experimental classification only. "
                "Not a food-safety certification or "
                "calibrated confidence estimate."
            ),
        })

    finally:
        await hdr.close()
        await bil.close()

        # Delete temporary uploaded copies.
        for path in (hdr_path, bil_path):
            if path.exists():
                path.unlink()

        if upload_dir.exists():
            upload_dir.rmdir()

@app.post("/predict/rgb/mold")
async def predict_mold_rgb(
    image: UploadFile = File(...)
):
    """
    Experimental RGB mold detection.

    Supported foods:
    Tomato, Orange, Onion, Carrot.

    Intended input:
    Microscope-assisted smartphone RGB photographs.
    """

    MAX_IMAGE_BYTES = 10 * 1024 * 1024

    if not image.filename:
        raise HTTPException(
            status_code=400,
            detail="Image filename is required."
        )

    allowed_extensions = {".jpg", ".jpeg", ".png"}

    extension = Path(image.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    if not RGB_MOLD_MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="RGB mold detection model is unavailable."
        )

    request_id = uuid4().hex

    upload_dir = UPLOAD_ROOT / request_id
    upload_dir.mkdir(parents=True, exist_ok=False)

    image_path = upload_dir / f"{request_id}{extension}"

    try:
        await save_limited_upload(
            image,
            image_path,
            MAX_IMAGE_BYTES
        )

        # Verify that the uploaded file is a readable image.
        try:
            with Image.open(image_path) as img:
                if img.format not in {"JPEG", "PNG"}:
                    raise HTTPException(
                        status_code=400,
                        detail="Image must be a valid JPEG or PNG."
                    )
                img.verify()

        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image."
            )

        # Run the trained MobileNetV2 model.
        try:
            prediction = predict_rgb_mold(image_path)

        except Exception as error:
            print("RGB mold inference failed:", repr(error))

            raise HTTPException(
                status_code=500,
                detail="RGB mold inference failed. Check server logs."
            )

        return JSONResponse({
            "request_id": request_id,
            "module": "rgb_mold",
            "status": "completed",
            "result": prediction,
            "warning": prediction["warning"]
        })

    finally:
        await image.close()

        # Remove temporary uploaded image.
        if image_path.exists():
            image_path.unlink()

        if upload_dir.exists():
            upload_dir.rmdir()

# =========================================================
# RGB Freshness Classification
# =========================================================

@app.post("/predict/rgb/freshness")
async def predict_rgb_freshness_endpoint(
    image: UploadFile = File(...)
):
    request_id = str(uuid4())
    image_path = None

    try:
        # Validate extension.
        extension = Path(
            image.filename or ""
        ).suffix.lower()

        if extension not in {".jpg", ".jpeg", ".png"}:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, JPEG and PNG images are supported."
            )

        # Check model availability.
        if not RGB_FRESHNESS_MODEL_PATH.is_file():
            raise HTTPException(
                status_code=503,
                detail="RGB Freshness model is unavailable."
            )

        # Store uploaded image temporarily.
        request_dir = UPLOAD_ROOT / request_id
        request_dir.mkdir(parents=True, exist_ok=True)

        image_path = request_dir / f"freshness{extension}"

        await save_limited_upload(
            image,
            image_path,
            10 * 1024 * 1024
        )

        # Validate actual image contents.
        try:
            with Image.open(image_path) as img:
                if img.format not in {"JPEG", "PNG"}:
                    raise ValueError("Unsupported image format.")
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid JPEG or PNG image."
            )

        # Run MobileNet inference.
        prediction = predict_rgb_freshness(
            image_path
        )

        return {
            "request_id": request_id,
            "module": "rgb_freshness",
            "status": "completed",
            "result": prediction,
            "warning": prediction["warning"]
        }

    except HTTPException:
        raise

    except Exception as exc:
        print(
            f"RGB Freshness inference error: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail="RGB Freshness prediction failed."
        )

    finally:
        await image.close()

        if image_path is not None:
            import shutil
            shutil.rmtree(
                image_path.parent,
                ignore_errors=True
            )
