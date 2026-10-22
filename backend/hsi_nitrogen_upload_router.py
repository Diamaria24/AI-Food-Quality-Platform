
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore
import re

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.hsi_nitrogen_upload_inference import (
    predict_uploaded_hsi_nitrogen,
)

from backend.hsi_upload_validation import validate_envi_upload


router = APIRouter(
    prefix="/predict/hsi",
    tags=["Experimental HSI Upload Classification"],
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

UPLOAD_ROOT = (
    PROJECT_ROOT / "outputs" / "api_uploads"
)

MAX_HDR_BYTES = 2 * 1024 * 1024
MAX_BINARY_BYTES = 100 * 1024 * 1024

# Limit simultaneous memory- and disk-intensive analyses.
ANALYSIS_SLOTS = BoundedSemaphore(value=1)


async def save_upload(upload, destination, max_bytes):
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


def prepare_envi_header(header_path, binary_path):
    """
    Verify basic ENVI header fields and force the
    data-file reference to our isolated upload.

    This prevents an uploaded header from referring
    to an arbitrary local file.
    """

    try:
        text = header_path.read_text(encoding="utf-8-sig")
    except UnicodeError:
        raise ValueError("HDR must be a readable text file.")

    if not text.lstrip().upper().startswith("ENVI"):
        raise ValueError("Invalid ENVI header.")

    # Reject any existing data-file reference.
    # We supply our own trusted binary filename.
    text = re.sub(
        r"(?im)^\s*data\s+file\s*=.*(?:\r?\n|$)",
        "",
        text,
    )

    # Reject other ENVI metadata constructs that could
    # refer to external files.
    if re.search(
        r"(?im)^\s*(?:file\s+name|data\s+filename)\s*=",
        text,
    ):
        raise ValueError(
            "Unsupported external file reference in HDR."
        )

    text += f"\ndata file = {binary_path.name}\n"

    header_path.write_text(text, encoding="utf-8")


def run_analysis(food, header_path):
    if not ANALYSIS_SLOTS.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Another HSI analysis is running. Try again later.",
        )

    try:
        return predict_uploaded_hsi_nitrogen(
            food=food,
            reflectance_header=header_path,
            max_pixels=10000,
            chunk_rows=8,
            foreground_threshold=0.05,
        )
    finally:
        ANALYSIS_SLOTS.release()


@router.post("/{food}/upload")
async def analyze_uploaded_hsi(
    food: str,
    hdr: UploadFile = File(...),
    bil: UploadFile = File(...),
):
    food = food.lower()

    if food not in {"eggplant", "cabbage"}:
        raise HTTPException(
            status_code=404,
            detail="Unsupported food.",
        )

    if not hdr.filename or not bil.filename:
        raise HTTPException(
            status_code=400,
            detail="Both filenames are required.",
        )

    if Path(hdr.filename).suffix.lower() != ".hdr":
        raise HTTPException(
            status_code=400,
            detail="First file must be .hdr.",
        )

    if Path(bil.filename).suffix.lower() not in {
        ".bil", ""
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Spectral binary must be .bil or "
                "extensionless ENVI data."
            ),
        )

    UPLOAD_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with TemporaryDirectory(
            prefix="hsi_nitrogen_",
            dir=UPLOAD_ROOT,
        ) as temporary_directory:

            directory = Path(temporary_directory)

            header_path = directory / "sample.hdr"
            binary_path = directory / "sample.bil"

            await save_upload(
                hdr,
                header_path,
                MAX_HDR_BYTES,
            )

            await save_upload(
                bil,
                binary_path,
                MAX_BINARY_BYTES,
            )

            try:
                
                prepare_envi_header(
                    header_path,
                    binary_path,
                )

                # Validate ENVI metadata and binary size
                # before opening the spectral cube.
                validate_envi_upload(
                    header_path,
                    binary_path,
                )

                result = await run_in_threadpool(
                    run_analysis,
                    food,
                    header_path,
                )


            except HTTPException:
                raise

            except (ValueError, OSError, KeyError) as error:
                raise HTTPException(
                    status_code=422,
                    detail=str(error),
                )

            except Exception as error:
                print(
                    "HSI upload inference failed:",
                    repr(error),
                )

                raise HTTPException(
                    status_code=500,
                    detail="HSI analysis failed.",
                )

            return {
                "module": "experimental_hsi_nitrogen",
                "status": "completed",
                "result": result,
                "warning": result["warning"],
            }

    finally:
        await hdr.close()
        await bil.close()
