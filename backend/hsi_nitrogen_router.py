
from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.hsi_nitrogen_inference import (
    predict_hsi_nitrogen,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
)

DATASETS = {
    "eggplant": {
        "reflectance": (
            DATA_ROOT
            / "eggplant_hsi"
            / "Eggplant_Crop"
            / "Eggplant_Reflectance_Data.hdr"
        ),
        "ground_truth": (
            DATA_ROOT
            / "eggplant_hsi"
            / "Eggplant_Crop"
            / "Eggplant_N2_Concentration_GT.hdr"
        ),
    },
    "cabbage": {
        "reflectance": (
            DATA_ROOT
            / "cabbage_hsi"
            / "Cabbage_Crop"
            / "Cabbage_Reflectance_Data.hdr"
        ),
        "ground_truth": (
            DATA_ROOT
            / "cabbage_hsi"
            / "Cabbage_Crop"
            / "Cabbage_N2_Concentration_GT.hdr"
        ),
    },
}

router = APIRouter(
    prefix="/predict/hsi",
    tags=["Experimental HSI Nitrogen Classification"],
)


@router.post("/{food}/local")
async def predict_local_hsi(food: str):
    food = food.lower()

    if food not in DATASETS:
        raise HTTPException(
            status_code=404,
            detail="Unsupported HSI dataset.",
        )

    dataset = DATASETS[food]

    for path in dataset.values():
        if not path.is_file():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Required local HSI dataset file "
                    "is unavailable."
                ),
            )

    try:
        result = await run_in_threadpool(
            predict_hsi_nitrogen,
            food=food,
            reflectance_header=dataset["reflectance"],
            ground_truth_header=dataset["ground_truth"],
            max_pixels=10000,
            chunk_rows=8,
        )

    except Exception as exc:
        print(
            f"{food} HSI inference failed:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Experimental HSI inference failed. "
                "Check backend logs."
            ),
        ) from exc

    return {
        "module": "hsi_nitrogen",
        "source": "registered_local_dataset",
        "experimental": True,
        **result,
    }
