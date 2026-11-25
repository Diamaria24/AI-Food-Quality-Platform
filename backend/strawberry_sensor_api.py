from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from backend.database import SessionLocal
from backend.analysis_history_service import save_analysis_history

router = APIRouter(prefix="/predict/sensor", tags=["Strawberry Sensor"])

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "datasets" / "processed" / "strawberry_sensor"
ANALYSIS = ROOT / "outputs" / "strawberry_sensor_analysis"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ROWS = 100000
METADATA = {"HHMMSS", "FTime", "experiment"}
PC_COLUMNS = [f"PC{i}" for i in range(1, 9)]


@lru_cache(maxsize=1)
def load_pipeline():
    clean_path = PROCESSED / "strawberry_sensor_clean_63_features.csv"

    # Read only the header to recover the original training feature order.
    columns = pd.read_csv(clean_path, nrows=0).columns.tolist()
    features = [column for column in columns if column not in METADATA]

    scaler = joblib.load(
        PROCESSED / "strawberry_sensor_standard_scaler.joblib"
    )
    pca = joblib.load(
        PROCESSED / "strawberry_sensor_pca_model.joblib"
    )
    forest = joblib.load(
        ANALYSIS / "strawberry_sensor_isolation_forest.joblib"
    )

    if not (
        len(features) == 63
        and scaler.n_features_in_ == 63
        and pca.n_features_in_ == 63
        and forest.n_features_in_ == 8
    ):
        raise RuntimeError("Strawberry model feature dimensions do not match.")

    return features, scaler, pca, forest


@router.post("/strawberry")
async def predict_strawberry_sensor(
    request: Request,
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="CSV exceeds the 25 MB upload limit."
        )

    try:
        df = pd.read_csv(BytesIO(content), low_memory=False)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded CSV."
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV contains no data.")

    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"Maximum {MAX_ROWS} observations per upload."
        )

    try:
        features, scaler, pca, forest = load_pipeline()
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Strawberry analysis models are unavailable."
        )

    missing = [name for name in features if name not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Required Strawberry sensor features are missing.",
                "missing_columns": missing,
            },
        )

    try:
        X = df[features].apply(pd.to_numeric, errors="raise").to_numpy(
            dtype=np.float64
        )
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail="Sensor features must contain numeric values."
        )

    if not np.isfinite(X).all():
        raise HTTPException(
            status_code=400,
            detail="Sensor features contain missing or infinite values."
        )

    try:
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)[:, :8]

        pc_df = pd.DataFrame(X_pca, columns=PC_COLUMNS)
        predictions = forest.predict(pc_df)
        scores = forest.score_samples(pc_df)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Strawberry model inference failed."
        )

    anomaly_count = int(np.sum(predictions == -1))
    total = len(predictions)

    response_data = {
        "food": "strawberry",
        "analysis_type": "unsupervised_sensor_anomaly_detection",
        "filename": file.filename,
        "total_observations": total,
        "normal_observations": total - anomaly_count,
        "anomaly_observations": anomaly_count,
        "anomaly_percentage": round(100 * anomaly_count / total, 4),
        "mean_anomaly_score": round(float(np.mean(scores)), 6),
        "minimum_anomaly_score": round(float(np.min(scores)), 6),
        "maximum_anomaly_score": round(float(np.max(scores)), 6),
        "preview": [
            {
                "row": int(i + 1),
                "is_anomaly": bool(predictions[i] == -1),
                "anomaly_score": round(float(scores[i]), 6),
                "PC1": round(float(X_pca[i, 0]), 5),
                "PC2": round(float(X_pca[i, 1]), 5),
            }
            for i in range(min(total, 20))
        ],
        "interpretation": (
            "Anomalies indicate unusual sensor measurement patterns "
            "relative to the training data. They do not establish "
            "food freshness, spoilage, mold, or safety."
        ),
    }

    try:
        with SessionLocal() as history_db:
            save_analysis_history(
                db=history_db,
                user_id=request.state.user_id,
                module="sensor",
                analysis_type="unsupervised_sensor_anomaly_detection",
                food_name="Strawberry",
                prediction="Sensor anomaly analysis completed",
                confidence=None,
                result=response_data,
            )
    except Exception:
        import logging
        logging.getLogger(__name__).exception(
            "Strawberry sensor history saving failed"
        )

    return response_data
