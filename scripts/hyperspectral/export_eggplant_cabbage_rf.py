
from pathlib import Path
import json
import time

import joblib
import numpy as np

from scipy.signal import savgol_filter

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "hyperspectral"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "hsi"
)

CLASS_NAMES = {
    1: "Low",
    2: "Medium",
    3: "High",
}

DATASETS = {
    "eggplant": (
        INPUT_ROOT
        / "eggplant_classification"
        / "eggplant_spatial_classification.npz"
    ),
    "cabbage": (
        INPUT_ROOT
        / "cabbage_classification"
        / "cabbage_spatial_classification.npz"
    ),
}


def process_dataset(name, input_path):
    print("\n" + "=" * 70)
    print(f"PROCESSING {name.upper()}")
    print("=" * 70)

    start = time.time()

    with np.load(input_path, allow_pickle=False) as data:
        X_train = data["X_train"].astype(np.float32)
        X_test = data["X_test"].astype(np.float32)
        y_train = data["y_train"].astype(np.int64)
        y_test = data["y_test"].astype(np.int64)

    if X_train.shape[1] != 277 or X_test.shape[1] != 277:
        raise ValueError("Expected exactly 277 spectral features.")

    if not np.isfinite(X_train).all():
        raise ValueError("Training features contain NaN or Inf.")

    if not np.isfinite(X_test).all():
        raise ValueError("Testing features contain NaN or Inf.")

    print("Training shape:", X_train.shape)
    print("Testing shape :", X_test.shape)

    print("\nApplying Savitzky-Golay smoothing...")

    X_train_smooth = savgol_filter(
        X_train,
        window_length=7,
        polyorder=2,
        axis=1,
    ).astype(np.float32)

    X_test_smooth = savgol_filter(
        X_test,
        window_length=7,
        polyorder=2,
        axis=1,
    ).astype(np.float32)

    del X_train, X_test

    print("Fitting StandardScaler...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_smooth
    ).astype(np.float32)

    X_test_scaled = scaler.transform(
        X_test_smooth
    ).astype(np.float32)

    del X_train_smooth, X_test_smooth

    print("Fitting 30-component PCA...")

    pca = PCA(
        n_components=30,
        random_state=42,
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    ).astype(np.float32)

    X_test_pca = pca.transform(
        X_test_scaled
    ).astype(np.float32)

    del X_train_scaled, X_test_scaled

    print(
        "Explained variance:",
        round(
            float(pca.explained_variance_ratio_.sum()) * 100,
            2,
        ),
        "%",
    )

    print("\nTraining Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train_pca, y_train)

    print("Evaluating...")

    predictions = rf.predict(X_test_pca)

    accuracy = accuracy_score(y_test, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions,
    )

    print("\nAccuracy:", round(accuracy * 100, 2), "%")
    print(
        "Balanced accuracy:",
        round(balanced_accuracy * 100, 2),
        "%",
    )

    print(
        classification_report(
            y_test,
            predictions,
            labels=[1, 2, 3],
            target_names=["Low", "Medium", "High"],
            zero_division=0,
        )
    )

    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "rf_pipeline.joblib"

    artifact = {
        "food": name,
        "task": "N2 concentration category classification",
        "class_names": CLASS_NAMES,
        "expected_bands": 277,
        "savgol_window": 7,
        "savgol_polyorder": 2,
        "scaler": scaler,
        "pca": pca,
        "classifier": rf,
        "input_is_pre_normalized": True,
    }

    joblib.dump(artifact, model_path)

    metrics = {
        "food": name,
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "explained_variance_30pc": float(
            pca.explained_variance_ratio_.sum()
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[1, 2, 3],
        ).tolist(),
        "evaluation_note": (
            "Spatial-block pixel-level evaluation within "
            "the available dataset; not independent-food "
            "sample validation."
        ),
    }

    metrics_path = output_dir / "evaluation.json"

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSaved model:", model_path)
    print("Saved metrics:", metrics_path)
    print(
        "Elapsed seconds:",
        round(time.time() - start, 2),
    )


def main():
    for name, input_path in DATASETS.items():
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        process_dataset(name, input_path)

    print("\nAll model exports completed.")


if __name__ == "__main__":
    main()
