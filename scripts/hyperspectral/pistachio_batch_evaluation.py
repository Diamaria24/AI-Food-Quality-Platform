"""Batch experimental Pistachio HSI inference and cube-level evaluation.

Place beside pistachio_xgboost_inference.py and run from the project root:
  python scripts/hyperspectral/pistachio_batch_evaluation.py

Uses existing train/test manifests, PCA and XGBoost. No retraining.
Only load trusted local joblib artifacts.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from pistachio_xgboost_inference import load_cube, extract_spectra, snv

LABELS = {0: "Below_Threshold", 1: "At_or_Above_Threshold"}
REQUIRED = {"Sample_ID", "Level", "AFB1_ug_kg", "Binary_Label", "Binary_Class", "BIL_Path", "HDR_Path", "Group_ID"}


def resolve_file(value, root, manifest):
    path = Path(str(value).strip())
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        for candidate in (root / path, manifest.parent / path):
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(f"File not found from manifest {manifest}: {value}")


def metrics_for(df):
    if df.empty:
        return None
    true = df["true_label"].to_numpy(dtype=int)
    pred = df["predicted_label"].to_numpy(dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(true, pred, labels=[0, 1]).tolist()
    return {
        "n_cubes": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix_rows_true_0_1_columns_pred_0_1": cm,
        "per_class": {
            LABELS[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support_cubes": int(support[i]),
            } for i in (0, 1)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-pixel-predictions", action="store_true",
                        help="Also save a CSV for every cube (more disk usage)")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    root = args.project_root.resolve()
    data_dir = root / "datasets/processed/hyperspectral/pistachio_classification"
    pca_dir = data_dir / "pca/pca_100pc"
    manifests = {
        "train": data_dir / "pistachio_train_group_split.csv",
        "test": data_dir / "pistachio_test_group_split.csv",
    }
    pca_path = pca_dir / "pistachio_pca_model.joblib"
    model_path = pca_dir / "pistachio_final_xgboost_model.joblib"
    for path in (*manifests.values(), pca_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    frames = {}
    for split, path in manifests.items():
        df = pd.read_csv(path)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if df[list(REQUIRED)].isna().any().any():
            raise ValueError(f"{path}: missing manifest values")
        df["Binary_Label"] = pd.to_numeric(df["Binary_Label"], errors="raise")
        if not df["Binary_Label"].isin([0, 1]).all():
            raise ValueError(f"{path}: labels must be 0 or 1")
        frames[split] = df

    for key in ("Sample_ID", "Group_ID"):
        overlap = set(frames["train"][key].astype(str)) & set(frames["test"][key].astype(str))
        if overlap:
            raise ValueError(f"Train/test {key} overlap: {sorted(overlap)}")
    for split, df in frames.items():
        if df["Sample_ID"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate Sample_ID within {split} manifest")
    print("PASS: No train/test Sample_ID or Group_ID overlap", flush=True)

    pca = joblib.load(pca_path)
    model = joblib.load(model_path)
    if getattr(pca, "n_features_in_", None) != 462:
        raise ValueError("PCA input is not 462 bands")
    if getattr(model, "n_features_in_", None) != 100:
        raise ValueError("XGBoost input is not 100 PCs")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected XGBoost classes")

    out = args.output_dir or root / "outputs/pistachio_batch_evaluation"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for split, manifest in manifests.items():
        df = frames[split]
        for index, row in df.iterrows():
            sample_id = str(row["Sample_ID"])
            bil = resolve_file(row["BIL_Path"], root, manifest)
            hdr = resolve_file(row["HDR_Path"], root, manifest)
            print(f"[{split} {len(records) + 1}] {sample_id}", flush=True)
            raw = load_cube(bil, hdr)
            spectra, positions, n_foreground = extract_spectra(raw, args.samples, args.seed)
            del raw
            features = pca.transform(snv(spectra))[:, :100].astype(np.float32)
            if not np.isfinite(features).all():
                raise ValueError(f"Nonfinite PCA values: {sample_id}")
            pred = model.predict(features).astype(int)
            prob = model.predict_proba(features)[:, 1]
            positive_fraction = float(np.mean(pred == 1))
            cube_label = int(positive_fraction >= 0.5)
            true_label = int(row["Binary_Label"])
            record = {
                "split": split,
                "sample_id": sample_id,
                "level": int(row["Level"]),
                "group_id": str(row["Group_ID"]),
                "afb1_ug_kg": float(row["AFB1_ug_kg"]),
                "true_label": true_label,
                "true_class": LABELS[true_label],
                "predicted_label": cube_label,
                "predicted_class": LABELS[cube_label],
                "correct": bool(cube_label == true_label),
                "foreground_pixels": int(n_foreground),
                "sampled_pixels": int(len(pred)),
                "predicted_below_pixels": int(np.sum(pred == 0)),
                "predicted_at_or_above_pixels": int(np.sum(pred == 1)),
                "fraction_predicted_at_or_above": positive_fraction,
                "mean_model_probability_at_or_above": float(np.mean(prob)),
                "bil_path": str(bil),
                "hdr_path": str(hdr),
            }
            records.append(record)
            print(f"    true={true_label} predicted={cube_label} positive_fraction={positive_fraction:.4f}", flush=True)
            if args.save_pixel_predictions:
                pixel_dir = out / "pixels" / split / sample_id
                pixel_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "row": positions[:, 0],
                    "column": positions[:, 1],
                    "predicted_label": pred,
                    "model_probability_at_or_above": prob,
                }).to_csv(pixel_dir / "pixel_predictions.csv", index=False)

    results = pd.DataFrame.from_records(records)
    results.to_csv(out / "cube_predictions.csv", index=False)
    report = {
        "method": "Experimental majority vote across up to 2000 randomly sampled foreground pixels per cube",
        "seed_reset_for_each_cube": args.seed,
        "threshold_ug_per_kg": 8.0,
        "pca_model": str(pca_path),
        "xgboost_model": str(model_path),
        "train": metrics_for(results[results["split"] == "train"]),
        "held_out_test": metrics_for(results[results["split"] == "test"]),
        "caution": "Exploratory cube aggregation. Pixel probabilities are uncalibrated; no food-safety certification. "
                   "Cube-level metrics are distinct from the original pixel-level metrics. "
                   "Training split is not independent evaluation.",
    }
    (out / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nHELD-OUT TEST METRICS")
    print(json.dumps(report["held_out_test"], indent=2))
    print(f"\nSaved: {out / 'cube_predictions.csv'}")
    print(f"Saved: {out / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
