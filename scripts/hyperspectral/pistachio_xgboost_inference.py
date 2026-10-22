"""Experimental Pistachio HSI inference, matching the original preprocessing.

Usage (from project root):
 python scripts/hyperspectral/pistachio_xgboost_inference.py --hdr "path/to/cube.hdr" --bil "path/to/cube.bil"

Only load trusted, locally generated joblib artifacts.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

EXPECTED_SHAPE = (256, 462, 384)  # BIL: lines, bands, samples
SCALE = 10000.0
CLIP_MAX = 1.5
FOREGROUND_THRESHOLD = 0.02
N_SPECTRA = 2000
SEED = 42
CLASS_NAMES = {0: "Below_Threshold", 1: "At_or_Above_Threshold"}


def parse_hdr(path):
    import re
    text = path.read_text(encoding="utf-8", errors="ignore")
    patterns = {
        "samples": r"(?im)^\s*samples\s*=\s*(\d+)",
        "lines": r"(?im)^\s*lines\s*=\s*(\d+)",
        "bands": r"(?im)^\s*bands\s*=\s*(\d+)",
        "data_type": r"(?im)^\s*data\s*type\s*=\s*(\d+)",
        "interleave": r"(?im)^\s*interleave\s*=\s*([^\s]+)",
        "byte_order": r"(?im)^\s*byte order\s*=\s*(\d+)",
        "header_offset": r"(?im)^\s*header offset\s*=\s*(\d+)",
    }
    result = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = match.group(1).lower() if key == "interleave" else int(match.group(1))
    required = set(patterns) - {"header_offset"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Missing ENVI HDR fields: {sorted(missing)}")
    if result.get("header_offset", 0) != 0:
        raise ValueError("Nonzero header offset not supported by the original training loader")
    return result


def load_cube(bil_path, hdr_path):
    meta = parse_hdr(hdr_path)
    shape = (meta["lines"], meta["bands"], meta["samples"])
    if shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected BIL shape {shape}; expected {EXPECTED_SHAPE}")
    if meta["interleave"] != "bil" or meta["data_type"] != 12:
        raise ValueError("Expected ENVI BIL with data type 12 (uint16)")
    if meta["byte_order"] not in (0, 1):
        raise ValueError("Unsupported ENVI byte order")
    dtype = np.dtype("<u2" if meta["byte_order"] == 0 else ">u2")
    expected_bytes = int(np.prod(shape)) * dtype.itemsize
    if bil_path.stat().st_size != expected_bytes:
        raise ValueError(f"BIL size mismatch: expected {expected_bytes}, found {bil_path.stat().st_size}")
    return np.memmap(bil_path, dtype=dtype, mode="r", shape=shape)


def extract_spectra(raw_cube, n_spectra, seed):
    # Original pipeline: convert entire raw cube to float32 reflectance,
    # clip to [0, 1.5], mask pixels by mean reflectance > 0.02,
    # then uniformly sample without replacement.
    reflectance = np.clip(raw_cube.astype(np.float32) / SCALE, 0.0, CLIP_MAX)
    mean_reflectance = np.mean(reflectance, axis=1)
    positions = np.argwhere(mean_reflectance > FOREGROUND_THRESHOLD)
    if len(positions) == 0:
        raise ValueError("No foreground pixels with mean reflectance > 0.02")
    rng = np.random.default_rng(seed)
    chosen = positions[rng.choice(len(positions), size=min(n_spectra, len(positions)), replace=False)]
    spectra = np.empty((len(chosen), EXPECTED_SHAPE[1]), dtype=np.float32)
    for i, (row, col) in enumerate(chosen):
        spectra[i] = reflectance[row, :, col]
    return spectra, chosen, len(positions)


def snv(spectra):
    means = np.mean(spectra, axis=1, keepdims=True)
    stds = np.std(spectra, axis=1, keepdims=True)
    stds[stds < 1e-8] = 1.0
    return ((spectra - means) / stds).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdr", type=Path, required=True)
    parser.add_argument("--bil", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=N_SPECTRA)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    root = args.project_root.resolve()
    pca_dir = root / "datasets/processed/hyperspectral/pistachio_classification/pca/pca_100pc"
    pca_path = pca_dir / "pistachio_pca_model.joblib"
    model_path = pca_dir / "pistachio_final_xgboost_model.joblib"
    for path in (args.hdr, args.bil, pca_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    out = args.output_dir or root / "outputs/pistachio_inference" / args.bil.stem
    out.mkdir(parents=True, exist_ok=True)

    # joblib loading is safe only for artifacts from trusted sources.
    pca = joblib.load(pca_path)
    model = joblib.load(model_path)
    if getattr(pca, "n_features_in_", None) != 462:
        raise ValueError("PCA does not accept 462 input bands")
    if getattr(model, "n_features_in_", None) != 100:
        raise ValueError("XGBoost does not accept 100 PCA features")
    if list(model.classes_) != [0, 1]:
        raise ValueError("Unexpected classifier label mapping")

    raw = load_cube(args.bil, args.hdr)
    spectra, positions, foreground_count = extract_spectra(raw, args.samples, args.seed)
    normalized = snv(spectra)
    X = pca.transform(normalized)[:, :100].astype(np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite PCA features detected")
    predictions = model.predict(X).astype(int)
    probability_at_or_above = model.predict_proba(X)[:, 1]
    results = pd.DataFrame({
        "row": positions[:, 0],
        "column": positions[:, 1],
        "predicted_label": predictions,
        "predicted_class": [CLASS_NAMES[v] for v in predictions],
        "model_probability_at_or_above": probability_at_or_above,
    })
    results.to_csv(out / "pixel_predictions.csv", index=False)

    positive_fraction = float(np.mean(predictions == 1))
    summary = {
        "cube": str(args.bil.resolve()),
        "hdr": str(args.hdr.resolve()),
        "model": "pistachio_final_xgboost_model.joblib",
        "pca": "pistachio_pca_model.joblib (first 100 of 462 components)",
        "afb1_threshold_ug_per_kg": 8.0,
        "label_mapping": CLASS_NAMES,
        "foreground_pixels": foreground_count,
        "sampled_pixels": len(predictions),
        "random_seed": args.seed,
        "predicted_below_threshold_pixels": int(np.sum(predictions == 0)),
        "predicted_at_or_above_threshold_pixels": int(np.sum(predictions == 1)),
        "fraction_predicted_at_or_above": positive_fraction,
        "mean_model_probability_at_or_above": float(np.mean(probability_at_or_above)),
        "exploratory_majority_vote": CLASS_NAMES[int(positive_fraction >= 0.5)],
        "note": "Exploratory pixel aggregation only; not a validated cube-level AFB1 estimate, safety certification, or calibrated confidence.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {out / 'pixel_predictions.csv'}")
    print(f"Saved: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
