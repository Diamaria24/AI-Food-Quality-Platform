
from pathlib import Path
import numpy as np
from spectral import envi

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(r"D:\Projects\AI-Food-Quality-Platform")

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
    / "pistachio_hsi"
)

# Compare a healthy sample and a contaminated sample
SAMPLES = [
    DATASET_DIR / "Level 01" / "L01_0001.bil",
    DATASET_DIR / "Level 26" / "L26_0001.bil",
]


def inspect_cube(bil_path):
    hdr_path = bil_path.with_suffix(".hdr")

    print("\n" + "=" * 70)
    print(f"FILE: {bil_path.name}")
    print("=" * 70)

    if not bil_path.exists():
        print("ERROR: BIL file not found.")
        return

    if not hdr_path.exists():
        print("ERROR: HDR file not found.")
        return

    image = envi.open(str(hdr_path), str(bil_path))

    print("\nHEADER INFORMATION")
    print("-" * 50)

    for key in [
        "samples",
        "lines",
        "bands",
        "data type",
        "interleave",
        "reflectance scale factor",
    ]:
        print(f"{key}: {image.metadata.get(key, 'NOT FOUND')}")

    # Memory-map the original data without modifying it
    cube = image.open_memmap()

    print("\nRAW DATA INFORMATION")
    print("-" * 50)

    print("Shape:", cube.shape)
    print("Data type:", cube.dtype)

    # Sample pixels to avoid unnecessary memory usage
    rng = np.random.default_rng(42)

    total_pixels = cube.shape[0] * cube.shape[1]
    sample_count = min(5000, total_pixels)

    selected = rng.choice(
        total_pixels,
        size=sample_count,
        replace=False
    )

    rows, cols = np.unravel_index(
        selected,
        cube.shape[:2]
    )

    spectra = np.asarray(
        cube[rows, cols, :],
        dtype=np.float64
    )

    finite = spectra[np.isfinite(spectra)]

    if finite.size == 0:
        print("ERROR: No finite values found.")
        return

    print("\nRAW PIXEL STATISTICS")
    print("-" * 50)

    print(f"Minimum : {finite.min():.10f}")
    print(f"Maximum : {finite.max():.10f}")
    print(f"Mean    : {finite.mean():.10f}")
    print(f"Median  : {np.median(finite):.10f}")

    print("\nPERCENTILES")
    print("-" * 50)

    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(
            f"{p:2d}th percentile : "
            f"{np.percentile(finite, p):.10f}"
        )

    print("\nVALUE DISTRIBUTION")
    print("-" * 50)

    print("Values < 0:", np.sum(finite < 0))
    print("Values 0 to 1:", np.sum(
        (finite >= 0) & (finite <= 1)
    ))
    print("Values > 1:", np.sum(finite > 1))
    print("Values > 100:", np.sum(finite > 100))
    print("Values > 1000:", np.sum(finite > 1000))

    print("\nSCALING COMPARISON")
    print("-" * 50)

    print(
        f"Original maximum   : {finite.max():.10f}"
    )
    print(
        f"Maximum after /10000: "
        f"{(finite / 10000).max():.10f}"
    )

    print("\nEXAMPLE RAW SPECTRAL VALUES")
    print("-" * 50)

    print("First 15 bands of sampled spectrum:")
    print(np.round(spectra[0, :15], 8))


print("=" * 70)
print("PISTACHIO HSI REFLECTANCE SCALE DIAGNOSTIC")
print("=" * 70)

for sample in SAMPLES:
    inspect_cube(sample)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETED")
print("=" * 70)
