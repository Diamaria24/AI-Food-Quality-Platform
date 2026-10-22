from pathlib import Path
import tifffile
import pandas as pd
from spectral import open_image


print("\n" + "=" * 70)
print("APPLE")
print("=" * 70)

try:
    apple_file = next(
        Path(
            "datasets/raw/hyperspectral/apple_hsi"
        ).rglob("*.tif")
    )

    img = tifffile.imread(apple_file)

    print("File :", apple_file.name)
    print("Shape:", img.shape)
    print("Dtype:", img.dtype)

except Exception as e:
    print("Apple Error:", e)


print("\n" + "=" * 70)
print("PAPAYA")
print("=" * 70)

try:
    papaya_hdr = next(
        Path(
            "datasets/raw/hyperspectral/papaya_hsi"
        ).rglob("*.hdr")
    )

    img = open_image(str(papaya_hdr))

    print("File :", papaya_hdr.name)
    print("Shape:", img.shape)
    print("Bands:", img.nbands)
    print("Dtype:", img.dtype)

except Exception as e:
    print("Papaya Error:", e)


print("\n" + "=" * 70)
print("EGGPLANT")
print("=" * 70)

try:
    egg_hdr = next(
        Path(
            "datasets/raw/hyperspectral/eggplant_hsi"
        ).rglob("*Reflectance*.hdr")
    )

    img = open_image(str(egg_hdr))

    print("File :", egg_hdr.name)
    print("Shape:", img.shape)
    print("Bands:", img.nbands)
    print("Dtype:", img.dtype)

except Exception as e:
    print("Eggplant Error:", e)


print("\n" + "=" * 70)
print("CABBAGE")
print("=" * 70)

try:
    cab_hdr = next(
        Path(
            "datasets/raw/hyperspectral/cabbage_hsi"
        ).rglob("*Reflectance*.hdr")
    )

    img = open_image(str(cab_hdr))

    print("File :", cab_hdr.name)
    print("Shape:", img.shape)
    print("Bands:", img.nbands)
    print("Dtype:", img.dtype)

except Exception as e:
    print("Cabbage Error:", e)


print("\n" + "=" * 70)
print("STRAWBERRY")
print("=" * 70)

try:
    csv_file = Path(
        "datasets/raw/hyperspectral/strawberry_sensor/exp-1-plant.csv"
    )

    df = pd.read_csv(csv_file)

    print("Shape:", df.shape)
    print("\nColumns:\n")
    print(df.columns.tolist())

except Exception as e:
    print("Strawberry Error:", e)