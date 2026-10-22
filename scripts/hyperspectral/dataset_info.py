from pathlib import Path

import spectral as spy
import pandas as pd

from loaders import HyperspectralLoader


def print_separator(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ---------------------------------------------------
# PAPAYA
# ---------------------------------------------------

papaya_hdr = Path(
    "datasets/raw/hyperspectral/papaya_hsi/"
    "Garillos-Manliguez_Hyperspectral Images of Papaya/"
    "HSI_RAW_DATA--cropped-subregions-32x32/"
    "1-10-a-s1.hdr"
)

img, cube = HyperspectralLoader.load_hdr(papaya_hdr)

print_separator("PAPAYA")

print("Shape :", cube.shape)
print("Bands :", cube.shape[2])
print("Datatype :", cube.dtype)


# ---------------------------------------------------
# EGGPLANT
# ---------------------------------------------------

eggplant_hdr = Path(
    "datasets/raw/hyperspectral/eggplant_hsi/"
    "Eggplant_Crop/"
    "Eggplant_Reflectance_Data.hdr"
)

img, cube = HyperspectralLoader.load_hdr(eggplant_hdr)

print_separator("EGGPLANT")

print("Shape :", cube.shape)
print("Bands :", cube.shape[2])
print("Datatype :", cube.dtype)


# ---------------------------------------------------
# CABBAGE
# ---------------------------------------------------

cabbage_hdr = Path(
    "datasets/raw/hyperspectral/cabbage_hsi/"
    "Cabbage_Crop/"
    "Cabbage_Reflectance_Data.hdr"
)

img, cube = HyperspectralLoader.load_hdr(cabbage_hdr)

print_separator("CABBAGE")

print("Shape :", cube.shape)
print("Bands :", cube.shape[2])
print("Datatype :", cube.dtype)


# ---------------------------------------------------
# STRAWBERRY
# ---------------------------------------------------

strawberry_csv = Path(
    "datasets/raw/hyperspectral/strawberry_sensor/"
    "exp-1-plant.csv"
)

df = HyperspectralLoader.load_csv(strawberry_csv)

print_separator("STRAWBERRY")

print("Shape :", df.shape)
print("Columns :", len(df.columns))