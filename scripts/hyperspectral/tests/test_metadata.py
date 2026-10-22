import spectral as spy
import pandas as pd
import tifffile as tiff
from pathlib import Path


print("=" * 70)
print("PAPAYA")
print("=" * 70)

img = spy.open_image(
    r"datasets/raw/hyperspectral/papaya_hsi/"
    r"Garillos-Manliguez_Hyperspectral Images of Papaya/"
    r"HSI_RAW_DATA--cropped-subregions-32x32/"
    r"1-10-a-s1.hdr"
)

print(img.metadata)


print("\n")
print("=" * 70)
print("EGGPLANT")
print("=" * 70)

img = spy.open_image(
    r"datasets/raw/hyperspectral/eggplant_hsi/"
    r"Eggplant_Crop/"
    r"Eggplant_Reflectance_Data.hdr"
)

print(img.metadata)


print("\n")
print("=" * 70)
print("CABBAGE")
print("=" * 70)

img = spy.open_image(
    r"datasets/raw/hyperspectral/cabbage_hsi/"
    r"Cabbage_Crop/"
    r"Cabbage_Reflectance_Data.hdr"
)

print(img.metadata)