import tifffile as tiff
from pathlib import Path

file = Path(
    "datasets/raw/hyperspectral/apple_hsi/"
    "Apple-Hyperspectral images/"
    "Apple_Samples/Monostar/Fresh/1_1PU.tif"
)

img = tiff.TiffFile(file)

print("="*60)

for page in img.pages:
    print("Shape :", page.shape)
    print("Dtype :", page.dtype)

print("="*60)

print("Series:")

for s in img.series:
    print("shape =", s.shape)
    print("axes  =", s.axes)

print("="*60)

print("Metadata:")
print(img.pages[0].tags)