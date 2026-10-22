import spectral as spy
from pathlib import Path

folder = Path(
    r"datasets/raw/hyperspectral/apple_hsi/"
    r"Sortedsplitted/Train/Fresh"
)

hdr_files = list(folder.rglob("*.hdr"))

print("Total hdr files:", len(hdr_files))

if hdr_files:
    file = hdr_files[0]

    img = spy.open_image(str(file))

    print("\nFile:", file.name)
    print("Shape:", img.shape)
    print("Metadata keys:")

    for k in img.metadata.keys():
        print("-", k)

    if "wavelength" in img.metadata:
        print("\nBands:", len(img.metadata["wavelength"]))