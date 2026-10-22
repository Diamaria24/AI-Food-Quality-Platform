from pathlib import Path

from loaders import HyperspectralLoader
from visualization import HSIVisualizer


papaya_path = Path(
    "datasets/raw/hyperspectral/"
    "papaya_hsi/"
    "Garillos-Manliguez_Hyperspectral Images of Papaya/"
    "HSI_RAW_DATA--cropped-subregions-32x32/"
    "1-10-a-s1.hdr"
)

cube = HyperspectralLoader.load_hdr(papaya_path)

print("Cube Shape :", cube.shape)

print()

print("Displaying Band...")

HSIVisualizer.show_band(
    cube,
    band=70,
    save=True,
    filename="papaya_band70.png"
)

print("Displaying False RGB...")

HSIVisualizer.show_rgb(
    cube,
    bands=(110, 75, 40),
    save=True,
    filename="papaya_false_rgb.png"
)

print("Displaying Spectrum...")

HSIVisualizer.plot_spectrum(
    cube,
    x=16,
    y=16,
    save=True,
    filename="papaya_spectrum.png"
)

print()

print("Graphs saved to outputs/graphs/")