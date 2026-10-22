"""
visualization.py

Visualization utilities for hyperspectral images.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs/graphs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class HSIVisualizer:

    @staticmethod
    def _normalize(image):

        image = image.astype(np.float32)

        image -= image.min()

        maximum = image.max()

        if maximum != 0:
            image /= maximum

        return image

    @staticmethod
    def show_band(cube,
                  band,
                  save=False,
                  filename="band.png"):

        plt.figure(figsize=(6, 6))

        plt.imshow(cube[:, :, band], cmap="gray")

        plt.title(f"Band {band}")

        plt.axis("off")

        if save:
            plt.savefig(
                OUTPUT_DIR / filename,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

    @staticmethod
    def show_rgb(cube,
                 bands=(110, 75, 40),
                 save=False,
                 filename="false_rgb.png"):

        rgb = np.stack([
            cube[:, :, bands[0]],
            cube[:, :, bands[1]],
            cube[:, :, bands[2]]
        ], axis=-1)

        rgb = HSIVisualizer._normalize(rgb)

        plt.figure(figsize=(6, 6))

        plt.imshow(rgb)

        plt.title(f"False RGB {bands}")

        plt.axis("off")

        if save:
            plt.savefig(
                OUTPUT_DIR / filename,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

    @staticmethod
    def plot_spectrum(cube,
                      x,
                      y,
                      save=False,
                      filename="spectrum.png"):

        spectrum = cube[y, x, :]

        plt.figure(figsize=(8, 5))

        plt.plot(spectrum)

        plt.title(f"Pixel ({x}, {y})")

        plt.xlabel("Band")

        plt.ylabel("Reflectance")

        plt.grid(True)

        if save:
            plt.savefig(
                OUTPUT_DIR / filename,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()