"""
loaders.py

Functions to load different hyperspectral datasets used in the project.
Includes memory-efficient loading for large hyperspectral cubes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import spectral as spy


class HyperspectralLoader:

    @staticmethod
    def load_hdr(hdr_path):
        """
        Load a complete ENVI HDR hyperspectral cube.

        Suitable for small datasets such as Papaya.

        Returns
        -------
        img : SpyFile
            Spectral Python image object.

        cube : numpy.ndarray
            Complete hyperspectral cube.
        """

        hdr_path = Path(hdr_path)

        img = spy.open_image(str(hdr_path))

        cube = np.asarray(img.load(), dtype=np.float32)

        return img, cube

    @staticmethod
    def open_hdr_memmap(hdr_path):
        """
        Open an ENVI HDR file using memory mapping.

        This avoids loading the complete hyperspectral cube
        into RAM.

        Suitable for large datasets such as Eggplant and Cabbage.
        """

        hdr_path = Path(hdr_path)

        img = spy.open_image(str(hdr_path))

        # Open using memory mapping
        cube = img.open_memmap()

        return img, cube

    @staticmethod
    def sample_hdr(hdr_path, sample_size=10000, random_state=42):
        """
        Randomly sample pixels from a large hyperspectral cube.

        Only the required pixels are read into memory.

        Parameters
        ----------
        hdr_path : str or Path
            Path to the HDR file.

        sample_size : int
            Maximum number of pixels to sample.

        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        img : SpyFile
            Spectral Python image object.

        samples : numpy.ndarray
            Sampled spectral pixels.
            Shape: (number_of_samples, bands)

        cube_shape : tuple
            Original hyperspectral cube shape.
        """

        hdr_path = Path(hdr_path)

        img = spy.open_image(str(hdr_path))

        # Memory mapped cube
        cube = img.open_memmap()

        height, width, bands = cube.shape

        total_pixels = height * width

        print("Original cube :", cube.shape)
        print("Total pixels  :", total_pixels)

        # Determine number of samples
        actual_sample_size = min(sample_size, total_pixels)

        rng = np.random.default_rng(random_state)

        # Random pixel indices
        indices = rng.choice(
            total_pixels,
            size=actual_sample_size,
            replace=False
        )

        rows = indices // width
        cols = indices % width

        # Read only selected pixels
        samples = np.asarray(
            cube[rows, cols, :],
            dtype=np.float32
        )

        print("Sampled pixels:", samples.shape)

        return img, samples, cube.shape

    @staticmethod
    def load_csv(csv_path):
        """
        Load Strawberry sensor CSV.
        """

        return pd.read_csv(csv_path)

    @staticmethod
    def load_tiff(tiff_path):
        """
        Load TIFF image.
        """

        img = spy.open_image(str(tiff_path))

        cube = np.asarray(
            img.load(),
            dtype=np.float32
        )

        return img, cube