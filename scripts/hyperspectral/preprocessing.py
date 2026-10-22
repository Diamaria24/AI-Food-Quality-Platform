"""
preprocessing.py

Preprocessing utilities for hyperspectral image cubes
and sampled hyperspectral pixels.

Cube shape:
    (height, width, bands)

Sample matrix shape:
    (samples, bands)
"""

import numpy as np
from scipy.signal import savgol_filter


class HSIProcessor:

    # ==========================================================
    # REMOVE NaN / INFINITE VALUES
    # ==========================================================

    @staticmethod
    def remove_nan_inf(data):
        """
        Replace NaN and infinite values with 0.

        Works with both:
            3D hyperspectral cubes
            2D sampled spectral pixels
        """

        data = np.asarray(data, dtype=np.float32)

        cleaned = np.nan_to_num(
            data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return cleaned.astype(np.float32)

    # ==========================================================
    # NORMALIZATION - COMPLETE CUBE
    # ==========================================================

    @staticmethod
    def normalize(cube):
        """
        Min-Max normalization for a complete hyperspectral cube.

        Converts values to approximately [0, 1].
        """

        cube = np.asarray(cube, dtype=np.float32)

        minimum = np.min(cube)
        maximum = np.max(cube)

        if maximum == minimum:
            return np.zeros_like(cube)

        normalized = (
            (cube - minimum)
            / (maximum - minimum)
        )

        return normalized.astype(np.float32)

    # ==========================================================
    # NORMALIZATION - SAMPLED PIXELS
    # ==========================================================

    @staticmethod
    def normalize_samples(samples):
        """
        Min-Max normalization for sampled spectral pixels.

        Input:
            (samples, bands)
        """

        samples = np.asarray(
            samples,
            dtype=np.float32
        )

        minimum = np.min(samples)
        maximum = np.max(samples)

        if maximum == minimum:
            return np.zeros_like(samples)

        normalized = (
            (samples - minimum)
            / (maximum - minimum)
        )

        return normalized.astype(np.float32)

    # ==========================================================
    # SPECTRAL SMOOTHING - COMPLETE CUBE
    # ==========================================================

    @staticmethod
    def spectral_smoothing(
        cube,
        window_length=7,
        polyorder=2
    ):
        """
        Apply Savitzky-Golay smoothing along spectral bands.

        For a cube:
            axis=2
        """

        cube = np.asarray(
            cube,
            dtype=np.float32
        )

        if cube.ndim != 3:
            raise ValueError(
                "Expected 3D cube with shape "
                "(height, width, bands)."
            )

        bands = cube.shape[2]

        # Make window valid
        if window_length >= bands:

            window_length = (
                bands
                if bands % 2 == 1
                else bands - 1
            )

        if window_length <= polyorder:

            window_length = polyorder + 2

            if window_length % 2 == 0:
                window_length += 1

        if window_length > bands:
            raise ValueError(
                "Not enough spectral bands "
                "for Savitzky-Golay smoothing."
            )

        smoothed = savgol_filter(
            cube,
            window_length=window_length,
            polyorder=polyorder,
            axis=2
        )

        return smoothed.astype(np.float32)

    # ==========================================================
    # SPECTRAL SMOOTHING - SAMPLED PIXELS
    # ==========================================================

    @staticmethod
    def spectral_smoothing_samples(
        samples,
        window_length=7,
        polyorder=2
    ):
        """
        Apply Savitzky-Golay smoothing to sampled pixels.

        Input:
            (samples, bands)

        Smoothing is applied along:
            axis=1
        """

        samples = np.asarray(
            samples,
            dtype=np.float32
        )

        if samples.ndim != 2:
            raise ValueError(
                "Expected 2D samples with shape "
                "(samples, bands)."
            )

        bands = samples.shape[1]

        if bands < 3:
            return samples

        # Make window valid
        if window_length >= bands:

            window_length = (
                bands
                if bands % 2 == 1
                else bands - 1
            )

        if window_length <= polyorder:

            window_length = polyorder + 2

            if window_length % 2 == 0:
                window_length += 1

        if window_length > bands:
            return samples

        smoothed = savgol_filter(
            samples,
            window_length=window_length,
            polyorder=polyorder,
            axis=1
        )

        return smoothed.astype(np.float32)

    # ==========================================================
    # MEAN CENTER - COMPLETE CUBE
    # ==========================================================

    @staticmethod
    def mean_center(cube):
        """
        Mean-center each spectral band of a cube.
        """

        cube = np.asarray(
            cube,
            dtype=np.float32
        )

        if cube.ndim != 3:
            raise ValueError(
                "Expected 3D hyperspectral cube."
            )

        band_means = np.mean(
            cube,
            axis=(0, 1),
            keepdims=True
        )

        centered = cube - band_means

        return centered.astype(np.float32)

    # ==========================================================
    # MEAN CENTER - SAMPLED PIXELS
    # ==========================================================

    @staticmethod
    def mean_center_samples(samples):
        """
        Mean-center each spectral band
        using sampled pixels.
        """

        samples = np.asarray(
            samples,
            dtype=np.float32
        )

        if samples.ndim != 2:
            raise ValueError(
                "Expected 2D samples."
            )

        band_means = np.mean(
            samples,
            axis=0,
            keepdims=True
        )

        centered = samples - band_means

        return centered.astype(np.float32)

    # ==========================================================
    # COMPLETE CUBE PIPELINE
    # ==========================================================

    @staticmethod
    def process(cube):
        """
        Complete preprocessing pipeline for a cube.

        Pipeline:

        Remove invalid values
                ↓
        Min-Max normalization
                ↓
        Spectral smoothing
                ↓
        Mean centering
        """

        cube = HSIProcessor.remove_nan_inf(cube)

        cube = HSIProcessor.normalize(cube)

        cube = HSIProcessor.spectral_smoothing(cube)

        cube = HSIProcessor.mean_center(cube)

        return cube.astype(np.float32)

    # ==========================================================
    # SAMPLED PIXEL PIPELINE
    # ==========================================================

    @staticmethod
    def process_samples(samples):
        """
        Memory-efficient preprocessing pipeline
        for sampled spectral pixels.

        Input:
            (samples, bands)

        Pipeline:

        Remove invalid values
                ↓
        Min-Max normalization
                ↓
        Spectral smoothing
                ↓
        Mean centering
        """

        samples = HSIProcessor.remove_nan_inf(
            samples
        )

        samples = HSIProcessor.normalize_samples(
            samples
        )

        samples = (
            HSIProcessor.spectral_smoothing_samples(
                samples
            )
        )

        samples = (
            HSIProcessor.mean_center_samples(
                samples
            )
        )

        return samples.astype(np.float32)