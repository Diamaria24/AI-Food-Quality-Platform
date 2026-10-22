"""
tomato_preprocessing.py

Preprocessing utilities specifically for the Tomato Disease
Hyperspectral Imaging dataset.

Expected tomato HSI cube:

    (height, width, bands)

Expected dimensions:

    140 × 280 × 550

Expected spectral range:

    400–950 nm

The dataset contains:
    - Alternaria alternata
    - Alternaria solani
    - Botrytis cinerea
    - Fusarium oxysporum
    - Healthy

IMPORTANT:
The raw dataset contains files with different numerical scales.
Some files are approximately 0–1 while others are approximately
0–4095.

Therefore scale handling is explicit and is NOT silently performed
inside the generic preprocessing pipeline.
"""

import numpy as np
from scipy.signal import savgol_filter


class TomatoHSIPreprocessor:

    EXPECTED_HEIGHT = 140
    EXPECTED_WIDTH = 280
    EXPECTED_BANDS = 550

    WAVELENGTH_START = 400.0
    WAVELENGTH_END = 950.0

    @staticmethod
    def validate_cube(cube):
        """
        Validate the dimensions of a tomato hyperspectral cube.

        Expected shape:
            (140, 280, 550)

        Returns
        -------
        bool
            True if dimensions are correct.
        """

        cube = np.asarray(cube)

        if cube.ndim != 3:
            raise ValueError(
                f"Expected a 3D hyperspectral cube, "
                f"got {cube.ndim} dimensions."
            )

        expected_shape = (
            TomatoHSIPreprocessor.EXPECTED_HEIGHT,
            TomatoHSIPreprocessor.EXPECTED_WIDTH,
            TomatoHSIPreprocessor.EXPECTED_BANDS
        )

        if cube.shape != expected_shape:
            raise ValueError(
                f"Unexpected tomato HSI shape: {cube.shape}. "
                f"Expected: {expected_shape}."
            )

        return True

    @staticmethod
    def detect_scale(cube):
        """
        Determine the approximate numerical scale of the cube.

        Categories:

            0-1-like
            4095-like
            unknown

        This function DOES NOT modify the data.
        """

        cube = np.asarray(cube, dtype=np.float32)

        finite_values = cube[np.isfinite(cube)]

        if finite_values.size == 0:
            return "unknown"

        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))

        if maximum <= 1.5 and minimum >= -0.1:
            return "0-1-like"

        if maximum <= 4095.5 and maximum >= 1000:
            return "4095-like"

        return "unknown"

    @staticmethod
    def standardize_scale(cube, scale_category=None):
        """
        Convert supported raw numerical scales to a common scale.

        For 0-1-like data:
            No scaling is required.

        For 4095-like data:
            Divide by 4095.

        This produces an approximately 0–1 representation.

        Parameters
        ----------
        cube : numpy.ndarray
            Hyperspectral cube.

        scale_category : str or None
            Previously detected scale.

        Returns
        -------
        standardized : numpy.ndarray
            Scale-standardized cube.

        detected_scale : str
            Detected/used scale category.
        """

        cube = np.asarray(cube, dtype=np.float32)

        if scale_category is None:
            scale_category = TomatoHSIPreprocessor.detect_scale(cube)

        if scale_category == "0-1-like":

            standardized = cube

        elif scale_category == "4095-like":

            standardized = cube / 4095.0

        else:

            raise ValueError(
                "Unknown numerical scale. "
                "Manual inspection is required before preprocessing."
            )

        return standardized.astype(np.float32), scale_category

    @staticmethod
    def remove_nan_inf(cube):
        """
        Replace NaN and infinite values with zero.
        """

        cube = np.asarray(cube, dtype=np.float32)

        cleaned = np.nan_to_num(
            cube,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return cleaned.astype(np.float32)

    @staticmethod
    def clip_to_valid_range(cube):
        """
        Clip standardized spectral values to [0, 1].

        This is performed AFTER scale standardization.
        """

        cube = np.asarray(cube, dtype=np.float32)

        clipped = np.clip(
            cube,
            0.0,
            1.0
        )

        return clipped.astype(np.float32)

    @staticmethod
    def spectral_smoothing(
        cube,
        window_length=7,
        polyorder=2
    ):
        """
        Apply Savitzky-Golay smoothing along the spectral axis.

        Smoothing is performed independently for every spatial pixel.
        """

        cube = np.asarray(cube, dtype=np.float32)

        bands = cube.shape[2]

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
                "Not enough spectral bands for "
                "Savitzky-Golay smoothing."
            )

        smoothed = savgol_filter(
            cube,
            window_length=window_length,
            polyorder=polyorder,
            axis=2
        )

        return smoothed.astype(np.float32)

    @staticmethod
    def mean_center(cube):
        """
        Mean-center each spectral band.

        The mean is calculated across the spatial dimensions.
        """

        cube = np.asarray(cube, dtype=np.float32)

        band_means = np.mean(
            cube,
            axis=(0, 1),
            keepdims=True
        )

        centered = cube - band_means

        return centered.astype(np.float32)

    @staticmethod
    def get_wavelengths(
        start=400.0,
        end=950.0,
        bands=550
    ):
        """
        Generate the wavelength axis.

        The tomato dataset contains 550 spectral points
        spanning 400–950 nm.
        """

        wavelengths = np.linspace(
            start,
            end,
            bands,
            dtype=np.float32
        )

        return wavelengths

    @staticmethod
    def process(
        cube,
        standardize=True,
        smoothing=True,
        mean_center=True,
        clip=True,
        window_length=7,
        polyorder=2
    ):
        """
        Run the tomato HSI preprocessing pipeline.

        Pipeline:

            Validate dimensions
                    ↓
            Detect numerical scale
                    ↓
            Standardize scale
                    ↓
            Remove NaN / Inf
                    ↓
            Clip to [0, 1]
                    ↓
            Savitzky-Golay smoothing
                    ↓
            Mean centering

        Returns
        -------
        processed_cube : numpy.ndarray

        metadata : dict
            Processing information.
        """

        cube = np.asarray(cube, dtype=np.float32)

        # --------------------------------------------------
        # 1. Validate
        # --------------------------------------------------

        TomatoHSIPreprocessor.validate_cube(cube)

        original_min = float(np.nanmin(cube))
        original_max = float(np.nanmax(cube))
        original_mean = float(np.nanmean(cube))

        # --------------------------------------------------
        # 2. Detect scale
        # --------------------------------------------------

        detected_scale = (
            TomatoHSIPreprocessor.detect_scale(cube)
        )

        # --------------------------------------------------
        # 3. Standardize numerical scale
        # --------------------------------------------------

        if standardize:

            cube, used_scale = (
                TomatoHSIPreprocessor.standardize_scale(
                    cube,
                    detected_scale
                )
            )

        else:

            used_scale = detected_scale

        # --------------------------------------------------
        # 4. Remove invalid values
        # --------------------------------------------------

        cube = (
            TomatoHSIPreprocessor.remove_nan_inf(cube)
        )

        # --------------------------------------------------
        # 5. Clip to common range
        # --------------------------------------------------

        if clip:

            cube = (
                TomatoHSIPreprocessor.clip_to_valid_range(
                    cube
                )
            )

        # --------------------------------------------------
        # 6. Spectral smoothing
        # --------------------------------------------------

        if smoothing:

            cube = (
                TomatoHSIPreprocessor.spectral_smoothing(
                    cube,
                    window_length=window_length,
                    polyorder=polyorder
                )
            )

        # --------------------------------------------------
        # 7. Mean centering
        # --------------------------------------------------

        if mean_center:

            cube = (
                TomatoHSIPreprocessor.mean_center(cube)
            )

        # --------------------------------------------------
        # 8. Final statistics
        # --------------------------------------------------

        final_min = float(np.min(cube))
        final_max = float(np.max(cube))
        final_mean = float(np.mean(cube))

        metadata = {

            "original_shape": tuple(cube.shape),

            "expected_shape": (
                TomatoHSIPreprocessor.EXPECTED_HEIGHT,
                TomatoHSIPreprocessor.EXPECTED_WIDTH,
                TomatoHSIPreprocessor.EXPECTED_BANDS
            ),

            "spectral_range_nm": (
                TomatoHSIPreprocessor.WAVELENGTH_START,
                TomatoHSIPreprocessor.WAVELENGTH_END
            ),

            "number_of_bands":
                TomatoHSIPreprocessor.EXPECTED_BANDS,

            "detected_scale":
                detected_scale,

            "used_scale":
                used_scale,

            "original_min":
                original_min,

            "original_max":
                original_max,

            "original_mean":
                original_mean,

            "final_min":
                final_min,

            "final_max":
                final_max,

            "final_mean":
                final_mean,

            "smoothing":
                smoothing,

            "savgol_window":
                window_length,

            "savgol_polyorder":
                polyorder,

            "mean_centering":
                mean_center,

            "scale_standardization":
                standardize
        }

        return cube.astype(np.float32), metadata