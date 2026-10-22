"""
pca_analysis.py

PCA utilities for hyperspectral image data.

Supports:

1. Complete small HSI cubes
2. Memory-efficient sampled pixels

PCA automatically determines the number of
components required to reach the target variance.

Default:
    Target variance = 95%
    Maximum components = 50
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

OUTPUT_DIR = Path("outputs/graphs")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


class HSI_PCA:

    # ======================================================
    # PREPARE PIXELS FROM COMPLETE CUBE
    # ======================================================

    @staticmethod
    def prepare_pixels(
        cube,
        max_samples=10000,
        random_state=42
    ):
        """
        Convert a 3D HSI cube into a 2D
        pixel-by-band matrix.

        NOTE:
        This method is suitable only when the cube
        is already safely loaded into memory.
        """

        cube = np.asarray(
            cube,
            dtype=np.float32
        )

        if cube.ndim != 3:
            raise ValueError(
                "Expected HSI cube with shape "
                "(height, width, bands)."
            )

        height, width, bands = cube.shape

        print("\nPreparing pixels...")
        print("Original cube :", cube.shape)

        pixels = cube.reshape(
            -1,
            bands
        )

        print(
            "Total pixels  :",
            pixels.shape[0]
        )

        valid = np.isfinite(
            pixels
        ).all(axis=1)

        pixels = pixels[valid]

        print(
            "Valid pixels  :",
            pixels.shape[0]
        )

        if len(pixels) > max_samples:

            print(
                f"Sampling {max_samples} pixels "
                f"for PCA..."
            )

            rng = np.random.default_rng(
                random_state
            )

            indices = rng.choice(
                len(pixels),
                size=max_samples,
                replace=False
            )

            pixels = pixels[indices]

        else:

            print(
                "No sampling required."
            )

        print(
            "PCA input     :",
            pixels.shape
        )

        return pixels

    # ======================================================
    # PREPARE ALREADY-SAMPLED PIXELS
    # ======================================================

    @staticmethod
    def prepare_samples(
        samples,
        max_samples=10000,
        random_state=42
    ):
        """
        Prepare a 2D sampled spectral matrix.

        Input:
            (samples, bands)

        This method does NOT create a huge
        intermediate array.
        """

        samples = np.asarray(
            samples,
            dtype=np.float32
        )

        if samples.ndim != 2:
            raise ValueError(
                "Expected samples with shape "
                "(samples, bands)."
            )

        print("\nPreparing sampled pixels...")

        print(
            "Input samples :",
            samples.shape
        )

        valid = np.isfinite(
            samples
        ).all(axis=1)

        samples = samples[valid]

        print(
            "Valid samples :",
            samples.shape[0]
        )

        if len(samples) > max_samples:

            print(
                f"Sampling {max_samples} "
                f"spectral pixels..."
            )

            rng = np.random.default_rng(
                random_state
            )

            indices = rng.choice(
                len(samples),
                size=max_samples,
                replace=False
            )

            samples = samples[indices]

        else:

            print(
                "No additional sampling required."
            )

        print(
            "PCA input     :",
            samples.shape
        )

        return samples

    # ======================================================
    # FIT PCA ON COMPLETE CUBE
    # ======================================================

    @staticmethod
    def fit(
        cube,
        variance_threshold=0.95,
        max_components=50,
        max_samples=10000,
        random_state=42
    ):
        """
        Fit PCA using a complete cube.

        The cube is converted into pixels and sampled
        after it has already been loaded.
        """

        pixels = HSI_PCA.prepare_pixels(
            cube,
            max_samples=max_samples,
            random_state=random_state
        )

        return HSI_PCA.fit_samples(
            pixels,
            variance_threshold=variance_threshold,
            max_components=max_components,
            max_samples=max_samples,
            random_state=random_state
        )

    # ======================================================
    # FIT PCA ON SAMPLED DATA
    # ======================================================

    @staticmethod
    def fit_samples(
        samples,
        variance_threshold=0.95,
        max_components=50,
        max_samples=10000,
        random_state=42
    ):
        """
        Fit PCA directly on sampled spectral pixels.

        Input:
            samples = (samples, bands)

        This is the memory-efficient method used
        for large datasets such as Cabbage.
        """

        pixels = HSI_PCA.prepare_samples(
            samples,
            max_samples=max_samples,
            random_state=random_state
        )

        n_samples, n_bands = pixels.shape

        # --------------------------------------------------
        # Maximum possible components
        # --------------------------------------------------

        max_possible = min(
            n_samples,
            n_bands
        )

        max_components = min(
            max_components,
            max_possible
        )

        print("\nFitting PCA...")

        print(
            "Samples        :",
            n_samples
        )

        print(
            "Spectral bands :",
            n_bands
        )

        print(
            "Maximum PCs    :",
            max_components
        )

        # --------------------------------------------------
        # FULL PCA
        #
        # Needed to determine exactly how many
        # components are required for 95%.
        # --------------------------------------------------

        pca_full = PCA(
            n_components=max_possible,
            svd_solver="full"
        )

        pca_full.fit(
            pixels
        )

        variance = (
            pca_full
            .explained_variance_ratio_
        )

        cumulative_variance = np.cumsum(
            variance
        )

        # --------------------------------------------------
        # Determine required components
        # --------------------------------------------------

        indices = np.where(
            cumulative_variance
            >= variance_threshold
        )[0]

        if len(indices) > 0:

            required_components = (
                int(indices[0]) + 1
            )

        else:

            required_components = (
                max_possible
            )

        # --------------------------------------------------
        # Apply safety limit
        # --------------------------------------------------

        selected_components = min(
            required_components,
            max_components
        )

        actual_variance = (
            cumulative_variance[
                selected_components - 1
            ]
        )

        # --------------------------------------------------
        # Final PCA
        # --------------------------------------------------

        pca = PCA(
            n_components=selected_components,
            svd_solver="full"
        )

        transformed = pca.fit_transform(
            pixels
        )

        # --------------------------------------------------
        # Print information
        # --------------------------------------------------

        print("\n" + "=" * 65)
        print("PCA COMPONENT SELECTION")
        print("=" * 65)

        print(
            f"Target variance        : "
            f"{variance_threshold * 100:.2f}%"
        )

        print(
            f"Components required    : "
            f"{required_components}"
        )

        print(
            f"Maximum allowed        : "
            f"{max_components}"
        )

        print(
            f"Components selected    : "
            f"{selected_components}"
        )

        print(
            f"Actual retained        : "
            f"{actual_variance * 100:.2f}%"
        )

        if required_components > max_components:

            print("\nWARNING:")

            print(
                f"{variance_threshold * 100:.0f}% "
                f"variance requires "
                f"{required_components} components, "
                f"but the safety limit is "
                f"{max_components}."
            )

            print(
                "The selected PCA representation "
                "therefore retains less than the "
                "target variance."
            )

        else:

            print(
                "\n95% variance target achieved."
            )

        return (
            pca,
            pixels,
            transformed,
            selected_components,
            cumulative_variance
        )

    # ======================================================
    # ALIAS FOR COMPATIBILITY
    # ======================================================

    @staticmethod
    def fit_transform_samples(
        samples,
        target_variance=0.95,
        max_components=50,
        dataset_name="Dataset",
        output_dir=OUTPUT_DIR
    ):
        """
        Compatibility wrapper for scripts that use
        fit_transform_samples().
        """

        result = HSI_PCA.fit_samples(
            samples,
            variance_threshold=target_variance,
            max_components=max_components
        )

        (
            pca,
            pixels,
            transformed,
            selected_components,
            cumulative_variance
        ) = result

        HSI_PCA.plot_explained_variance(
            cumulative_variance,
            selected_components,
            dataset_name,
            output_dir
        )

        HSI_PCA.plot_pca_scatter(
            transformed,
            dataset_name,
            output_dir
        )

        return result

    # ======================================================
    # EXPLAINED VARIANCE
    # ======================================================

    @staticmethod
    def explained_variance(pca):

        variance = (
            pca.explained_variance_ratio_
        )

        cumulative = np.cumsum(
            variance
        )

        return variance, cumulative

    # ======================================================
    # EXPLAINED VARIANCE GRAPH
    # ======================================================

    @staticmethod
    def plot_explained_variance(
        cumulative_variance,
        selected_components,
        dataset_name,
        output_dir=OUTPUT_DIR
    ):
        """
        Plot cumulative explained variance.
        """

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        components = np.arange(
            1,
            len(cumulative_variance) + 1
        )

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            components,
            cumulative_variance,
            marker="o",
            markersize=3
        )

        plt.axhline(
            0.95,
            linestyle="--",
            label="95% variance"
        )

        plt.axvline(
            selected_components,
            linestyle="--",
            label=(
                f"Selected components: "
                f"{selected_components}"
            )
        )

        plt.xlabel(
            "Number of Principal Components"
        )

        plt.ylabel(
            "Cumulative Explained Variance"
        )

        plt.title(
            f"{dataset_name} - "
            "PCA Explained Variance"
        )

        plt.ylim(
            0,
            1.05
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        filename = (
            f"{dataset_name.lower()}"
            "_pca_explained_variance.png"
        )

        output_path = (
            output_dir / filename
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        print(
            "\nSaved:",
            output_path
        )

    # ======================================================
    # PC1 VS PC2
    # ======================================================

    @staticmethod
    def plot_pca_scatter(
        transformed,
        dataset_name,
        output_dir=OUTPUT_DIR
    ):
        """
        Plot PC1 against PC2.
        """

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if transformed.shape[1] < 2:

            print(
                "Not enough components "
                "for PC1 vs PC2 plot."
            )

            return

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            transformed[:, 0],
            transformed[:, 1],
            s=10,
            alpha=0.6
        )

        plt.xlabel(
            "Principal Component 1"
        )

        plt.ylabel(
            "Principal Component 2"
        )

        plt.title(
            f"{dataset_name} - "
            "PCA Projection: PC1 vs PC2"
        )

        plt.grid(True)

        plt.tight_layout()

        filename = (
            f"{dataset_name.lower()}"
            "_pca_pc1_pc2.png"
        )

        output_path = (
            output_dir / filename
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        print(
            "Saved:",
            output_path
        )