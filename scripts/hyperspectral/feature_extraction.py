"""
feature_extraction.py

Feature extraction utilities for hyperspectral PCA data.

The input is expected to be a 2D PCA matrix:

    (number_of_pixels, number_of_components)

Each row represents one pixel.
Each column represents one principal component.
"""

import numpy as np


class HSIFeatureExtractor:

    @staticmethod
    def validate_input(pca_data):
        """
        Validate PCA feature matrix.

        Parameters
        ----------
        pca_data : numpy.ndarray
            PCA-transformed data with shape:

                (samples, components)

        Returns
        -------
        numpy.ndarray
            Validated float32 matrix.
        """

        pca_data = np.asarray(pca_data, dtype=np.float32)

        if pca_data.ndim != 2:
            raise ValueError(
                "PCA data must be a 2D matrix "
                "(samples, components)."
            )

        if pca_data.shape[0] == 0:
            raise ValueError(
                "PCA data contains no samples."
            )

        if pca_data.shape[1] == 0:
            raise ValueError(
                "PCA data contains no components."
            )

        # Replace invalid values if present.
        pca_data = np.nan_to_num(
            pca_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return pca_data.astype(np.float32)

    @staticmethod
    def statistical_features(pca_data):
        """
        Extract statistical features from PCA components.

        Features are calculated independently for every pixel.

        Features:
            Mean
            Standard deviation
            Minimum
            Maximum
            Median
            Range
            Energy

        Parameters
        ----------
        pca_data : numpy.ndarray
            PCA matrix of shape:

                (samples, components)

        Returns
        -------
        features : numpy.ndarray
            Feature matrix of shape:

                (samples, 7)
        """

        pca_data = HSIFeatureExtractor.validate_input(
            pca_data
        )

        mean = np.mean(
            pca_data,
            axis=1
        )

        std = np.std(
            pca_data,
            axis=1
        )

        minimum = np.min(
            pca_data,
            axis=1
        )

        maximum = np.max(
            pca_data,
            axis=1
        )

        median = np.median(
            pca_data,
            axis=1
        )

        feature_range = maximum - minimum

        energy = np.sum(
            pca_data ** 2,
            axis=1
        )

        features = np.column_stack(
            [
                mean,
                std,
                minimum,
                maximum,
                median,
                feature_range,
                energy
            ]
        )

        return features.astype(np.float32)

    @staticmethod
    def component_features(pca_data):
        """
        Preserve the original PCA components as features.

        This is useful when the classifier needs direct access
        to the PCA scores.

        Parameters
        ----------
        pca_data : numpy.ndarray
            PCA matrix.

        Returns
        -------
        numpy.ndarray
            Copy of PCA matrix.
        """

        pca_data = HSIFeatureExtractor.validate_input(
            pca_data
        )

        return pca_data.copy()

    @staticmethod
    def extract(pca_data, include_pca=True):
        """
        Complete feature extraction pipeline.

        Parameters
        ----------
        pca_data : numpy.ndarray
            PCA matrix with shape:

                (samples, components)

        include_pca : bool
            If True, original PCA components are included
            along with statistical features.

        Returns
        -------
        numpy.ndarray
            Final feature matrix.
        """

        pca_data = HSIFeatureExtractor.validate_input(
            pca_data
        )

        statistical = (
            HSIFeatureExtractor.statistical_features(
                pca_data
            )
        )

        if include_pca:

            features = np.hstack(
                [
                    pca_data,
                    statistical
                ]
            )

        else:

            features = statistical

        return features.astype(np.float32)

    @staticmethod
    def feature_names(number_of_components):
        """
        Generate names for extracted features.

        Parameters
        ----------
        number_of_components : int
            Number of PCA components.

        Returns
        -------
        list
            Feature names.
        """

        pca_names = [
            f"PC{i + 1}"
            for i in range(number_of_components)
        ]

        statistical_names = [
            "mean",
            "std",
            "minimum",
            "maximum",
            "median",
            "range",
            "energy"
        ]

        return pca_names + statistical_names