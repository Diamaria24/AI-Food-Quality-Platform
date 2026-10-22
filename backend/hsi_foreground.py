
import numpy as np


def select_foreground_spectra(
    spectra,
    threshold=0.05,
):
    """
    Experimental foreground selection using
    mean spectral intensity.

    Parameters
    ----------
    spectra : ndarray
        Shape (pixels, spectral_bands).
        Expected to be normalized reflectance.

    threshold : float
        Minimum mean spectral intensity.

    Returns
    -------
    selected_spectra : ndarray
        Spectra passing the foreground threshold.

    mask : ndarray
        Boolean mask for the original input pixels.
    """

    spectra = np.asarray(
        spectra,
        dtype=np.float32,
    )

    if spectra.ndim != 2:
        raise ValueError(
            "Expected a 2D spectral matrix."
        )

    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(
            "Threshold must be finite and non-negative."
        )

    finite_mask = np.isfinite(spectra).all(axis=1)

    mean_intensity = np.zeros(
        len(spectra),
        dtype=np.float32,
    )

    if np.any(finite_mask):
        mean_intensity[finite_mask] = (
            spectra[finite_mask].mean(axis=1)
        )

    foreground_mask = (
        finite_mask
        & (mean_intensity > threshold)
    )

    return spectra[foreground_mask], foreground_mask
