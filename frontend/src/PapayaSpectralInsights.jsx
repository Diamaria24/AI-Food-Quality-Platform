import React from "react";
import "./PapayaSpectralInsights.css";

export default function PapayaSpectralInsights({
  result,
  type
}) {
  if (!result) return null;

  const variance = Array.isArray(
    result.pca_explained_variance_percent
  )
    ? result.pca_explained_variance_percent
    : [];

  const totalVariance = variance.reduce(
    (sum, value) => sum + value, 0
  );

  const spectrum = Array.isArray(result.mean_spectrum)
    ? result.mean_spectrum
    : [];

  const validSpectrum = spectrum.filter(
    (point) => Number.isFinite(Number(point.mean))
  );

  const minimum = validSpectrum.length
    ? Math.min(...validSpectrum.map((p) => Number(p.mean)))
    : null;

  const maximum = validSpectrum.length
    ? Math.max(...validSpectrum.map((p) => Number(p.mean)))
    : null;

  return (
    <details className="papaya-spectral-insights"
      style={{
        marginTop: "18px",
        padding: "16px",
        border: "1px solid #cbd5e1",
        borderRadius: "12px"
      }}
    >
      <summary
        style={{
          cursor: "pointer",
          fontWeight: 700,
          fontSize: "15px"
        }}
      >
        Understand This {type === "pca" ? "PCA Result" : "Graph"}
      </summary>

      {type === "spectrum" && (
        <div style={{ marginTop: "14px", lineHeight: 1.7 }}>
          <h4>What does this graph show?</h4>
          <p>
            This graph represents the average spectral signal
            measured across the Papaya hyperspectral image.
            Each point corresponds to a spectral band.
          </p>

          <h4>How do I read it?</h4>
          <p>
            The horizontal axis represents spectral band index.
            The vertical position represents the average measured
            signal. Higher points indicate a higher mean signal
            at that band.
          </p>

          <h4>Interpretation of your result</h4>
          {minimum !== null && maximum !== null ? (
            <p>
              Across the plotted spectral bands, the minimum
              mean signal is {minimum.toFixed(6)} and the
              maximum mean signal is {maximum.toFixed(6)}.
              The difference is {(maximum - minimum).toFixed(6)}.
              This indicates how much the average spectral
              response varies across bands.
            </p>
          ) : (
            <p>
              Spectral values are not available for numerical
              interpretation.
            </p>
          )}

          <h4>Technical explanation</h4>
          <p>
            A mean spectral signature summarizes the spectral
            measurements across image pixels. Peaks and
            troughs represent differences in the recorded
            signal between spectral bands. Their physical
            interpretation depends on sensor calibration,
            wavelength metadata, and sample conditions.
          </p>

          <p>
            <strong>Important:</strong> This graph alone cannot
            establish freshness, mold contamination, or food
            safety. Spectral band indices should not be
            interpreted as wavelengths without verified
            wavelength metadata.
          </p>
        </div>
      )}

      {type === "pca" && (
        <div style={{ marginTop: "14px", lineHeight: 1.7 }}>
          <h4>What is PCA?</h4>
          <p>
            Principal Component Analysis (PCA) reduces a large
            number of correlated spectral measurements into
            a smaller set of components that capture patterns
            of variation in the uploaded sample.
          </p>

          <h4>Understanding PC1, PC2 and PC3</h4>
          <p>
            PC1 captures the greatest amount of variance.
            PC2 captures the greatest remaining variance
            while being orthogonal to PC1. PC3 captures
            further remaining variance.
          </p>

          <h4>Interpretation of your result</h4>
          <p>
            The first {variance.length} principal components
            together explain {totalVariance.toFixed(2)}%
            of the variance in the analyzed Papaya cube.
          </p>

          {variance.map((value, index) => (
            <p key={index}>
              <strong>PC{index + 1}:</strong>{" "}
              {Number(value).toFixed(2)}% of total variance.
            </p>
          ))}

          <h4>What does this percentage mean?</h4>
          <p>
            A higher explained variance percentage means
            that the selected components retain more of
            the variation present in the spectral data.
            It does not indicate classification accuracy,
            freshness, or the probability of mold.
          </p>

          <h4>Research limitation</h4>
          <p>
            This PCA is fitted to the uploaded Papaya cube.
            The resulting components describe variation
            within that particular sample and are not
            equivalent to predictions from a separately
            validated classification model.
          </p>
        </div>
      )}
    </details>
  );
}
