import React from "react";
import "./PapayaSpectralInsights.css";

export default function TomatoSpectralInsights({ type, result }) {
  const spectrum = type === "csv"
    ? result?.curve
    : result?.mean_spectrum;

  const values = Array.isArray(spectrum)
    ? spectrum.map((point) =>
        Number(type === "csv" ? point.normalized_signal : point.mean)
      ).filter(Number.isFinite)
    : [];

  const min = values.length ? Math.min(...values) : null;
  const max = values.length ? Math.max(...values) : null;

  const correlation = Number(result?.reference_correlation);
  const hasCorrelation =
    result?.reference_correlation !== null &&
    result?.reference_correlation !== undefined &&
    Number.isFinite(correlation);

  const explanations = {
    csv: {
      title: "Understand This Graph",
      simple: "The blue curve represents your uploaded Tomato spectrum. The orange dashed curve represents the healthy reference. Similar shapes indicate similar spectral patterns, but do not prove that the sample is healthy.",
      technical: "Both curves are independently min-max normalized. Their vertical values describe relative spectral signal rather than calibrated reflectance. The horizontal axis represents wavelength in nanometers. Peaks and troughs may reflect differences in optical response, but identifying their physical causes requires validated calibration and experimental evidence.",
      limitation: "Reference correlation is a similarity measure, not a probability of freshness, disease, or food safety."
    },
    matlab: {
      title: "Understand This Graph",
      simple: "This curve shows how the average recorded signal changes across the Tomato hyperspectral cube's 550 spectral bands.",
      technical: "The mean spectral signature summarizes measurements across 39,200 image pixels. The horizontal axis represents spectral band index, not wavelength. The normalized vertical values describe relative signal variation between bands.",
      limitation: "This analysis does not perform fungal classification. Band indices cannot be converted to wavelengths without verified sensor metadata."
    },
    signatures: {
      title: "Understand These Spectral Signatures",
      simple: "The curves compare spectral patterns from five Tomato research classes. Their shapes help illustrate how the recorded responses differ between classes.",
      technical: "Smoothed, normalized class signatures allow comparison of relative spectral patterns. Differences may be influenced by sample composition, measurement conditions, preprocessing, and within-class variability.",
      limitation: "Visible separation between curves does not independently establish classification accuracy or disease diagnosis."
    },
    pca: {
      title: "Understand This PCA Graph",
      simple: "PCA helps display complex Tomato spectral data using fewer dimensions. Points that appear closer together have more similar representations in the plotted components.",
      technical: "Principal Component Analysis projects spectral observations onto orthogonal directions of variance. PC1 captures the greatest variance, followed by PC2. Separation or overlap in the plot is exploratory and depends on the data and preprocessing.",
      limitation: "PCA separation alone does not establish predictive accuracy, and principal components do not automatically identify biological causes."
    },
    variance: {
      title: "Understand This Variance Graph",
      simple: "This graph highlights spectral positions where the Tomato research classes differ more or less from one another.",
      technical: "Between-class spectral variance summarizes variation among class-level spectral responses at each plotted band. Larger values indicate greater between-class differences in the analyzed dataset, not necessarily greater predictive importance.",
      limitation: "High between-class variance does not prove that a band is a reliable disease marker or that a classifier will generalize to new samples."
    }
  };

  const info = explanations[type];
  if (!info) return null;

  return (
    <details className="papaya-spectral-insights">
      <summary>{info.title}</summary>
      <div style={{ marginTop: "16px" }}>
        <h4>Simple explanation</h4>
        <p>{info.simple}</p>

        {(type === "csv" || type === "matlab") &&
          min !== null && max !== null && (
          <>
            <h4>Interpretation of your result</h4>
            <p>
              The plotted mean spectral signal ranges from{" "}
              <strong>{min.toFixed(4)}</strong> to{" "}
              <strong>{max.toFixed(4)}</strong>.
              These values describe the variation across
              the displayed spectral bands.
            </p>
          </>
        )}

        {type === "csv" && result && (
          <>
            <p>
              <strong>Peak wavelength:</strong>{" "}
              {Number(result.peak_wavelength_nm).toFixed(2)} nm.
              This is the wavelength associated with the
              highest recorded sample signal.
            </p>
            <p>
              <strong>Healthy reference correlation:</strong>{" "}
              {hasCorrelation ? correlation.toFixed(4) : "Not available"}.
              {hasCorrelation
                ? " This measures similarity between the sample and reference spectral patterns."
                : " A reference similarity value was not available."}
            </p>
          </>
        )}

        <h4>Technical explanation</h4>
        <p>{info.technical}</p>

        <h4>Research limitations</h4>
        <p><strong>Important:</strong> {info.limitation}</p>
      </div>
    </details>
  );
}
