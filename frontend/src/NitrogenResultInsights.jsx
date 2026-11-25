import React from "react";
import "./PapayaSpectralInsights.css";

export default function NitrogenResultInsights({
  data,
  food,
  mode
}) {
  if (!data) return null;

  const name = food === "cabbage" ? "Cabbage" : "Eggplant";
  const percentages = data.predicted_percentages || {};
  const counts = data.predicted_counts || {};

  const categories = ["Low", "Medium", "High"];

  const dominant = categories.reduce((best, current) => {
    return Number(counts[current] || 0) >
      Number(counts[best] || 0)
      ? current
      : best;
  }, "Low");

  const formatPercentage = (category) => {
    const value = Number(percentages[category]);
    return Number.isFinite(value)
      ? `${value.toFixed(2)}%`
      : "Not available";
  };

  return (
    <details className="papaya-spectral-insights">
      <summary>Understand These Results</summary>

      <div style={{ marginTop: "16px" }}>
        <h4>Simple explanation</h4>
        <p>
          The experimental model analyzed{" "}
          <strong>
            {Number(data.pixels_analyzed || 0).toLocaleString()}
          </strong>{" "}
          selected pixels from the {name.toLowerCase()} hyperspectral
          cube. Most analyzed pixels were assigned to the{" "}
          <strong>{dominant}</strong> category.
        </p>

        <h4>Category distribution</h4>

        {categories.map((category) => (
          <p key={category}>
            <strong>{category}:</strong>{" "}
            {formatPercentage(category)} of analyzed pixels
            {" ("}
            {Number(counts[category] || 0).toLocaleString()}
            {" pixels)."}
          </p>
        ))}

        <h4>Technical explanation</h4>
        <p>
          The analysis uses Principal Component Analysis (PCA)
          with 30 components to reduce spectral dimensionality,
          followed by a Random Forest classifier. Each selected
          pixel receives an experimental Low, Medium, or High
          category assignment based on learned spectral patterns.
        </p>

        <p>
          The percentages represent the distribution of model
          predictions among analyzed pixels. They are not
          measured nitrogen concentrations or model accuracy
          scores.
        </p>

        <h4>How were the pixels selected?</h4>
        <p>
          {mode === "upload"
            ? "For this uploaded cube, foreground pixels were selected automatically without using ground-truth labels. Spatially distributed sampling was applied."
            : "For this local developer dataset, ground-truth labels were used to select eligible pixels. Spatially distributed sampling was applied."}
        </p>

        <h4>Research limitations</h4>
        <p>
          <strong>Important:</strong> This is experimental
          pixel-level classification. The model has not been
          independently validated on new specimens. These
          predictions may not represent the entire {name.toLowerCase()},
          and they do not establish actual nitrogen content,
          freshness, spoilage, or food safety.
        </p>
      </div>
    </details>
  );
}
