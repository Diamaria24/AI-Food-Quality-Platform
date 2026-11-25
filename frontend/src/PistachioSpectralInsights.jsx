import React from "react";
import "./PapayaSpectralInsights.css";

export default function PistachioSpectralInsights({
  type,
  result
}) {
  const explanations = {
    prediction: {
      title: "Understand This Prediction",
      simple: "The trained machine-learning model analyzes the uploaded Pistachio hyperspectral cube and produces a classification result based on patterns learned from its training data.",
      technical: "The model processes spectral features and estimates class-associated scores. These outputs depend on the training dataset, preprocessing pipeline, model configuration, and similarity between training and uploaded samples.",
      limitation: "An uncalibrated model probability is not a validated probability of aflatoxin contamination. This result cannot replace laboratory testing or establish regulatory compliance."
    },
    signatures: {
      title: "Understand These Spectral Signatures",
      simple: "The curves compare the average spectral responses of two Pistachio groups defined by an AFB1 threshold in the research dataset.",
      technical: "Mean spectral signatures summarize wavelength-dependent measurements within each threshold group. Differences in curve shape or intensity may reflect associations with sample properties, measurement conditions, and group composition.",
      limitation: "Differences between the curves do not prove that AFB1 directly causes those spectral changes or that individual samples can be classified reliably."
    },
    difference: {
      title: "Understand This Spectral Difference Graph",
      simple: "This graph highlights wavelengths where the two Pistachio threshold groups show larger or smaller differences in their average spectral measurements.",
      technical: "The plotted differences compare group-level spectral responses across wavelengths. Larger magnitudes indicate greater observed separation at those wavelengths in the analyzed dataset.",
      limitation: "Large spectral differences are not automatically reliable biomarkers. Their usefulness requires independent validation."
    },
    correlation: {
      title: "Understand This Correlation Graph",
      simple: "This graph shows how strongly measurements at different wavelengths are statistically associated with AFB1 values in the research dataset.",
      technical: "Correlation describes the direction and strength of a statistical association. Positive and negative values indicate opposite directions of association; values closer to zero indicate weaker linear association.",
      limitation: "Correlation does not establish causation, classification accuracy, or a validated AFB1 concentration estimate for a new sample."
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

        <h4>Technical explanation</h4>
        <p>{info.technical}</p>

        <h4>Research limitations</h4>
        <p>
          <strong>Important:</strong> {info.limitation}
        </p>
      </div>
    </details>
  );
}
