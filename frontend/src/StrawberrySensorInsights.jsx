import React from "react";
import "./PapayaSpectralInsights.css";

export default function StrawberrySensorInsights({ type, result }) {
  const format = (value, digits = 2) => {
    const number = Number(value);
    return value !== null &&
      value !== undefined &&
      Number.isFinite(number)
      ? number.toFixed(digits)
      : "N/A";
  };

  const sections = {
    experiments: {
      title: "Understand the Experiment Comparison",
      simple: "Experiment 1 recorded 95 anomalies among 127,588 observations, while Experiment 2 recorded 2,405 anomalies among 122,388 observations. The second experiment had a higher proportion of observations flagged as unusual.",
      technical: "The reported anomaly rates are approximately 0.074% and 1.965%, respectively. These are descriptive results from recorded sensor experiments. Differences may arise from sensor conditions, measurement distributions, preprocessing, or anomaly-detection settings.",
      limitation: "The two anomaly percentages cannot independently establish differences in strawberry freshness, spoilage, or shelf life."
    },
    pca: {
      title: "Understand the PCA Sensor Trajectory",
      simple: "This graph shows how sensor observations are distributed when many original measurements are summarized using two principal components.",
      technical: "The horizontal and vertical axes represent PC1 and PC2. Principal Component Analysis constructs directions that capture variation in the standardized sensor measurements. Nearby observations have similar projected scores; separated observations differ in this reduced representation.",
      limitation: "PCA separation is not proof of freshness classes, contamination, or spoilage. The graph represents research dataset observations, not necessarily the current uploaded CSV."
    },
    condition: {
      title: "Understand the Sensor Condition Trend",
      simple: "This graph illustrates how recorded sensor conditions changed during the research experiment. Rising, falling, or fluctuating patterns indicate changes in the plotted measurements.",
      technical: "The interpretation of individual curves depends on the specific sensor variables, units, and horizontal-axis labels shown in the figure. A trend may reflect changing environmental conditions, sensor response, experimental stages, or measurement drift.",
      limitation: "Without independently verified food-condition labels, sensor trends cannot establish strawberry freshness or remaining shelf life. This is a research visualization."
    },
    temporal: {
      title: "Understand the Temporal PCA Trend",
      simple: "This graph shows how the first principal-component score changes across the recorded observations.",
      technical: "PC1 is a weighted combination of standardized sensor features that captures the largest direction of variation in the analyzed dataset. Positive and negative scores indicate positions relative to the PCA reference, not good or bad food quality. Changes in PC1 show changes in the combined sensor measurement pattern.",
      limitation: "Observation order is not necessarily elapsed time unless the dataset explicitly records chronological sampling. PC1 is not a freshness scale, and this research graph is not generated from the current upload."
    }
  };

  if (type === "results") {
    if (!result) return null;

    const total = Number(result.total_observations || 0);
    const normal = Number(result.normal_observations || 0);
    const anomalies = Number(result.anomaly_observations || 0);

    return (
      <details className="papaya-spectral-insights">
        <summary>Understand These Sensor Results</summary>

        <div style={{ marginTop: "16px" }}>
          <h4>Simple explanation</h4>
          <p>
            The system analyzed <strong>{total.toLocaleString()}</strong>
            {" "}sensor observations. Of these,
            {" "}<strong>{normal.toLocaleString()}</strong> were not
            flagged as anomalies, while
            {" "}<strong>{anomalies.toLocaleString()}</strong> were
            flagged as unusual.
          </p>

          <h4>Anomaly percentage</h4>
          <p>
            <strong>{format(result.anomaly_percentage)}%</strong>
            {" "}of the uploaded observations were flagged.
            This is the proportion of unusual observations
            detected by the model, not a spoilage percentage.
          </p>

          <h4>Mean anomaly score</h4>
          <p>
            The reported mean anomaly score is
            {" "}<strong>{format(result.mean_anomaly_score, 4)}</strong>.
            It summarizes model scoring across the observations.
            The direction and scale of this score depend on the
            exact Isolation Forest scoring function used by the
            backend, so the value should not be interpreted as
            a probability without verifying that implementation.
          </p>

          <h4>Technical explanation</h4>
          <p>
            The processing pipeline uses the trained
            StandardScaler, Principal Component Analysis (PCA),
            and Isolation Forest. StandardScaler transforms the
            original sensor measurements using learned scaling
            parameters. PCA reduces the feature dimensions,
            and Isolation Forest identifies observations whose
            patterns differ from the learned distribution.
          </p>

          <h4>Research limitations</h4>
          <p>
            <strong>Important:</strong> Anomaly flags indicate
            statistical unusualness, not independently verified
            spoilage, contamination, freshness, or shelf life.
            Results depend on the training data, preprocessing,
            and anomaly-detection settings.
          </p>
        </div>
      </details>
    );
  }

  const info = sections[type];
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
        <p><strong>Important:</strong> {info.limitation}</p>
      </div>
    </details>
  );
}
