import { useState } from "react";
import { ChevronDown, ChevronUp, Info } from "lucide-react";
import "./AnalysisResultReport.css";

function displayText(value) {
  if (value === null || value === undefined || value === "") {
    return "Not available";
  }
  return String(value).replaceAll("_", " ");
}

function percent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return `${(number * 100).toFixed(2)}%`;
}

function Metric({ label, value }) {
  return (
    <div className="sg-report-metric">
      <span>{label}</span>
      <strong>{displayText(value)}</strong>
    </div>
  );
}

export default function AnalysisResultReport({ analysis }) {
  const [showRaw, setShowRaw] = useState(false);

  const data = analysis?.result_data || {};
  const result =
    data.result && typeof data.result === "object"
      ? data.result
      : data;

  const moduleName = (analysis.module || "").toLowerCase();
  const analysisType = (analysis.analysis_type || "").toLowerCase();

  const isRgb = moduleName.includes("rgb");
  const isFreshness = isRgb && analysisType.includes("freshness");

  const modelScore = result.model_score;
  const scoreLabel =
    modelScore !== null && modelScore !== undefined
      ? percent(modelScore)
      : null;

  const prediction =
    result.prediction || analysis.prediction;

  const freshness =
    result.freshness || null;

  const model =
    result.model || null;

  const classScores =
    result.class_scores &&
    typeof result.class_scores === "object" &&
    !Array.isArray(result.class_scores)
      ? Object.entries(result.class_scores)
          .filter(([, score]) =>
            Number.isFinite(Number(score)) &&
            Number(score) >= 0 &&
            Number(score) <= 1
          )
          .sort((a, b) => Number(b[1]) - Number(a[1]))
      : [];

  return (
    <div className="sg-report">
      <div className="sg-report-heading">
        <div>
          <span className="sg-report-eyebrow">
            SAVED ANALYSIS REPORT
          </span>
          <h3>{displayText(analysis.food_name)} Analysis Report</h3>
          <p>
            {displayText(analysis.module).toUpperCase()} |{" "}
            {displayText(analysis.analysis_type)}
          </p>
        </div>
        <span className="sg-report-completed">Completed</span>
      </div>

      <div className="sg-report-metrics">
        <Metric
          label="Food Identified"
          value={analysis.food_name}
        />
        <Metric
          label="Prediction / Outcome"
          value={prediction}
        />

        {freshness && (
          <Metric
            label="Freshness Classification"
            value={freshness}
          />
        )}

        {scoreLabel && (
          <Metric
            label="Top-class Model Score"
            value={scoreLabel}
          />
        )}

        {model && (
          <Metric
            label="AI Model"
            value={model}
          />
        )}

        <Metric
          label="Analysis Type"
          value={analysis.analysis_type}
        />
      </div>

      {classScores.length > 0 && (
        <section className="sg-report-scores">
          <h4>Model Class Scores</h4>
          <p>
            Scores produced by the classification model.
            These are not necessarily calibrated probabilities.
          </p>

          {classScores.map(([label, score]) => (
            <div className="sg-report-score" key={label}>
              <div className="sg-report-score-label">
                <span>{displayText(label)}</span>
                <strong>{percent(score)}</strong>
              </div>

              <div className="sg-report-score-track">
                <div
                  className="sg-report-score-fill"
                  style={{
                    width: `${Number(score) * 100}%`,
                  }}
                />
              </div>
            </div>
          ))}
        </section>
      )}

      {!isFreshness && (
        <div className="sg-report-note">
          <Info size={18} />
          <p>
            This report preserves the outcome of the
            original analysis. Additional spectral,
            anomaly, or research-specific measurements
            are available under View Raw JSON.
          </p>
        </div>
      )}

      {isFreshness && (
        <div className="sg-report-note">
          <Info size={18} />
          <p>
            Experimental RGB freshness classification.
            Model softmax scores are not calibrated
            confidence estimates. Predictions must
            not be used as food-safety certification.
          </p>
        </div>
      )}

      <div className="sg-report-raw-section">
        <button
          type="button"
          className="sg-report-raw-toggle"
          aria-expanded={showRaw}
          onClick={() => setShowRaw(!showRaw)}
        >
          {showRaw ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
          {showRaw ? "Hide Raw JSON" : "View Raw JSON"}
        </button>

        {showRaw && (
          <pre className="sg-report-json">
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
