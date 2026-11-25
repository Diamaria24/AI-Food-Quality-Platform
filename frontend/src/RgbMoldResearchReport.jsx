import "./RgbMoldResearchReport.css";

export default function RgbMoldResearchReport({ report }) {
  if (!report) return null;

  const overview = report.analysis_overview || {};
  const summary = report.prediction_summary || {};
  const interpretation = report.scientific_interpretation || {};
  const methodology = report.methodology || {};
  const education = report.educational_background || {};

  const detected = summary.predicted_class === "Mold";

  return (
    <article className="mold-research-report">
      <header className="mrr-header">
        <span className="mrr-eyebrow">
          SPECTRAGUARD ? RESEARCH ANALYSIS
        </span>
        <h2>RGB Mold Detection Research Report</h2>
        <p>
          Experimental microscope-assisted image analysis
          for research and educational use.
        </p>
      </header>

      <section className="mrr-section">
        <h3>1. Analysis Overview</h3>
        <div className="mrr-grid">
          <div><span>Model</span><strong>{overview.model}</strong></div>
          <div><span>Input</span><strong>{overview.image_type}</strong></div>
          <div>
            <span>Supported foods</span>
            <strong>{overview.supported_foods?.join(", ")}</strong>
          </div>
          <div>
            <span>Food identification</span>
            <strong>{overview.food_identification}</strong>
          </div>
        </div>
      </section>

      <section className="mrr-section">
        <h3>2. Model Prediction</h3>
        <div className={`mrr-prediction ${detected ? "detected" : "not-detected"}`}>
          <span>Predicted Classification</span>
          <strong>{summary.predicted_class}</strong>
        </div>
        <div className="mrr-grid">
          <div>
            <span>Raw Mold Model Score</span>
            <strong>{summary.mold_score_percentage}%</strong>
          </div>
          <div>
            <span>Classification Threshold</span>
            <strong>{summary.classification_threshold}</strong>
          </div>
        </div>
        <p className="mrr-note">{summary.score_description}</p>
      </section>

      <section className="mrr-section">
        <h3>3. Scientific Interpretation</h3>
        <p>{interpretation.summary}</p>
        <p>{interpretation.scope}</p>
      </section>

      <section className="mrr-section">
        <h3>4. Model Methodology</h3>
        <div className="mrr-grid">
          <div>
            <span>Architecture</span>
            <strong>{methodology.model_architecture}</strong>
          </div>
          <div>
            <span>Input Dimensions</span>
            <strong>{methodology.input_dimensions}</strong>
          </div>
          <div>
            <span>Color Channels</span>
            <strong>{methodology.color_channels}</strong>
          </div>
        </div>
        <ol>
          {(methodology.preprocessing || []).map((step, index) => (
            <li key={index}>{step}</li>
          ))}
        </ol>
        <p className="mrr-note">{methodology.note}</p>
      </section>

      <section className="mrr-section">
        <h3>5. Educational Background</h3>
        <p>{education.title}</p>
        <div className="mrr-topics">
          {(education.topics || []).map((topic) => (
            <div key={topic.term}>
              <h4>{topic.term}</h4>
              <p>{topic.explanation}</p>
            </div>
          ))}
        </div>
        <p className="mrr-note">{education.important_note}</p>
      </section>

      <section className="mrr-section">
        <h3>6. Research Limitations</h3>
        <ul>
          {(report.limitations || []).map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="mrr-section">
        <h3>7. Research Recommendations</h3>
        <ul>
          {(report.research_recommendations || []).map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="mrr-section">
        <h3>8. Research Conclusion</h3>
        <p>{report.conclusion}</p>
      </section>

      <footer className="mrr-disclaimer">
        <strong>Research & Educational Disclaimer</strong>
        <p>{report.disclaimer}</p>
      </footer>
    </article>
  );
}
