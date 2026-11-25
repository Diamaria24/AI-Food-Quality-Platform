import { useEffect, useState } from "react";
import { ArrowLeft, History, CalendarDays, Eye, X, Trash2 } from "lucide-react";
import { authenticatedFetch } from "./authSession";
import AnalysisResultReport from "./AnalysisResultReport";
import "./AnalysisHistory.css";

function formatDate(value) {
  if (!value) return "Date unavailable";

  const normalized = /(?:Z|[+-]\\d{2}:\\d{2})$/i.test(value)
    ? value
    : `${value}Z`;

  const date = new Date(normalized);

  return Number.isNaN(date.getTime())
    ? "Date unavailable"
    : date.toLocaleString();
}

function formatPrediction(analysis) {
  const food = analysis.food_name || "Unknown";
  const prediction = analysis.prediction || "Analysis completed";

  if (prediction === "Analysis completed") {
    return prediction;
  }

  const prefix = `${food}_`;

  return prediction
    .replace(new RegExp(`^${prefix}`, "i"), "")
    .replaceAll("_", " ");
}

export default function AnalysisHistory({ onBack }) {
  const [records, setRecords] = useState([]);
  const [activeFilter, setActiveFilter] = useState("all");
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [deletingId, setDeletingId] = useState(null);
  const [deleteError, setDeleteError] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadHistory() {
      try {
        const response = await authenticatedFetch(
          "http://127.0.0.1:8000/history?limit=100",
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(
            `Unable to load analysis history (${response.status}).`
          );
        }

        const data = await response.json();

        if (!controller.signal.aborted) {
          setRecords(data);
          setError("");
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err.message || "Unable to load history.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    loadHistory();

    return () => controller.abort();
  }, []);

  async function handleDeleteAnalysis(analysis) {
    const confirmed = window.confirm(
      `Delete the saved analysis for ${analysis.food_name}?\\n\\n` +
      "This action cannot be undone."
    );

    if (!confirmed) return;

    setDeletingId(analysis.id);
    setDeleteError("");

    try {
      const response = await authenticatedFetch(
        `http://127.0.0.1:8000/history/${analysis.id}`,
        { method: "DELETE" }
      );

      if (!response.ok) {
        throw new Error(
          `Unable to delete analysis (${response.status}).`
        );
      }

      setRecords((current) =>
        current.filter((record) => record.id !== analysis.id)
      );

      setSelectedAnalysis((current) =>
        current?.id === analysis.id ? null : current
      );
    } catch (err) {
      setDeleteError(
        err.message || "Unable to delete the analysis."
      );
    } finally {
      setDeletingId(null);
    }
  }

  const filteredRecords = records.filter((analysis) => {
    if (activeFilter === "all") return true;

    const moduleName = (analysis.module || "").toLowerCase();

    if (activeFilter === "rgb") {
      return moduleName.includes("rgb");
    }

    if (activeFilter === "hyperspectral") {
      return (
        moduleName.includes("hsi") ||
        moduleName.includes("hyper")
      );
    }

    if (activeFilter === "sensor") {
      return (
        moduleName.includes("sensor") ||
        moduleName.includes("strawberry")
      );
    }

    return false;
  });

  return (
    <main className="sg-history-page">
      <div className="sg-history-container">
        <button
          type="button"
          className="sg-history-back"
          onClick={onBack}
        >
          <ArrowLeft size={18} />
          Back to Dashboard
        </button>

        <header className="sg-history-header">
          <span className="sg-history-eyebrow">
            YOUR RESEARCH WORKSPACE
          </span>
          <h1>Analysis History</h1>
          <p>
            Review your previous food quality analyses,
            predictions, and research results.
          </p>
        </header>

        <section className="sg-history-panel">
          {deleteError && (
            <p className="sg-history-delete-error" role="alert">
              {deleteError}
            </p>
          )}
          {loading ? (
            <div className="sg-history-empty">
              <p>Loading your analysis history...</p>
            </div>
          ) : error ? (
            <div className="sg-history-empty" role="alert">
              <h2>Unable to load history</h2>
              <p>{error}</p>
            </div>
          ) : records.length === 0 ? (
            <div className="sg-history-empty">
              <History size={42} strokeWidth={1.5} />
              <h2>No saved analyses yet</h2>
              <p>
                Complete an analysis to see your results here.
              </p>
            </div>
          ) : (
            <>
              <div className="sg-history-list-heading">
                <h2>Saved Analyses</h2>
                <span>{records.length} records displayed</span>
              </div>

              <div
                className="sg-history-filters"
                role="group"
                aria-label="Filter analyses by module"
              >
                {[
                  ["all", "All Analyses"],
                  ["rgb", "RGB"],
                  ["hyperspectral", "Hyperspectral"],
                  ["sensor", "Sensor"],
                ].map(([value, label]) => (
                  <button
                    key={value}
                    type="button"
                    className={
                      activeFilter === value
                        ? "sg-history-filter active"
                        : "sg-history-filter"
                    }
                    aria-pressed={activeFilter === value}
                    onClick={() => setActiveFilter(value)}
                  >
                    {label}
                  </button>
                ))}
              </div>

              {filteredRecords.length === 0 ? (
                <div className="sg-history-filter-empty">
                  No analyses found for this category.
                </div>
              ) : (
              <div className="sg-history-list">
                {filteredRecords.map((analysis) => (
                  <article
                    key={analysis.id}
                    className="sg-history-record"
                  >
                    <div className="sg-history-record-main">
                      <div className="sg-history-record-icon">
                        <History size={22} />
                      </div>

                      <div className="sg-history-record-details">
                        <h3>
                          {analysis.food_name} -{" "}
                          {formatPrediction(analysis)}
                        </h3>

                        <p>
                          {analysis.module.toUpperCase()} |{" "}
                          {analysis.analysis_type.replaceAll("_", " ")}
                        </p>

                        <span className="sg-history-record-date">
                          <CalendarDays size={14} />
                          {formatDate(analysis.created_at)}
                        </span>
                      </div>
                    </div>

                    <div className="sg-history-record-actions">
                      <span className="sg-history-record-status">
                        Completed
                      </span>
                      <button
                        type="button"
                        className="sg-history-view-button"
                        onClick={() => setSelectedAnalysis(analysis)}
                      >
                        <Eye size={16} />
                        View Details
                      </button>

                      <button
                        type="button"
                        className="sg-history-delete-button"
                        disabled={deletingId === analysis.id}
                        onClick={() => handleDeleteAnalysis(analysis)}
                      >
                        <Trash2 size={16} />
                        {deletingId === analysis.id
                          ? "Deleting..."
                          : "Delete"}
                      </button>
                    </div>
                  </article>
                ))}
              </div>
              )}
            </>
          )}
        </section>

        {selectedAnalysis && (
          <section
            className="sg-history-details-panel"
            aria-label="Detailed analysis result"
          >
            <div className="sg-history-details-header">
              <div>
                <span className="sg-history-eyebrow">
                  SAVED ANALYSIS RESULT
                </span>
                <h2>
                  {selectedAnalysis.food_name} -{" "}
                  {formatPrediction(selectedAnalysis)}
                </h2>
              </div>

              <button
                type="button"
                className="sg-history-details-close"
                onClick={() => setSelectedAnalysis(null)}
                aria-label="Close analysis details"
              >
                <X size={20} />
              </button>
            </div>

            <div className="sg-history-details-grid">
              <div>
                <span>Food</span>
                <strong>{selectedAnalysis.food_name}</strong>
              </div>
              <div>
                <span>Module</span>
                <strong>{selectedAnalysis.module}</strong>
              </div>
              <div>
                <span>Analysis Type</span>
                <strong>
                  {selectedAnalysis.analysis_type.replaceAll("_", " ")}
                </strong>
              </div>
              <div>
                <span>Prediction / Outcome</span>
                <strong>{selectedAnalysis.prediction}</strong>
              </div>
              <div>
                <span>Recorded At</span>
                <strong>
                  {formatDate(selectedAnalysis.created_at)}
                </strong>
              </div>
              <div>
                <span>Calibrated Confidence</span>
                <strong>
                  {selectedAnalysis.confidence == null
                    ? "Not available"
                    : `${(selectedAnalysis.confidence * 100).toFixed(2)}%`}
                </strong>
              </div>
            </div>

            <AnalysisResultReport
              analysis={selectedAnalysis}
            />
          </section>
        )}
      </div>
    </main>
  );
}
