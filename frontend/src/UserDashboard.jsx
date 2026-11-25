import { useEffect, useState } from "react";
import { authenticatedFetch } from "./authSession";
import "./UserDashboard.css";
import {
  ScanLine,
  Microscope,
  Waves,
  ChartNoAxesCombined,
  History,
  BarChart3,
  ArrowRight,
} from "lucide-react";

const modules = [
  {
    id: "rgb-freshness",
    icon: ScanLine,
    title: "RGB Freshness Analysis",
    description: "Classify food freshness using RGB photographs.",
    label: "RGB Module",
  },
  {
    id: "rgb-mold",
    icon: Microscope,
    title: "RGB Mold Detection",
    description: "Analyze microscope-assisted images for mold.",
    label: "RGB Module",
  },
  {
    id: "hsi",
    icon: Waves,
    title: "Hyperspectral Analysis",
    description: "Explore food quality using spectral data and ML models.",
    label: "HSI Module",
  },
  {
    id: "sensor",
    icon: ChartNoAxesCombined,
    title: "Strawberry Sensor Analysis",
    description: "Analyze recorded multispectral sensor measurements.",
    label: "Sensor Module",
  },
];

export default function UserDashboard({
  user,
  onOpenModule,
  onOpenHistory,
  onOpenFreshnessAnalytics,
}) {
  const displayName = user?.full_name?.trim() || "Researcher";
  const firstName = displayName.split(/\s+/)[0];

  const [analyses, setAnalyses] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function loadHistory() {
      try {
        const response = await authenticatedFetch(
          "http://127.0.0.1:8000/history?limit=5",
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(
            `Unable to load analysis history (${response.status}).`
          );
        }

        const records = await response.json();

        if (!controller.signal.aborted) {
          setAnalyses(records);
          setHistoryError("");
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setHistoryError(
            error.message || "Unable to load analysis history."
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setHistoryLoading(false);
        }
      }
    }

    loadHistory();

    return () => controller.abort();
  }, [user?.id]);

  function formatAnalysisDate(value) {
    if (!value) return "Date unavailable";

    // SQLite stores the current backend timestamps without a
    // timezone suffix. They originate from UTC.
    const normalized =
      /(?:Z|[+-]\d{2}:\d{2})$/i.test(value)
        ? value
        : `${value}Z`;

    const date = new Date(normalized);

    return Number.isNaN(date.getTime())
      ? "Date unavailable"
      : date.toLocaleString();
  }

  return (
    <div className="sg-dashboard">
      <div className="sg-dashboard-inner">
        <section className="sg-dashboard-welcome">
          <div>
            <span className="sg-dashboard-eyebrow">
              YOUR RESEARCH WORKSPACE
            </span>
            <h1>Welcome back, {firstName}!</h1>
            <p>
              Explore SpectraGuard's AI-powered food quality
              analysis tools from your personal dashboard.
            </p>
          </div>
          <div className="sg-dashboard-avatar" aria-hidden="true">
            {firstName.charAt(0).toUpperCase()}
          </div>
        </section>

        <section className="sg-dashboard-section">
          <div className="sg-dashboard-heading">
            <div>
              <h2>Analysis Modules</h2>
              <p>Select a module to begin a new analysis.</p>
            </div>
          </div>

          <div className="sg-dashboard-modules">
            {modules.map((module) => (
              <button
                key={module.id}
                type="button"
                className="sg-dashboard-module"
                onClick={() => onOpenModule(module.id)}
              >
                <span className="sg-dashboard-module-icon">
                  <module.icon size={25} strokeWidth={1.8} />
                </span>
                <span className="sg-dashboard-module-label">
                  {module.label}
                </span>
                <strong>{module.title}</strong>
                <span className="sg-dashboard-module-description">
                  {module.description}
                </span>
                <span className="sg-dashboard-module-link">
                  Open Analysis <ArrowRight size={15} />
                </span>
              </button>
            ))}
          </div>
        </section>

        <section className="sg-dashboard-section">
          <div className="sg-dashboard-heading">
            <div>
              <h2>Freshness Insights</h2>
              <p>
                Explore your saved RGB freshness predictions
                and classification trends.
              </p>
            </div>
          </div>

          <button
            type="button"
            className="sg-dashboard-analytics-card"
            onClick={onOpenFreshnessAnalytics}
          >
            <span className="sg-dashboard-analytics-icon">
              <BarChart3 size={28} strokeWidth={1.8} />
            </span>

            <span className="sg-dashboard-analytics-content">
              <strong>RGB Freshness Analytics</strong>
              <span>
                View Fresh and Rotten counts, fruit-wise
                comparisons, activity trends and recent analyses.
              </span>
            </span>

            <span className="sg-dashboard-analytics-link">
              View Analytics <ArrowRight size={17} />
            </span>
          </button>
        </section>

        <div className="sg-dashboard-bottom">
          <section className="sg-dashboard-panel">
            <div className="sg-dashboard-heading">
              <div>
                <h2>Recent Analyses</h2>
                <p>Your latest analysis activity.</p>
              </div>
              <button
                type="button"
                className="sg-dashboard-view-history"
                onClick={() => onOpenHistory()}
              >
                View All History <ArrowRight size={16} />
              </button>
            </div>

            {historyLoading ? (
              <div className="sg-dashboard-empty">
                <p>Loading your recent analyses...</p>
              </div>
            ) : historyError ? (
              <div className="sg-dashboard-empty" role="alert">
                <h3>History unavailable</h3>
                <p>{historyError}</p>
              </div>
            ) : analyses.length === 0 ? (
              <div className="sg-dashboard-empty">
                <span className="sg-dashboard-empty-icon">
                  <History size={38} strokeWidth={1.5} />
                </span>
                <h3>No saved analyses yet</h3>
                <p>
                  Your completed analyses will appear here.
                </p>
              </div>
            ) : (
              <div className="sg-dashboard-history-list">
                {analyses.map((analysis) => (
                  <div
                    key={analysis.id}
                    className="sg-dashboard-history-item"
                  >
                    <div className="sg-dashboard-history-details">
                      <strong>
                        {analysis.food_name} -{" "}
                        {analysis.prediction
                          .replace(`${analysis.food_name}_`, "")
                          .replaceAll("_", " ")}
                      </strong>
                      <span>
                        {analysis.module.toUpperCase()} |{" "}
                        {analysis.analysis_type.replaceAll("_", " ")}
                      </span>
                      <small>
                        {formatAnalysisDate(analysis.created_at)}
                      </small>
                    </div>
                    <span className="sg-dashboard-history-status">
                      Completed
                    </span>
                  </div>
                ))}
              </div>
            )}
          </section>

          <section className="sg-dashboard-panel">
            <div className="sg-dashboard-heading">
              <div>
                <h2>My Profile</h2>
                <p>Your SpectraGuard account information.</p>
              </div>
            </div>

            <div className="sg-dashboard-profile">
              <div className="sg-dashboard-profile-avatar">
                {firstName.charAt(0).toUpperCase()}
              </div>
              <strong>{displayName}</strong>
              <span>{user?.email || "Email unavailable"}</span>
            </div>

            <div className="sg-dashboard-profile-note">
              Your account is authenticated.
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
