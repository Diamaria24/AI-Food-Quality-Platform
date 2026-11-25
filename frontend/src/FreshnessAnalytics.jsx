import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  Activity,
  Apple,
  CheckCircle2,
  AlertTriangle,
  CalendarDays,
} from "lucide-react";

import { authenticatedFetch } from "./authSession";
import { calculateFreshnessAnalytics } from "./freshnessAnalytics";
import "./FreshnessAnalytics.css";

const API_URL = "http://127.0.0.1:8000";
const PAGE_SIZE = 100;

async function loadAllHistory(signal) {
  const records = [];
  let offset = 0;

  while (true) {
    const response = await authenticatedFetch(
      `${API_URL}/history?limit=${PAGE_SIZE}&offset=${offset}`,
      { signal }
    );

    if (!response.ok) {
      throw new Error(
        `Unable to load analysis history (HTTP ${response.status}).`
      );
    }

    const page = await response.json();

    if (!Array.isArray(page)) {
      throw new Error("Unexpected history response.");
    }

    records.push(...page);

    if (page.length < PAGE_SIZE) break;

    offset += PAGE_SIZE;
  }

  return records;
}

export default function FreshnessAnalytics({ onBack }) {
  const [records, setRecords] = useState([]);
  const [period, setPeriod] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    async function load() {
      try {
        const data = await loadAllHistory(controller.signal);

        if (!controller.signal.aborted) {
          setRecords(data);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(err.message || "Unable to load analytics.");
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    load();

    return () => controller.abort();
  }, []);

  const analytics = useMemo(
    () => calculateFreshnessAnalytics(records, period),
    [records, period]
  );

  const maxFruitCount = Math.max(
    1,
    ...analytics.byFruit.map((item) => item.total)
  );

  const maxDailyCount = Math.max(
    1,
    ...analytics.timeline.map((item) => item.count)
  );

  return (
    <main className="fa-page">
      <div className="fa-container">
        <button
          type="button"
          className="fa-back"
          onClick={onBack}
        >
          <ArrowLeft size={18} />
          Back to Dashboard
        </button>

        <header className="fa-header">
          <span className="fa-eyebrow">
            SPECTRAGUARD ANALYTICS
          </span>
          <h1>RGB Freshness Analytics</h1>
          <p>
            Explore your saved Fresh and Rotten classifications
            across Apple, Banana, Strawberry and Tomato.
          </p>
        </header>

        <section className="fa-panel">
          <div className="fa-toolbar">
            <h2>Freshness Overview</h2>

            <div className="fa-filters">
              {[
                ["7d", "Last 7 Days"],
                ["30d", "Last 30 Days"],
                ["all", "All Time"],
              ].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  className={
                    period === value
                      ? "fa-filter active"
                      : "fa-filter"
                  }
                  aria-pressed={period === value}
                  onClick={() => setPeriod(value)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {loading ? (
            <p className="fa-message">
              Loading your freshness history...
            </p>
          ) : error ? (
            <p className="fa-message fa-error" role="alert">
              {error}
            </p>
          ) : (
            <>
              <div className="fa-stats">
                <article className="fa-stat">
                  <Activity size={23} />
                  <span>Total Analyses</span>
                  <strong>{analytics.total}</strong>
                </article>

                <article className="fa-stat">
                  <CheckCircle2 size={23} />
                  <span>Fresh Predictions</span>
                  <strong>{analytics.fresh}</strong>
                </article>

                <article className="fa-stat">
                  <AlertTriangle size={23} />
                  <span>Rotten Predictions</span>
                  <strong>{analytics.rotten}</strong>
                </article>
              </div>

              {analytics.total === 0 ? (
                <p className="fa-message">
                  No RGB Freshness analyses found for
                  this date range.
                </p>
              ) : (
                <>
                  <div className="fa-chart-grid">
                    <section className="fa-chart-card">
                      <h3>
                        <Apple size={20} />
                        Fruit-wise Classification
                      </h3>

                      <div className="fa-legend">
                        <span>
                          <i className="fa-dot fresh" />
                          Fresh
                        </span>
                        <span>
                          <i className="fa-dot rotten" />
                          Rotten
                        </span>
                      </div>

                      {analytics.byFruit.map((item) => (
                        <div className="fa-fruit-row" key={item.food}>
                          <div className="fa-row-heading">
                            <strong>{item.food}</strong>
                            <span>{item.total} analyses</span>
                          </div>

                          <div className="fa-bar-track">
                            <div
                              className="fa-bar fresh"
                              style={{
                                width: `${
                                  (item.fresh / maxFruitCount) * 100
                                }%`,
                              }}
                              title={`Fresh: ${item.fresh}`}
                            />
                            <div
                              className="fa-bar rotten"
                              style={{
                                width: `${
                                  (item.rotten / maxFruitCount) * 100
                                }%`,
                              }}
                              title={`Rotten: ${item.rotten}`}
                            />
                          </div>

                          <small>
                            Fresh: {item.fresh} ? Rotten: {item.rotten}
                          </small>
                        </div>
                      ))}
                    </section>

                    <section className="fa-chart-card">
                      <h3>
                        <CalendarDays size={20} />
                        Analysis Activity
                      </h3>

                      <p className="fa-chart-description">
                        Number of saved analyses by date (UTC).
                      </p>

                      <div className="fa-activity-list">
                        {analytics.timeline.map((item) => (
                          <div className="fa-activity-row" key={item.date}>
                            <span>{item.date}</span>
                            <div className="fa-activity-track">
                              <div
                                className="fa-activity-fill"
                                style={{
                                  width: `${
                                    (item.count / maxDailyCount) * 100
                                  }%`,
                                }}
                              />
                            </div>
                            <strong>{item.count}</strong>
                          </div>
                        ))}
                      </div>
                    </section>
                  </div>

                  <section className="fa-chart-card fa-recent">
                    <h3>Recent Freshness Analyses</h3>

                    <div className="fa-recent-list">
                      {analytics.recent.map((item) => (
                        <div className="fa-recent-row" key={item.id}>
                          <div>
                            <strong>{item.food}</strong>
                            <span>
                              {item.createdAt
                                ? item.createdAt.toLocaleString()
                                : "Date unavailable"}
                            </span>
                          </div>
                          <span
                            className={
                              item.freshness === "Fresh"
                                ? "fa-outcome fresh"
                                : "fa-outcome rotten"
                            }
                          >
                            {item.freshness}
                          </span>
                        </div>
                      ))}
                    </div>
                  </section>
                </>
              )}

              <p className="fa-disclaimer">
                These statistics describe saved image classifications,
                not changes in the freshness of an individual fruit.
                Predictions are experimental and do not certify food safety.
              </p>
            </>
          )}
        </section>
      </div>
    </main>
  );
}
