import { useEffect, useState } from "react";
import App from "./App";
import Auth from "./Auth";
import LandingPage from "./LandingPage";
import UserDashboard from "./UserDashboard";
import AnalysisHistory from "./AnalysisHistory";
import FreshnessAnalytics from "./FreshnessAnalytics.jsx";
import {
  clearAccessToken,
  setAccessToken,
} from "./authSession";

export default function ProtectedApp() {
  const [session, setSession] = useState(null);
  const [visitorScreen, setVisitorScreen] = useState("landing");
  const [view, setView] = useState("dashboard");
  const [selectedModule, setSelectedModule] = useState(null);

  useEffect(() => {
    function handleExpiredSession() {
      clearAccessToken();
      setSession(null);
      setView("dashboard");
      setSelectedModule(null);
    }

    window.addEventListener(
      "spectraguard:session-expired",
      handleExpiredSession
    );

    return () => {
      window.removeEventListener(
        "spectraguard:session-expired",
        handleExpiredSession
      );
    };
  }, []);

  function handleLogin({ token, user }) {
    setAccessToken(token);
    setSession({ user });
    setView("dashboard");
    setSelectedModule(null);
  }

  function handleLogout() {
    clearAccessToken();
    setSession(null);
    setView("dashboard");
    setSelectedModule(null);
  }

  function handleOpenModule(moduleId) {
    setSelectedModule(moduleId);
    setView("analysis");
    window.scrollTo(0, 0);
  }

  function handleBackToDashboard() {
    setView("dashboard");
    setSelectedModule(null);
    window.scrollTo(0, 0);
  }

  function handleOpenHistory() {
    setSelectedModule(null);
    setView("history");
    window.scrollTo(0, 0);
  }

  function handleOpenFreshnessAnalytics() {
    setSelectedModule(null);
    setView("freshness-analytics");
    window.scrollTo(0, 0);
  }

  if (!session) {
    if (visitorScreen === "landing") {
      return (
        <LandingPage
          onGetStarted={() => {
            setVisitorScreen("auth");
            window.scrollTo(0, 0);
          }}
        />
      );
    }

    return <Auth onLogin={handleLogin} />;
  }

  let initialModule = "hsi";
  let initialRgbMode = "freshness";

  if (selectedModule === "rgb-freshness") {
    initialModule = "rgb";
    initialRgbMode = "freshness";
  } else if (selectedModule === "rgb-mold") {
    initialModule = "rgb";
    initialRgbMode = "mold";
  } else if (selectedModule === "sensor") {
    initialModule = "sensor";
  }

  return (
    <>
      <div className="sg-session-bar">
        <span>
          Signed in as <strong>{session.user.full_name}</strong>
        </span>

        <div className="sg-session-actions">
          {view === "analysis" && (
            <button
              type="button"
              onClick={handleBackToDashboard}
            >
              Back to Dashboard
            </button>
          )}

          <button type="button" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>

      {view === "dashboard" ? (
        <UserDashboard
          user={session.user}
          onOpenModule={handleOpenModule}
          onOpenHistory={handleOpenHistory}
          onOpenFreshnessAnalytics={handleOpenFreshnessAnalytics}
        />
      ) : view === "history" ? (
        <AnalysisHistory onBack={handleBackToDashboard} />
      ) : view === "freshness-analytics" ? (
        <FreshnessAnalytics onBack={handleBackToDashboard} />
      ) : (
        <App
          key={selectedModule}
          initialModule={initialModule}
          initialRgbMode={initialRgbMode}
        />
      )}
    </>
  );
}
