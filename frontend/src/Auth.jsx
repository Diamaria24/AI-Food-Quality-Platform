import { useState } from "react";
import "./Auth.css";

const API_URL = "http://127.0.0.1:8000";

export default function Auth({ onLogin }) {
  const [mode, setMode] = useState("login");
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const isRegister = mode === "register";

  function switchMode() {
    setMode(isRegister ? "login" : "register");
    setError("");
    setMessage("");
    setPassword("");
    setShowPassword(false);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (loading) return;

    setError("");
    setMessage("");
    setLoading(true);

    try {
      const normalizedEmail = email.trim().toLowerCase();

      if (isRegister) {
        const registration = await fetch(`${API_URL}/auth/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            full_name: fullName.trim(),
            email: normalizedEmail,
            password,
          }),
        });

        const data = await registration.json();

        if (!registration.ok) {
          throw new Error(
            typeof data.detail === "string"
              ? data.detail
              : "Registration failed. Please check your details."
          );
        }

        setMode("login");
        setPassword("");
        setMessage("Account created successfully. Please sign in.");
        return;
      }

      const response = await fetch(`${API_URL}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: normalizedEmail,
          password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Unable to sign in."
        );
      }

      const profileResponse = await fetch(`${API_URL}/auth/me`, {
        headers: {
          Authorization: `Bearer ${data.access_token}`,
        },
      });

      if (!profileResponse.ok) {
        throw new Error("Unable to verify your account.");
      }

      const user = await profileResponse.json();
      onLogin({ token: data.access_token, user });
    } catch (err) {
      setError(err.message || "Unable to connect to the server.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-panel">
        <div className="auth-brand">
          <div className="auth-brand-icon">?</div>
          <h1>SpectraGuard</h1>
          <p>AI-Powered Food Quality Analysis</p>
        </div>

        <div className="auth-card">
          <div className="auth-heading">
            <h2>{isRegister ? "Create your account" : "Welcome back"}</h2>
            <p>
              {isRegister
                ? "Join SpectraGuard to access intelligent food analysis."
                : "Sign in to access your analysis dashboard."}
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            {isRegister && (
              <label className="auth-field">
                <span>Full name</span>
                <input
                  type="text"
                  value={fullName}
                  onChange={(event) => setFullName(event.target.value)}
                  placeholder="Enter your full name"
                  autoComplete="name"
                  minLength={2}
                  maxLength={150}
                  required
                />
              </label>
            )}

            <label className="auth-field">
              <span>Email address</span>
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@example.com"
                autoComplete="email"
                required
              />
            </label>

            <label className="auth-field">
              <span>Password</span>
              <div className="auth-password">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder={
                    isRegister
                      ? "At least 8 characters"
                      : "Enter your password"
                  }
                  autoComplete={
                    isRegister ? "new-password" : "current-password"
                  }
                  minLength={isRegister ? 8 : undefined}
                  maxLength={128}
                  required
                />
                <button
                  type="button"
                  className="auth-password-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? "Hide" : "Show"}
                </button>
              </div>
            </label>

            {error && (
              <div className="auth-alert auth-error" role="alert">
                {error}
              </div>
            )}

            {message && (
              <div className="auth-alert auth-success" role="status">
                {message}
              </div>
            )}

            <button
              className="auth-submit"
              type="submit"
              disabled={loading}
            >
              {loading
                ? "Please wait..."
                : isRegister
                  ? "Create Account"
                  : "Sign In"}
            </button>
          </form>

          <p className="auth-switch">
            {isRegister
              ? "Already have an account?"
              : "Don't have an account?"}
            <button type="button" onClick={switchMode}>
              {isRegister ? "Sign In" : "Create Account"}
            </button>
          </p>
        </div>

        <p className="auth-footer">
          SpectraGuard ? Research Prototype
        </p>
      </div>
    </div>
  );
}
