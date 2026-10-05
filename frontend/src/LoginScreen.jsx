import { useState } from "react";
import { login, register } from "./api.js";

export default function LoginScreen({ onLoggedIn }) {
  const [mode, setMode] = useState("login"); // login | register
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setBusy(true);
    setErrorMessage("");
    try {
      if (mode === "register") await register(username, password);
      await login(username, password);
      onLoggedIn(username);
    } catch (err) {
      setErrorMessage(err.message || "Something went wrong.");
    }
    setBusy(false);
  }

  const isRegister = mode === "register";

  return (
    <div style={styles.page}>
      <h1>ResearchLens</h1>
      <p style={styles.subtitle}>
        {isRegister ? "Create an account to start generating reports." : "Log in to continue."}
      </p>

      <form style={styles.card} onSubmit={handleSubmit}>
        <label style={styles.label} htmlFor="username">Username</label>
        <input
          id="username"
          style={styles.input}
          autoComplete="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />

        <label style={styles.label} htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          style={styles.input}
          autoComplete={isRegister ? "new-password" : "current-password"}
          minLength={isRegister ? 8 : undefined}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {isRegister && <p style={styles.hint}>At least 8 characters.</p>}

        <button style={styles.button} type="submit" disabled={busy}>
          {busy ? "Please wait..." : isRegister ? "Create account" : "Log in"}
        </button>

        {errorMessage && <p style={styles.error}>{errorMessage}</p>}

        <p style={styles.switchText}>
          {isRegister ? "Already have an account? " : "No account yet? "}
          <button
            type="button"
            style={styles.linkButton}
            onClick={() => {
              setMode(isRegister ? "login" : "register");
              setErrorMessage("");
            }}
          >
            {isRegister ? "Log in" : "Create one"}
          </button>
        </p>
      </form>
    </div>
  );
}

const styles = {
  page: {
    maxWidth: 400,
    margin: "0 auto",
    padding: "4rem 1rem",
    fontFamily: "system-ui, sans-serif",
  },
  subtitle: { color: "#555", marginBottom: "1.5rem" },
  card: {
    border: "1px solid #ddd",
    borderRadius: 8,
    padding: "1.5rem",
  },
  label: { display: "block", marginTop: "1rem", marginBottom: "0.4rem", fontWeight: 600 },
  input: { width: "100%", boxSizing: "border-box", padding: "0.5rem", fontSize: "0.95rem" },
  hint: { color: "#777", fontSize: "0.85rem", margin: "0.3rem 0 0" },
  button: {
    marginTop: "1.2rem",
    padding: "0.6rem 1.4rem",
    fontSize: "1rem",
    cursor: "pointer",
    width: "100%",
  },
  error: { color: "#b02a2a", marginTop: "0.8rem" },
  switchText: { marginTop: "1.2rem", marginBottom: 0, fontSize: "0.9rem", color: "#555" },
  linkButton: {
    background: "none",
    border: "none",
    padding: 0,
    color: "#1a5fb4",
    textDecoration: "underline",
    cursor: "pointer",
    fontSize: "inherit",
  },
};
