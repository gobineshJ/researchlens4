import { useState } from "react";
import { apiFetch } from "../api.js";
import { styles } from "../styles.js";

// The current user's past runs, one column per check, for before/after
// comparisons.
export default function HistoryPanel() {
  const [runs, setRuns] = useState(null);
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);

  async function loadHistory() {
    setLoading(true);
    try {
      const response = await apiFetch("/evaluation-history?limit=20");
      const data = await response.json();
      setRuns(data.runs || []);
    } catch {
      setRuns([]);
    }
    setLoading(false);
  }

  function toggle() {
    const next = !visible;
    setVisible(next);
    if (next && runs === null) loadHistory();
  }

  // Every check name that's appeared across any run, so columns stay
  // consistent even if a check was added/removed/renamed between runs.
  const checkKeys = runs
    ? [...new Set(runs.flatMap((r) => r.checks.map((c) => `${c.stage}/${c.name}`)))]
    : [];

  return (
    <section style={styles.card}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ margin: 0 }}>Your run history</h2>
        <button style={styles.smallButton} onClick={toggle}>
          {visible ? "Hide" : "Show"}
        </button>
      </div>

      {visible && (
        <>
          {loading && <p style={styles.statusText}>Loading history...</p>}
          {!loading && runs && runs.length === 0 && (
            <p style={styles.statusText}>No runs logged yet — process something first.</p>
          )}
          {!loading && runs && runs.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Timestamp</th>
                    <th style={styles.th}>Instructions</th>
                    <th style={styles.th}>Summary</th>
                    {checkKeys.map((key) => (
                      <th key={key} style={styles.th}>{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => {
                    const byKey = {};
                    run.checks.forEach((c) => { byKey[`${c.stage}/${c.name}`] = c; });
                    return (
                      <tr key={run.run_id}>
                        <td style={styles.td}>{new Date(run.timestamp).toLocaleString()}</td>
                        <td style={{ ...styles.td, maxWidth: 200 }}>{run.instructions || "—"}</td>
                        <td style={styles.td}>{run.summary}</td>
                        {checkKeys.map((key) => {
                          const check = byKey[key];
                          if (!check) return <td key={key} style={{ ...styles.td, textAlign: "center" }}>—</td>;
                          const valueText = check.value !== null && check.value !== undefined
                            ? ` (${check.value})` : "";
                          return (
                            <td key={key} style={{ ...styles.td, textAlign: "center" }}>
                              {check.passed ? "✓" : "✗"}{valueText}
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}
