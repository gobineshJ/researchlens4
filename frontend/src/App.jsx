import { useState } from "react";

// Backend URL — change this if you run FastAPI on a different port/host.
const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [files, setFiles] = useState([]);
  const [seedUrls, setSeedUrls] = useState("");
  const [instructions, setInstructions] = useState("");
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [errorMessage, setErrorMessage] = useState("");
  const [result, setResult] = useState(null);

  async function handleProcess() {
    if (files.length === 0 && !seedUrls.trim()) {
      setErrorMessage("Add at least one file or one reference URL.");
      return;
    }
    setStatus("running");
    setErrorMessage("");
    setResult(null);

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("seed_urls", seedUrls);
    formData.append("instructions", instructions);

    try {
      const response = await fetch(`${API_BASE}/process`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorBody = await response.json().catch(() => ({}));
        throw new Error(errorBody.detail || `Request failed (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
      setStatus("done");
    } catch (err) {
      setErrorMessage(err.message || "Something went wrong.");
      setStatus("error");
    }
  }

  return (
    <div style={styles.page}>
      <h1>ResearchLens</h1>
      <p style={styles.subtitle}>
        RAG research paper summarizer — upload documents, give seed URLs, describe the report you want.
      </p>

      <section style={styles.card}>
        <label style={styles.label}>Research documents (PDF, DOCX, TXT)</label>
        <input
          type="file"
          multiple
          accept=".pdf,.docx,.txt"
          onChange={(e) => setFiles(Array.from(e.target.files))}
        />

        <label style={styles.label}>Reference URLs (one per line)</label>
        <textarea
          style={styles.textarea}
          rows={3}
          placeholder="https://example.com/related-survey"
          value={seedUrls}
          onChange={(e) => setSeedUrls(e.target.value)}
        />

        <label style={styles.label}>Report instructions</label>
        <textarea
          style={styles.textarea}
          rows={3}
          placeholder="Focus on research gaps, academic tone, add a comparison table, keep under 4 pages."
          value={instructions}
          onChange={(e) => setInstructions(e.target.value)}
        />

        <button style={styles.button} onClick={handleProcess} disabled={status === "running"}>
          {status === "running" ? "Processing..." : "Process"}
        </button>

        {errorMessage && <p style={styles.error}>{errorMessage}</p>}
      </section>

      {status === "running" && (
        <p style={styles.statusText}>
          Running the pipeline — crawling, filtering, embedding, and generating the report.
          This can take a minute or two depending on how many pages and files you gave it.
        </p>
      )}

      {result && <ResultsPanel result={result} apiBase={API_BASE} />}

      <HistoryPanel apiBase={API_BASE} />
    </div>
  );
}

function HistoryPanel({ apiBase }) {
  const [runs, setRuns] = useState(null);
  const [visible, setVisible] = useState(false);
  const [loading, setLoading] = useState(false);

  async function loadHistory() {
    setLoading(true);
    try {
      const response = await fetch(`${apiBase}/evaluation-history?limit=20`);
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
        <h2 style={{ margin: 0 }}>Run history</h2>
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
                          const mark = check.passed ? "✓" : "✗";
                          const valueText = check.value !== null && check.value !== undefined
                            ? ` (${check.value})` : "";
                          return (
                            <td key={key} style={{ ...styles.td, textAlign: "center" }}>
                              {mark}{valueText}
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

function ResultsPanel({ result, apiBase }) {
  return (
    <section style={styles.card}>
      <h2>Results</h2>

      <SummaryRow label="Extracted text blocks" value={result.extracted_records} />
      <SummaryRow label="Pages discovered by crawl" value={result.discovered_pages} />
      <SummaryRow label="Pages that passed the relevance filter" value={result.filtered_pages?.length ?? 0} />
      <SummaryRow label="Chunks created" value={result.chunks_created} />
      <SummaryRow label="Chunks retrieved for the report" value={result.retrieved_chunks?.length ?? 0} />

      {result.report_text && (
        <div style={styles.reportBox}>
          <h3>Generated report</h3>
          <p style={styles.reportText}>{result.report_text}</p>

          {result.citations_used?.length > 0 && (
            <>
              <h4>Sources cited</h4>
              <ul>
                {result.citations_used.map((label, i) => (
                  <li key={i}>{label}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {result.pdf_filename && (
        <a
          style={styles.downloadLink}
          href={`${apiBase}/download/${result.pdf_filename}`}
        >
          Download PDF report
        </a>
      )}

      <EvaluationPanel evaluation={result.evaluation} />
    </section>
  );
}

function EvaluationPanel({ evaluation }) {
  if (!evaluation) return null;
  return (
    <div style={styles.evalBox}>
      <h3>Evaluation report</h3>
      <p>{evaluation.summary}</p>

      <table style={styles.table}>
        <thead>
          <tr>
            <th style={styles.th}>Stage</th>
            <th style={styles.th}>Check</th>
            <th style={styles.th}>Passed</th>
            <th style={styles.th}>Detail</th>
          </tr>
        </thead>
        <tbody>
          {evaluation.checks.map((check, i) => (
            <tr key={i}>
              <td style={styles.td}>{check.stage}</td>
              <td style={styles.td}>{check.name}</td>
              <td style={{ ...styles.td, color: check.passed ? "#2a7a2a" : "#b02a2a" }}>
                {check.passed ? "✓" : "✗"}
              </td>
              <td style={styles.td}>{check.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h4>Stage timings</h4>
      <ul>
        {Object.entries(evaluation.stage_durations_seconds || {}).map(([stage, seconds]) => (
          <li key={stage}>{stage}: {seconds}s</li>
        ))}
      </ul>

      {evaluation.error_count > 0 && (
        <>
          <h4 style={{ color: "#b02a2a" }}>Errors ({evaluation.error_count})</h4>
          <ul>
            {evaluation.errors.map((err, i) => (
              <li key={i} style={{ color: "#b02a2a" }}>{err}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}

function SummaryRow({ label, value }) {
  return (
    <div style={styles.summaryRow}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

const styles = {
  page: {
    maxWidth: 760,
    margin: "0 auto",
    padding: "2rem 1rem",
    fontFamily: "system-ui, sans-serif",
  },
  subtitle: { color: "#555", marginBottom: "1.5rem" },
  card: {
    border: "1px solid #ddd",
    borderRadius: 8,
    padding: "1.5rem",
    marginBottom: "1.5rem",
  },
  label: { display: "block", marginTop: "1rem", marginBottom: "0.4rem", fontWeight: 600 },
  textarea: { width: "100%", fontFamily: "inherit", fontSize: "0.95rem" },
  button: {
    marginTop: "1.2rem",
    padding: "0.6rem 1.4rem",
    fontSize: "1rem",
    cursor: "pointer",
  },
  smallButton: { padding: "0.4rem 0.9rem", fontSize: "0.9rem", cursor: "pointer" },
  error: { color: "#b02a2a", marginTop: "0.8rem" },
  statusText: { color: "#555", fontStyle: "italic" },
  summaryRow: {
    display: "flex",
    justifyContent: "space-between",
    padding: "0.3rem 0",
    borderBottom: "1px solid #eee",
  },
  reportBox: { marginTop: "1.2rem", padding: "1rem", background: "#fafafa", borderRadius: 6 },
  reportText: { whiteSpace: "pre-wrap", lineHeight: 1.5 },
  downloadLink: {
    display: "inline-block",
    marginTop: "1rem",
    padding: "0.5rem 1rem",
    background: "#1a1a1a",
    color: "#fff",
    borderRadius: 6,
    textDecoration: "none",
  },
  evalBox: { marginTop: "1.5rem", paddingTop: "1rem", borderTop: "1px solid #eee" },
  table: { width: "100%", borderCollapse: "collapse", marginTop: "0.5rem" },
  th: { textAlign: "left", borderBottom: "2px solid #ccc", padding: "0.4rem" },
  td: { borderBottom: "1px solid #eee", padding: "0.4rem", fontSize: "0.9rem" },
};
