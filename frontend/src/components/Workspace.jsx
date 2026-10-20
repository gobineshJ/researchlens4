import { useEffect, useRef, useState } from "react";
import { apiFetch } from "../api.js";
import { styles } from "../styles.js";
import HistoryPanel from "./HistoryPanel.jsx";
import ProgressBox from "./ProgressBox.jsx";
import ResultsPanel from "./ResultsPanel.jsx";

const POLL_INTERVAL_MS = 2000;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// The upload form, the job it starts, and its results.
export default function Workspace({ username, onLogout }) {
  const [files, setFiles] = useState([]);
  const [seedUrls, setSeedUrls] = useState("");
  const [instructions, setInstructions] = useState("");
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [errorMessage, setErrorMessage] = useState("");
  const [result, setResult] = useState(null);
  const [jobStatus, setJobStatus] = useState(""); // queued | running, while a job is in flight
  const [progress, setProgress] = useState(null); // {stage, detail, draft} from the worker

  // Stops the polling loop if the user logs out mid-run.
  const unmounted = useRef(false);
  useEffect(() => {
    unmounted.current = false;
    return () => { unmounted.current = true; };
  }, []);

  async function handleProcess() {
    if (files.length === 0 && !seedUrls.trim()) {
      setErrorMessage("Add at least one file or one reference URL.");
      return;
    }
    setStatus("running");
    setErrorMessage("");
    setResult(null);
    setProgress(null);

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("seed_urls", seedUrls);
    formData.append("instructions", instructions);

    try {
      // /process only enqueues the run on a Celery worker; the report
      // itself arrives by polling /jobs/{job_id} until it's finished.
      const response = await apiFetch("/process", { method: "POST", body: formData });
      const { job_id } = await response.json();
      setJobStatus("queued");

      while (!unmounted.current) {
        const job = await (await apiFetch(`/jobs/${job_id}`)).json();
        if (job.status === "done") {
          setResult(job.result);
          setStatus("done");
          return;
        }
        if (job.status === "error") throw new Error(job.error || "The pipeline failed.");
        setJobStatus(job.status);
        setProgress(job.progress || null);
        await sleep(POLL_INTERVAL_MS);
      }
    } catch (err) {
      if (unmounted.current) return;
      setErrorMessage(err.message || "Something went wrong.");
      setStatus("error");
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.header}>
        <h1 style={{ margin: 0 }}>ResearchLens</h1>
        <div style={styles.userBox}>
          <span>{username}</span>
          <button style={styles.smallButton} onClick={onLogout}>Log out</button>
        </div>
      </div>
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

      {status === "running" && <ProgressBox jobStatus={jobStatus} progress={progress} />}

      {result && <ResultsPanel result={result} />}

      <HistoryPanel />
    </div>
  );
}
