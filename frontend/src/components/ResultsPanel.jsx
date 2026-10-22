import { useState } from "react";
import { downloadFile } from "../api.js";
import { styles } from "../styles.js";
import ClaimVerdicts from "./ClaimVerdicts.jsx";
import EvaluationPanel from "./EvaluationPanel.jsx";
import ReportView from "./ReportView.jsx";

function SummaryRow({ label, value }) {
  return (
    <div style={styles.summaryRow}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

// A finished job: counts, the report (with clickable citations), the claim
// check, the PDF download and the evaluation table.
export default function ResultsPanel({ result }) {
  const [downloadError, setDownloadError] = useState("");
  // Numbered by the report's own [n] markers. Results from before the API
  // returned `citations` only have the un-numbered labels.
  const citations = result.citations
    ?? (result.citations_used || []).map((label, i) => ({ n: i + 1, label }));

  async function handleDownload() {
    setDownloadError("");
    try {
      await downloadFile(`/download/${result.pdf_filename}`, result.pdf_filename);
    } catch (err) {
      setDownloadError(err.message || "Download failed.");
    }
  }

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
          <p style={styles.checkSubject}>Click a citation number to see the excerpt it refers to.</p>
          <ReportView text={result.report_text} sources={result.retrieved_chunks || []} />

          {citations.length > 0 && (
            <>
              <h4>Sources cited</h4>
              <ul>
                {citations.map((c) => (
                  <li key={c.n}>[{c.n}] {c.label}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      <ClaimVerdicts verdicts={result.claim_verdicts} />

      {result.pdf_filename && (
        <button style={styles.downloadLink} onClick={handleDownload}>
          Download PDF report
        </button>
      )}
      {downloadError && <p style={styles.error}>{downloadError}</p>}

      <EvaluationPanel evaluation={result.evaluation} />
    </section>
  );
}
