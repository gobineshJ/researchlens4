import { styles } from "../styles.js";

// Pipeline stage names (as the backend reports them) -> what the user sees.
const STAGE_LABELS = {
  extraction: "Reading your documents",
  crawl_discovery: "Discovering pages from the reference URLs",
  relevance_filter: "Checking which pages are relevant",
  scraping: "Scraping the relevant pages",
  chunking: "Splitting text into chunks",
  embedding_and_storage: "Embedding and indexing chunks",
  retrieval: "Finding the most relevant passages",
  generation: "Writing the report",
  self_correction_retry: "Report was weakly grounded — retrying with new searches",
  entailment: "Checking each cited claim against its source",
  claim_revision: "Rewriting the sentences the claim check flagged",
  pdf_export: "Exporting the PDF",
};

// Live status of a running job. While the report is being generated, the
// worker streams it, so the text written so far shows up here.
export default function ProgressBox({ jobStatus, progress }) {
  return (
    <div style={styles.progressBox}>
      {jobStatus === "queued" ? (
        <strong>Queued — waiting for a worker to pick up the job.</strong>
      ) : progress?.stage ? (
        <>
          <strong>{STAGE_LABELS[progress.stage] || progress.stage}…</strong>
          {progress.detail && <div style={styles.progressDetail}>{progress.detail}</div>}
        </>
      ) : (
        <strong>Running the pipeline…</strong>
      )}
      {progress?.draft && (
        <>
          <div style={styles.progressDetail}>
            {progress.stage === "generation" ? "Report so far:" : "Draft (checks still running):"}
          </div>
          <div style={styles.draft}>{progress.draft}</div>
        </>
      )}
      <p style={styles.statusText}>
        This can take a few minutes depending on how many pages, files and figures you gave it.
      </p>
    </div>
  );
}
