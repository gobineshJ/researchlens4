import { styles } from "../styles.js";

const FLAGGED = new Set(["unsupported", "contradicted"]);

// The entailment check's findings: sentences still not supported by the
// excerpt they cite, and sentences the revision step rewrote or removed
// (with the original wording, so nothing changes silently).
export default function ClaimVerdicts({ verdicts }) {
  if (!verdicts?.length) return null;
  const flagged = verdicts.filter((v) => FLAGGED.has(v.verdict));
  const revised = verdicts.filter((v) => v.original !== undefined);
  const remaining = verdicts.filter((v) => v.verdict !== "removed");
  const supported = remaining.filter((v) => v.verdict === "supported").length;

  return (
    <div style={styles.evalBox}>
      <h3>Claim check</h3>
      <p>
        {supported} of {remaining.length} cited claims are supported by the source they cite.
        {flagged.length === 0 && " None are flagged."}
      </p>

      {flagged.length > 0 && (
        <ul>
          {flagged.map((v, i) => (
            <li key={i} style={{ marginBottom: "0.5rem" }}>
              <strong style={v.verdict === "contradicted" ? styles.fail : styles.warn}>
                {v.verdict === "contradicted" ? "Contradicted" : "Unsupported"}
              </strong>{" "}
              “{v.claim}” {v.sources.map((n) => `[${n}]`).join("")}
              {v.reason && <div style={styles.checkSubject}>{v.reason}</div>}
            </li>
          ))}
        </ul>
      )}

      {revised.length > 0 && (
        <>
          <h4>Revised sentences</h4>
          <ul>
            {revised.map((v, i) => (
              <li key={i} style={{ marginBottom: "0.6rem" }}>
                <div style={styles.checkSubject}>
                  Was ({v.original_verdict}): “{v.original}”
                </div>
                {v.verdict === "removed" ? (
                  <div><strong style={styles.warn}>Removed</strong> — its cited excerpts support none of it.</div>
                ) : (
                  <div>
                    <strong style={v.verdict === "supported" ? styles.ok : styles.warn}>
                      Now: {v.verdict}
                    </strong>{" "}
                    “{v.claim}”
                  </div>
                )}
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
