import { useState } from "react";
import { styles } from "../styles.js";

// [1], [1, 3], [2-4] -- the same marker forms the backend reads.
const MARKER = /\[(\d+(?:\s*[-–,]\s*\d+)*)\]/g;

function markerNumbers(body) {
  return body.split(/\s*,\s*/).flatMap((part) => {
    const [lo, hi] = part.split(/\s*[-–]\s*/).map(Number);
    if (hi && hi > lo && hi - lo < 20) return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i);
    return [lo];
  });
}

// The report with every [n] marker clickable: clicking one shows the exact
// excerpt the model was given as source [n], so any sentence can be
// checked against its evidence in one click.
export default function ReportView({ text, sources }) {
  const [open, setOpen] = useState(null); // 1-based source number

  const parts = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    parts.push(text.slice(last, match.index));
    const numbers = markerNumbers(match[1]);
    parts.push(
      <span key={match.index}>
        [
        {numbers.map((n, i) => (
          <span key={n}>
            {i > 0 && ", "}
            <button
              type="button"
              style={{ ...styles.citation, ...(open === n ? styles.citationActive : {}) }}
              onClick={() => setOpen(open === n ? null : n)}
              title={sources[n - 1]?.source_label || `Source ${n}`}
            >
              {n}
            </button>
          </span>
        ))}
        ]
      </span>,
    );
    last = match.index + match[0].length;
  }
  parts.push(text.slice(last));

  const source = open ? sources[open - 1] : null;
  return (
    <>
      <p style={styles.reportText}>{parts}</p>
      {source && (
        <div style={styles.excerptBox}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem" }}>
            <strong>[{open}] {source.source_label}</strong>
            <button type="button" style={styles.smallButton} onClick={() => setOpen(null)}>Close</button>
          </div>
          <div style={{ marginTop: "0.5rem" }}>{source.text}</div>
        </div>
      )}
    </>
  );
}
