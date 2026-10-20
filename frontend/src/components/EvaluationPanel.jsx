import { styles } from "../styles.js";

// Every check the pipeline ran on this report, with timings and errors.
export default function EvaluationPanel({ evaluation }) {
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
              <td style={styles.td}>
                {check.name}
                {check.subject && <div style={styles.checkSubject}>{check.subject}</div>}
              </td>
              <td style={{ ...styles.td, ...(check.passed ? styles.ok : styles.fail) }}>
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
          <h4 style={styles.fail}>Errors ({evaluation.error_count})</h4>
          <ul>
            {evaluation.errors.map((err, i) => (
              <li key={i} style={styles.fail}>{err}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
