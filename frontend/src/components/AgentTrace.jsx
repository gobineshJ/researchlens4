import { styles } from "../styles.js";

const TOOL_LABELS = {
  search_documents: "Searched",
  read_passage: "Read passage",
  graph_lookup: "Graph lookup",
  graph_path: "Graph path",
  finish: "Finished",
};

function describeArgs(tool, args = {}) {
  if (tool === "search_documents") return `“${args.query}”${args.source ? ` in ${args.source}` : ""}`;
  if (tool === "read_passage") return args.passage_id;
  if (tool === "graph_lookup") return args.entity;
  if (tool === "graph_path") return `${args.from_entity} → ${args.to_entity}`;
  if (tool === "finish") return (args.passage_ids || []).join(", ");
  return JSON.stringify(args);
}

// The retrieval agent's tool calls, one line per step: what it chose to do
// and what came back. Shown live while it runs, and with the results.
export default function AgentTrace({ trace, live = false }) {
  if (!trace?.length) return null;
  const tokens = trace[trace.length - 1].tokens;
  return (
    <div style={live ? { marginTop: "0.8rem" } : styles.evalBox}>
      {live ? (
        <div style={styles.progressDetail}>Agent steps so far:</div>
      ) : (
        <h3>Agent steps</h3>
      )}
      <ol style={styles.trace}>
        {trace.map((step, i) => (
          <li key={i} style={styles.traceStep}>
            <strong>{TOOL_LABELS[step.tool] || step.tool}</strong> {describeArgs(step.tool, step.args)}
            <div style={styles.checkSubject}>→ {step.result}</div>
          </li>
        ))}
      </ol>
      {!live && tokens ? (
        <p style={styles.checkSubject}>
          {trace.length} tool call(s), {tokens.toLocaleString()} tokens.
        </p>
      ) : null}
    </div>
  );
}
