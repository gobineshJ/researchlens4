import { useEffect, useMemo, useState } from "react";
import { Background, Controls, Handle, Position, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { styles } from "../styles.js";

// The report as a graph (backend: mindmap.build_mind_map):
//   question -> themes -> claims -> the documents each claim cites.
// Layout is done here, in four columns, so the map reads left to right the
// way an argument does; React Flow only draws it and handles pan/zoom.

const COLUMN_X = { question: 0, theme: 290, claim: 580, source: 1090 };
const WIDTH = { question: 230, theme: 230, claim: 370, source: 230 };
const CLAIM_ROW = 86;      // vertical step between claims
const THEME_GAP = 34;      // extra space between one theme's claims and the next's
const SOURCE_ROW = 84;      // source boxes are ~70px tall

// Theme colors avoid green and red, which mean "supported" / "contradicted".
const THEME_COLORS = ["#1a5fb4", "#c64600", "#7b4fa8", "#1b8a8a", "#8a5a2b", "#b8336a", "#4a5d7a"];
const VERDICT = {
  supported: { color: "#2a7a2a", label: "Supported" },
  unsupported: { color: "#9a6700", label: "Unsupported" },
  contradicted: { color: "#b02a2a", label: "Contradicted" },
  unjudged: { color: "#9a9a9a", label: "Not checked" },
};
const FLAGGED = new Set(["unsupported", "contradicted"]);

const box = {
  background: "#fff",
  border: "1px solid #d0d0d0",
  borderRadius: 8,
  padding: "8px 10px",
  fontSize: 12,
  lineHeight: 1.35,
  boxShadow: "0 1px 2px rgba(0,0,0,0.06)",
  transition: "opacity 0.15s",
};
const clamp = (lines) => ({
  display: "-webkit-box",
  WebkitLineClamp: lines,
  WebkitBoxOrient: "vertical",
  overflow: "hidden",
});
const hidden = { opacity: 0, pointerEvents: "none" };
const ring = (data) => (data.selected ? { boxShadow: "0 0 0 3px #f5c542" } : {});

function QuestionNode({ data }) {
  return (
    <div style={{ ...box, width: WIDTH.question, background: "#1a1a1a", color: "#fff", border: "none",
                  fontSize: 13, fontWeight: 600, opacity: data.dim ? 0.3 : 1, ...ring(data) }}>
      <div style={clamp(5)}>{data.label}</div>
      <Handle type="source" position={Position.Right} style={hidden} />
    </div>
  );
}

function ThemeNode({ data }) {
  return (
    <div style={{ ...box, width: WIDTH.theme, borderColor: data.color, borderWidth: 2,
                  opacity: data.dim ? 0.25 : 1, ...ring(data) }}>
      <Handle type="target" position={Position.Left} style={hidden} />
      <div style={{ fontWeight: 700, color: data.color, fontSize: 13 }}>{data.label}</div>
      <div style={{ color: "#666", marginTop: 2 }}>
        {data.claim_count} claim{data.claim_count === 1 ? "" : "s"}
      </div>
      <Handle type="source" position={Position.Right} style={hidden} />
    </div>
  );
}

function ClaimNode({ data }) {
  const verdict = VERDICT[data.verdict] || VERDICT.unjudged;
  return (
    <div style={{ ...box, width: WIDTH.claim, borderLeft: `4px solid ${verdict.color}`,
                  opacity: data.dim ? 0.2 : 1, ...ring(data) }}>
      <Handle type="target" position={Position.Left} style={hidden} />
      <div style={clamp(3)}>{data.label}</div>
      <div style={{ marginTop: 3, color: verdict.color, fontSize: 11 }}>
        {verdict.label} · {data.citations.map((n) => `[${n}]`).join("")}
      </div>
      <Handle type="source" position={Position.Right} style={hidden} />
    </div>
  );
}

function SourceNode({ data }) {
  return (
    <div style={{ ...box, width: WIDTH.source, background: "#f4f7fb", opacity: data.dim ? 0.25 : 1, ...ring(data) }}>
      <Handle type="target" position={Position.Left} style={hidden} />
      <div style={{ fontSize: 10, color: "#666", textTransform: "uppercase", letterSpacing: 0.4 }}>
        {data.source_type === "web" ? "Web page" : "Document"}
      </div>
      <div style={{ ...clamp(2), fontWeight: 600, wordBreak: "break-word" }} title={data.name}>{data.label}</div>
      <div style={{ color: "#666", marginTop: 2 }}>
        cited by {data.cited_by} claim{data.cited_by === 1 ? "" : "s"}
      </div>
    </div>
  );
}

// Defined once, outside the component: React Flow re-mounts every node when
// this object changes identity.
const NODE_TYPES = { question: QuestionNode, theme: ThemeNode, claim: ClaimNode, source: SourceNode };

// Positions for every node. Claims stack in theme order; each theme sits at
// the middle of its claims and the question at the middle of the themes.
// Sources are ordered by the average height of the claims citing them, which
// keeps cites edges from crossing more than they must.
function layout(graph) {
  const byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  const children = {};
  const parents = {};
  for (const e of graph.edges) {
    (children[e.source] ||= []).push(e.target);
    (parents[e.target] ||= []).push(e.source);
  }

  const y = {};
  let cursor = 0;
  const themes = graph.nodes.filter((n) => n.type === "theme");
  for (const theme of themes) {
    const claims = children[theme.id] || [];
    claims.forEach((id, i) => { y[id] = cursor + i * CLAIM_ROW; });
    y[theme.id] = cursor + ((claims.length - 1) * CLAIM_ROW) / 2;
    cursor += claims.length * CLAIM_ROW + THEME_GAP;
  }
  const mean = (ids) => ids.reduce((sum, id) => sum + y[id], 0) / (ids.length || 1);
  y.q = mean(themes.map((t) => t.id));

  const sources = graph.nodes
    .filter((n) => n.type === "source")
    .map((n) => ({ id: n.id, want: mean(parents[n.id] || []) }))
    .sort((a, b) => a.want - b.want);
  let previous = -Infinity;
  for (const s of sources) {
    y[s.id] = Math.max(s.want, previous + SOURCE_ROW);
    previous = y[s.id];
  }

  return Object.fromEntries(Object.keys(y).map((id) => [id, { x: COLUMN_X[byId[id].type], y: y[id] }]));
}

// The node plus everything upstream and downstream of it: a theme lights up
// its claims and their sources, a source the claims citing it and their themes.
function neighbourhood(graph, id) {
  const lit = new Set([id]);
  for (const [from, to] of [["source", "target"], ["target", "source"]]) {
    const stack = [id];
    while (stack.length) {
      const current = stack.pop();
      for (const e of graph.edges) {
        if (e[from] === current && !lit.has(e[to])) {
          lit.add(e[to]);
          stack.push(e[to]);
        }
      }
    }
  }
  return lit;
}

// Only flagged claims, plus the themes and sources still connected to them.
function flaggedOnly(graph) {
  const keep = new Set(["q"]);
  for (const n of graph.nodes) if (n.type === "claim" && FLAGGED.has(n.verdict)) keep.add(n.id);
  for (const e of graph.edges) {
    if (e.type === "contains" && keep.has(e.target) && e.source !== "q") keep.add(e.source);
    if (e.type === "cites" && keep.has(e.source)) keep.add(e.target);
  }
  return {
    ...graph,
    nodes: graph.nodes.filter((n) => keep.has(n.id)),
    edges: graph.edges.filter((e) => keep.has(e.source) && keep.has(e.target)),
  };
}

function Excerpt({ n, chunk }) {
  const [open, setOpen] = useState(false);
  if (!chunk) return null;
  return (
    <li style={{ marginBottom: "0.4rem" }}>
      <button type="button" style={styles.citation} onClick={() => setOpen(!open)}>[{n}]</button>{" "}
      {chunk.source_label}
      {open && <div style={{ ...styles.excerptBox, marginTop: "0.3rem" }}>{chunk.text}</div>}
    </li>
  );
}

// What the selected node says in full -- node boxes only have room for a
// clamped preview.
function DetailPanel({ graph, node, colorOf, sources, onSelect }) {
  if (!node) {
    return (
      <p style={styles.checkSubject}>
        Click a theme, claim or source to see it in full and highlight what it connects to.
        Drag to pan, scroll to zoom.
      </p>
    );
  }
  const byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  const linked = (id, dir, type) =>
    graph.edges.filter((e) => e[dir === "out" ? "source" : "target"] === id && e.type === type)
      .map((e) => ({ edge: e, node: byId[dir === "out" ? e.target : e.source] }));
  const claimLink = (c) => {
    const verdict = VERDICT[c.verdict] || VERDICT.unjudged;
    return (
      <li key={c.id} style={{ marginBottom: "0.5rem", paddingLeft: "0.5rem", borderLeft: `3px solid ${verdict.color}` }}>
        <button type="button" style={{ ...styles.citation, textAlign: "left", textDecoration: "none" }}
                onClick={() => onSelect(c.id)}>
          {c.label}
        </button>
        <div style={{ color: verdict.color, fontSize: "0.78rem" }}>{verdict.label}</div>
      </li>
    );
  };
  const claimList = { listStyle: "none", paddingLeft: 0 };

  if (node.type === "question") {
    return <><h4 style={{ marginTop: 0 }}>Question</h4><p>{node.label}</p></>;
  }
  if (node.type === "theme") {
    return (
      <>
        <h4 style={{ marginTop: 0, color: colorOf[node.id] }}>{node.label}</h4>
        {node.summary && <p>{node.summary}</p>}
        <ul style={claimList}>{linked(node.id, "out", "contains").map((l) => claimLink(l.node))}</ul>
      </>
    );
  }
  if (node.type === "claim") {
    const verdict = VERDICT[node.verdict] || VERDICT.unjudged;
    const theme = linked(node.id, "in", "contains")[0]?.node;
    return (
      <>
        <h4 style={{ marginTop: 0 }}>Claim</h4>
        <p>“{node.label}”</p>
        <p>
          <strong style={{ color: verdict.color }}>{verdict.label}</strong>
          {node.reason && <span style={styles.checkSubject}> — {node.reason}</span>}
        </p>
        {theme && (
          <p style={styles.checkSubject}>
            Theme:{" "}
            <button type="button" style={styles.citation} onClick={() => onSelect(theme.id)}>{theme.label}</button>
          </p>
        )}
        <div style={styles.checkSubject}>Cited excerpts — click one to read it:</div>
        <ul style={{ paddingLeft: "1.1rem" }}>
          {node.citations.map((n) => <Excerpt key={n} n={n} chunk={sources[n - 1]} />)}
        </ul>
      </>
    );
  }
  // source
  const citing = linked(node.id, "in", "cites");
  const excerpts = [...new Set(citing.flatMap((l) => l.edge.excerpts))].sort((a, b) => a - b);
  return (
    <>
      <h4 style={{ marginTop: 0, wordBreak: "break-word" }}>{node.name}</h4>
      <div style={styles.checkSubject}>Claims citing it:</div>
      <ul style={claimList}>{citing.map((l) => claimLink(l.node))}</ul>
      <div style={styles.checkSubject}>Excerpts used from it:</div>
      <ul style={{ paddingLeft: "1.1rem" }}>
        {excerpts.map((n) => <Excerpt key={n} n={n} chunk={sources[n - 1]} />)}
      </ul>
    </>
  );
}

export default function MindMap({ graph: fullGraph, sources }) {
  const [selected, setSelected] = useState(null);
  const [hovered, setHovered] = useState(null);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const flaggedCount = fullGraph.nodes.filter((n) => n.type === "claim" && FLAGGED.has(n.verdict)).length;
  const graph = useMemo(() => (onlyFlagged ? flaggedOnly(fullGraph) : fullGraph), [fullGraph, onlyFlagged]);
  const positions = useMemo(() => layout(graph), [graph]);

  const colorOf = useMemo(() => {
    const colors = {};
    fullGraph.nodes.filter((n) => n.type === "theme")
      .forEach((t, i) => { colors[t.id] = THEME_COLORS[i % THEME_COLORS.length]; });
    for (const e of fullGraph.edges) if (e.type === "contains" && colors[e.source]) colors[e.target] = colors[e.source];
    return colors;
  }, [fullGraph]);

  const focus = hovered || selected;
  const lit = useMemo(() => (focus && graph.nodes.some((n) => n.id === focus) ? neighbourhood(graph, focus) : null),
    [graph, focus]);

  const citedBy = useMemo(() => {
    const counts = {};
    for (const e of graph.edges) if (e.type === "cites") counts[e.target] = (counts[e.target] || 0) + 1;
    return counts;
  }, [graph]);

  const nodes = useMemo(() => graph.nodes.map((n) => ({
    id: n.id,
    type: n.type,
    position: positions[n.id],
    data: { ...n, color: colorOf[n.id], cited_by: citedBy[n.id] || 0,
            dim: lit ? !lit.has(n.id) : false, selected: n.id === selected },
  })), [graph, positions, colorOf, citedBy, lit, selected]);

  const edges = useMemo(() => graph.edges.map((e) => {
    const on = !lit || (lit.has(e.source) && lit.has(e.target));
    const cites = e.type === "cites";
    return {
      id: `${e.source}-${e.target}`,
      source: e.source,
      target: e.target,
      style: {
        stroke: cites ? "#7f9cc4" : colorOf[e.target] || "#999",
        strokeWidth: on && lit ? 2 : 1.4,
        strokeDasharray: cites ? "4 4" : undefined,
        opacity: on ? 0.9 : 0.08,
      },
    };
  }), [graph, lit, colorOf]);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (event) => { if (event.key === "Escape") setExpanded(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  const selectedNode = graph.nodes.find((n) => n.id === selected) || null;
  const themesCount = graph.nodes.filter((n) => n.type === "theme").length;
  const claimsCount = graph.nodes.filter((n) => n.type === "claim").length;
  const sourcesCount = graph.nodes.filter((n) => n.type === "source").length;

  const shell = expanded
    ? { position: "fixed", inset: 0, zIndex: 1000, background: "#fff", padding: "0.8rem",
        display: "flex", flexDirection: "column" }
    : {};

  return (
    <div style={shell}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.6rem", alignItems: "center", marginBottom: "0.6rem" }}>
        <span style={styles.checkSubject}>
          {themesCount} theme{themesCount === 1 ? "" : "s"} · {claimsCount} claim{claimsCount === 1 ? "" : "s"} ·{" "}
          {sourcesCount} source{sourcesCount === 1 ? "" : "s"}
          {fullGraph.themes_from === "sources" && " · themes couldn't be generated, so claims are grouped by source"}
        </span>
        <span style={{ flex: 1 }} />
        {flaggedCount > 0 && (
          <label style={{ fontSize: "0.85rem", display: "flex", gap: "0.3rem", alignItems: "center" }}>
            <input type="checkbox" checked={onlyFlagged}
                   onChange={(e) => { setOnlyFlagged(e.target.checked); setSelected(null); }} />
            Only flagged claims ({flaggedCount})
          </label>
        )}
        <button type="button" style={styles.smallButton} onClick={() => setExpanded(!expanded)}>
          {expanded ? "Close full screen" : "Full screen"}
        </button>
      </div>

      <div style={{ display: "flex", flexDirection: expanded ? "row" : "column", gap: "0.8rem",
                    flex: expanded ? 1 : undefined, minHeight: 0 }}>
        <div style={{ height: expanded ? "auto" : 520, flex: expanded ? 1 : undefined,
                      border: "1px solid #ddd", borderRadius: 8, background: "#fcfcfd" }}>
          {/* Keyed so a new canvas size or node set re-mounts it, and fitView
              fits the new one. */}
          <ReactFlow
            key={`${expanded}-${onlyFlagged}`}
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            fitView
            fitViewOptions={{ padding: 0.12 }}
            minZoom={0.15}
            nodesConnectable={false}
            nodesDraggable={false}
            onNodeClick={(_, node) => setSelected(node.id === selected ? null : node.id)}
            onNodeMouseEnter={(_, node) => setHovered(node.id)}
            onNodeMouseLeave={() => setHovered(null)}
            onPaneClick={() => setSelected(null)}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={22} color="#e6e6e6" />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>

        <div style={{ width: expanded ? 380 : undefined, overflowY: "auto", fontSize: "0.92rem",
                      ...(expanded ? { borderLeft: "1px solid #eee", paddingLeft: "0.8rem" } : {}) }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.8rem", marginBottom: "0.6rem", fontSize: "0.8rem" }}>
            {Object.values(VERDICT).map((v) => (
              <span key={v.label} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: v.color }} /> {v.label}
              </span>
            ))}
            <span style={{ color: "#666", whiteSpace: "nowrap" }}>- - - cites</span>
          </div>
          <DetailPanel graph={graph} node={selectedNode} colorOf={colorOf} sources={sources}
                       onSelect={setSelected} />
        </div>
      </div>
    </div>
  );
}
