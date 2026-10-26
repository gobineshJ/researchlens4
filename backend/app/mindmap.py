"""
The report as a mind map: a graph the frontend draws, built from what the
pipeline already has once the report is final.

    question --contains--> theme --contains--> claim --cites--> source document

  claims   every report sentence carrying [n] markers
           (evaluation.extract_cited_claims), with the entailment verdict
           when the claim check judged it
  themes   one small Groq call groups the claims (groq_client.group_claims);
           if it fails, claims are grouped by the document they cite first,
           so a run never ends up without a map
  sources  one node per DOCUMENT (source_name), not per excerpt -- a cites
           edge lists which excerpts [n] it went through, so the UI can still
           show the exact passage

Only plain {nodes, edges} goes back -- layout is the frontend's business.
"""

from urllib.parse import urlparse

from . import evaluation, groq_client
from .config import MIND_MAP_MAX_CLAIMS, MIND_MAP_MAX_THEMES

QUESTION_LABEL_CHARS = 120
OTHER_THEME = "Other points"


def _short_source_label(chunk: dict) -> str:
    """A filename as is; a web page as host + path, without the scheme."""
    name = chunk["source_name"]
    if chunk.get("source_type") == "file":
        return name
    parsed = urlparse(name)
    return (parsed.netloc + parsed.path).rstrip("/") or name


def _themes_by_source(claims: list[dict], retrieved: list[dict]) -> list[dict]:
    """The fallback grouping: one theme per document, holding the claims
    whose FIRST citation points into it, in first-cited order."""
    themes: dict[str, dict] = {}
    for ci, claim in enumerate(claims):
        chunk = retrieved[claim["sources"][0] - 1]
        theme = themes.setdefault(chunk["source_name"], {
            "label": _short_source_label(chunk),
            "summary": "What the report draws from this source.",
            "claims": [],
        })
        theme["claims"].append(ci)
    return list(themes.values())


def build_mind_map(report_text: str, retrieved: list[dict], instructions: str,
                   claim_verdicts: list[dict], report: evaluation.EvaluationReport) -> dict | None:
    """
    Returns {"themes_from": "llm" | "sources", "nodes": [...], "edges": [...]},
    or None when the report cites nothing. Node ids: "q", "t<i>", "c<i>",
    "s<i>". Records how the themes were made as a "mind_map" check.
    """
    claims = evaluation.extract_cited_claims(report_text, len(retrieved), max_claims=MIND_MAP_MAX_CLAIMS)
    if not claims:
        return None

    try:
        themes = groq_client.group_claims(instructions, [c["text"] for c in claims], MIND_MAP_MAX_THEMES)
        themes_from = "llm"
        placed = {ci for t in themes for ci in t["claims"]}
        left_out = [ci for ci in range(len(claims)) if ci not in placed]
        if left_out:
            themes.append({"label": OTHER_THEME,
                           "summary": "Claims that did not fit any of the main themes.",
                           "claims": left_out})
        report.add_check(
            "mind_map", "themes", True,
            f"{len(claims)} cited claim(s) grouped into {len(themes)} theme(s)"
            + (f"; {len(left_out)} the grouping left out went under '{OTHER_THEME}'." if left_out else "."),
            value=len(themes),
        )
    except Exception as exc:
        themes = _themes_by_source(claims, retrieved)
        themes_from = "sources"
        report.record_error("mind_map", f"Claims could not be grouped into themes: {exc}")
        report.add_check(
            "mind_map", "themes", False,
            f"Theme grouping failed, so the {len(claims)} cited claim(s) are grouped by the "
            f"document they cite first ({len(themes)} group(s)) instead.",
            value=len(themes),
        )

    question = " ".join((instructions or "").split()) or "Summary of the sources"
    if len(question) > QUESTION_LABEL_CHARS:
        question = question[:QUESTION_LABEL_CHARS - 1].rstrip() + "…"
    nodes: list[dict] = [{"id": "q", "type": "question", "label": question}]
    edges: list[dict] = []

    verdict_by_claim = {v["claim"]: v for v in claim_verdicts if v.get("verdict") != "removed"}
    source_ids: dict[str, str] = {}
    for ti, theme in enumerate(themes):
        theme_id = f"t{ti}"
        nodes.append({"id": theme_id, "type": "theme", "label": theme["label"],
                      "summary": theme["summary"], "claim_count": len(theme["claims"])})
        edges.append({"source": "q", "target": theme_id, "type": "contains"})

        for ci in theme["claims"]:
            claim = claims[ci]
            judged = verdict_by_claim.get(claim["text"], {})
            claim_id = f"c{ci}"
            nodes.append({"id": claim_id, "type": "claim", "label": claim["text"],
                          "citations": claim["sources"],
                          "verdict": judged.get("verdict", "unjudged"),
                          "reason": judged.get("reason", "")})
            edges.append({"source": theme_id, "target": claim_id, "type": "contains"})

            # One cites edge per document, carrying every excerpt cited in it.
            excerpts_by_doc: dict[str, list[int]] = {}
            for n in claim["sources"]:
                excerpts_by_doc.setdefault(retrieved[n - 1]["source_name"], []).append(n)
            for name, excerpts in excerpts_by_doc.items():
                if name not in source_ids:
                    chunk = retrieved[excerpts[0] - 1]
                    source_ids[name] = f"s{len(source_ids)}"
                    nodes.append({"id": source_ids[name], "type": "source",
                                  "label": _short_source_label(chunk), "name": name,
                                  "source_type": chunk.get("source_type", "file")})
                edges.append({"source": claim_id, "target": source_ids[name],
                              "type": "cites", "excerpts": excerpts})

    return {"themes_from": themes_from, "nodes": nodes, "edges": edges}
