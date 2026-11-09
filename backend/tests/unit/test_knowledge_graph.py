"""
The rule-based knowledge graph: entity extraction, surname aliases, edges
that keep their source passage, lookup, and multi-document paths -- on
the hard golden set's real documents, plus small synthetic cases.
"""

from pathlib import Path

import pytest

from app import chunker
from app import knowledge_graph as kg

HARD_DOCS = Path(__file__).resolve().parents[2] / "eval" / "data" / "golden_hard" / "docs"


@pytest.fixture(scope="module")
def hard_graph():
    records = [{"source_type": "file", "filename": p.name, "location": "", "text": p.read_text(encoding="utf-8")}
               for p in sorted(HARD_DOCS.glob("*.txt"))]
    chunks = chunker.chunk_all(records)
    names = sorted({c["source_name"] for c in chunks})
    passages = [{"cid": f"{names.index(c['source_name']) + 1}.{c['position']}", "source_name": c["source_name"],
                 "text": c["text"]} for c in chunks]
    documents = [{"source_name": chunker._source_name_for(r), "text": r["text"]} for r in records]
    return kg.KnowledgeGraph.build(passages, documents), {p["cid"]: p["text"] for p in passages}


# ---------- extraction ----------

@pytest.mark.parametrize("sentence,expected", [
    ("All cells used VX-9, supplied by Varn Materials.", ["VX-9", "Varn Materials"]),
    ("The centre is accredited to the EN 6240 standard.", ["EN 6240"]),
    ("Entry 4: Merlin-Na-3 was tested at Oskarsund Battery Test Centre.", ["Merlin-Na-3", "Oskarsund Battery Test Centre"]),
    ("We compare KestrelCell with SodaCore-X on eight A100 GPUs.", ["KestrelCell", "SodaCore-X", "A100"]),
    ("Training took 64 hours on room-temperature cells.", []),           # sentence opener, plain hyphenation
    ("2 Production", []),                                                  # a section heading
    ("The chemistry developed by Kade and colleagues at the Halvik Institute of Electrochemistry.",
     ["Kade", "Halvik Institute of Electrochemistry"]),
])
def test_entity_extraction(sentence, expected):
    assert [name for name, _ in kg.extract_entities(sentence)] == expected


def test_long_capitalised_runs_are_titles_not_entities():
    assert kg.extract_entities("We read the Lithium Iron Phosphate Batteries for Municipal Grid Storage review.") == []


# ---------- building ----------

def _tiny():
    passages = [
        {"cid": "1.0", "source_name": "a", "text": "Dr. Ilse Kade designed KestrelCell. KestrelCell used VX-9."},
        {"cid": "2.0", "source_name": "b", "text": "The pilot in Norrby used the chemistry of Kade. Norrby is cold."},
        {"cid": "3.0", "source_name": "c", "text": "VX-9 is produced at the Brekke plant."},
    ]
    return kg.KnowledgeGraph.build(passages)


def test_a_lone_surname_merges_into_the_full_name():
    graph = _tiny()
    assert "kade" not in graph.names and graph.names["ilse kade"] == "Ilse Kade"
    assert graph.passages["ilse kade"] == {"1.0", "2.0"}


def test_edges_keep_the_sentence_and_passage_they_came_from():
    graph = _tiny()
    edge = graph.edges[tuple(sorted(("kestrelcell", "vx-9")))]
    assert edge["cost"] == 1
    assert edge["evidence"][0] == {"cid": "1.0", "text": "KestrelCell used VX-9"}


def test_lookup_and_fuzzy_matching():
    graph = _tiny()
    result = graph.lookup("vx 9")  # case and punctuation differ
    assert result["found"] and result["entity"] == "VX-9"
    assert {r["to"] for r in result["relations"]} >= {"KestrelCell", "Brekke"}
    assert graph.lookup("Nonexistent Thing")["found"] is False


def test_path_crosses_documents_and_cites_every_link():
    path = _tiny().path("Norrby", "Brekke")
    assert path["found"]
    assert [s["to"] for s in path["steps"]][-1] == "Brekke"
    assert {s["evidence"]["cid"] for s in path["steps"]} >= {"2.0", "3.0"}


def test_path_reports_unknown_entities_and_disconnected_ones():
    graph = kg.KnowledgeGraph.build([{"cid": "1.0", "source_name": "a", "text": "Alpha Corp met Beta Corp."},
                                     {"cid": "2.0", "source_name": "b", "text": "Gamma Labs met Delta Labs."}])
    assert "No entity matching" in graph.path("Alpha Corp", "Zeta")["reason"]
    assert graph.path("Alpha Corp", "Gamma Labs")["found"] is False


# ---------- on the hard golden set ----------

def test_hard_set_bridges_are_reachable(hard_graph):
    graph, texts = hard_graph
    for a, b in [("Norrby", "VX-9"), ("Norrby", "Amara Lindqvist"), ("Norrby", "Brekke"),
                 ("KestrelCell", "Oskarsund Battery Test Centre")]:
        path = graph.path(a, b)
        assert path["found"], (a, b)
        # Every link points at a real passage.
        assert all(step["evidence"]["cid"] in texts for step in path["steps"])


def test_hard_set_keeps_look_alikes_apart(hard_graph):
    graph, _ = hard_graph
    assert graph.match("KestrelCell")[0] == "kestrelcell"
    assert graph.match("Norrby")[0] == "norrby"
    assert "kestrelpack" in graph.names and "norrvik" in graph.names
