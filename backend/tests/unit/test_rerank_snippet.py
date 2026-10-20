"""
reranker._scoring_snippet: what the cross-encoder actually scores. Real
academic-PDF chunks often open with a running header, a caption or a row
of table numbers -- those must not fill the snippet.
"""

from app import reranker
from app.config import RERANK_SNIPPET_CHARS

BODY = ("We propose a contrastive pre-training objective for clinical notes and show that it "
        "improves downstream classification on three hospital datasets. ")


def test_arxiv_header_is_stripped():
    text = "arXiv:2101.00001v2 [cs.CL] 3 Jan 2021 " + BODY
    assert reranker._scoring_snippet(text).startswith("We propose")


def test_conference_header_and_page_number_are_stripped():
    text = "Published as a conference paper at ICLR 2021 4 " + BODY
    assert reranker._scoring_snippet(text).startswith("We propose")


def test_caption_sentence_is_stripped_but_in_text_reference_kept():
    text = ("Figure 3: Accuracy of all models across five random seeds. " + BODY
            + "As shown in Figure 3. the gap widens with more data.")
    snippet = reranker._scoring_snippet(text)
    assert snippet.startswith("We propose")
    assert "Accuracy of all models" not in snippet
    assert "As shown in Figure 3." in snippet


def test_table_number_rows_doi_url_and_email_are_stripped():
    text = ("Table 2. Results. 0.81 0.77 0.92 0.88 71.2 "
            "doi:10.1145/1234567 https://github.com/x/y a.b@uni.edu " + BODY)
    snippet = reranker._scoring_snippet(text)
    assert snippet.startswith("We propose")
    for junk in ("0.81", "doi", "github", "@"):
        assert junk not in snippet


def test_snippet_is_capped_at_the_configured_length():
    assert len(reranker._scoring_snippet(BODY * 20)) == RERANK_SNIPPET_CHARS


def test_mostly_table_chunk_falls_back_to_raw_text():
    text = " ".join(f"{i}.{i}" for i in range(200))
    assert reranker._scoring_snippet(text) == text[:RERANK_SNIPPET_CHARS]


def test_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(reranker, "RERANK_STRIP_BOILERPLATE", False)
    text = "arXiv:2101.00001v2 [cs.CL] 3 Jan 2021 " + BODY
    assert reranker._scoring_snippet(text) == text[:RERANK_SNIPPET_CHARS]


def test_rerank_scores_the_cleaned_snippet_but_returns_full_text(monkeypatch):
    seen = []

    class Model:
        def predict(self, pairs):
            seen.extend(text for _, text in pairs)
            return [1.0] * len(pairs)

    monkeypatch.setattr(reranker, "_get_reranker", lambda: Model())
    text = "arXiv:2101.00001v2 [cs.CL] 3 Jan 2021 " + BODY
    result = reranker.rerank("contrastive pre-training", [{"text": text}], top_k=1)
    assert seen[0].startswith("We propose")
    assert result[0]["text"] == text
