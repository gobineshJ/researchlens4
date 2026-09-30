"""
Runs Precision@K / Recall@K / MRR (evaluation.evaluate_retrieval) against
a REAL, labeled retrieval -- real embeddings, real ChromaDB, no mocking.

This closes a gap that was previously just flagged in the README:
evaluate_retrieval() existed and was unit-tested as a pure metric
function (given retrieved/relevant labels, does the math come out
right?), but had never actually been run against real retrieval output,
so there was no evidence retrieval quality was actually good, only that
the scoring formula was.

Needs real network access to download the embedding model (~90MB, first
run only) -- skipped by default like the other live-network tests in
this suite. Run with: RUN_EMBEDDING_TESTS=1 pytest tests/integration/test_retrieval_metrics.py -v
"""

import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_EMBEDDING_TESTS"),
    reason="Set RUN_EMBEDDING_TESTS=1 to run tests that download/use the real embedding model.",
)

from app import vectorstore, evaluation


def _labeled_corpus() -> list[dict]:
    """
    Three clearly distinct topics, several chunks each, so a query about
    ONE topic has an unambiguous, known set of "relevant" chunks and a
    known set of "irrelevant" ones -- real ground truth, not guessed.
    """
    chunks = []

    ml_sentences = [
        "Convolutional neural networks are widely used for image classification tasks.",
        "Transformer architectures rely on self-attention mechanisms to process sequences.",
        "Overfitting occurs when a model memorizes training data instead of generalizing.",
        "Gradient descent optimizes model parameters by following the loss function's slope.",
        "Transfer learning reuses a pretrained model's weights for a new, related task.",
    ]
    for i, text in enumerate(ml_sentences):
        chunks.append({
            "text": text, "source_label": f"ml.pdf, page {i}",
            "source_type": "file", "source_name": "ml.pdf",
        })

    whale_sentences = [
        "Blue whales are the largest animals known to have ever existed on Earth.",
        "Humpback whales migrate thousands of miles between feeding and breeding grounds.",
        "Whale populations were severely depleted by commercial whaling in the 20th century.",
        "Baleen whales filter small prey like krill from large volumes of seawater.",
        "Whale songs can travel enormous distances underwater for communication.",
    ]
    for i, text in enumerate(whale_sentences):
        chunks.append({
            "text": text, "source_label": f"whales.pdf, page {i}",
            "source_type": "file", "source_name": "whales.pdf",
        })

    printing_sentences = [
        "The printing press was invented by Johannes Gutenberg in the 15th century.",
        "Movable type allowed books to be reproduced far faster than by hand-copying.",
        "The spread of printed material contributed to rising literacy rates in Europe.",
        "Early printed books, known as incunabula, are highly valued by collectors today.",
        "The printing press is often credited with accelerating the Renaissance and Reformation.",
    ]
    for i, text in enumerate(printing_sentences):
        chunks.append({
            "text": text, "source_label": f"printing.pdf, page {i}",
            "source_type": "file", "source_name": "printing.pdf",
        })

    return chunks


def test_retrieval_precision_recall_mrr_on_labeled_corpus():
    chunks = _labeled_corpus()
    collection, _ = vectorstore.new_session_collection()
    vectorstore.add_chunks(collection, chunks)

    # A query clearly about the ML topic -- every ml.pdf chunk is
    # "relevant", every whales.pdf/printing.pdf chunk is not.
    query = "How do neural networks learn from data during training?"
    hits = vectorstore.retrieve_candidates(collection, [query], k_per_query=5)
    retrieved_labels = [h["source_label"] for h in hits]

    relevant_labels = {c["source_label"] for c in chunks if c["source_name"] == "ml.pdf"}

    metrics = evaluation.evaluate_retrieval(retrieved_labels, relevant_labels, k=5)

    # With three topically distinct documents and an unambiguous query,
    # a working embedding-based retriever should surface mostly (ideally
    # entirely) ml.pdf chunks in the top 5, with the single best match
    # ranked first or very near it.
    assert metrics["precision_at_k"] >= 0.6, (
        f"Precision@5 was only {metrics['precision_at_k']} -- retrieval may not be "
        f"distinguishing topics correctly. Retrieved: {retrieved_labels}"
    )
    assert metrics["mrr"] >= 0.5, (
        f"MRR was only {metrics['mrr']} -- the first relevant result ranked too low. "
        f"Retrieved: {retrieved_labels}"
    )


def test_retrieval_does_not_confuse_two_unrelated_topics():
    """A second, independent check: a query about whales shouldn't pull
    in machine-learning or printing-press chunks either."""
    chunks = _labeled_corpus()
    collection, _ = vectorstore.new_session_collection()
    vectorstore.add_chunks(collection, chunks)

    query = "What do we know about whale migration and populations?"
    hits = vectorstore.retrieve_candidates(collection, [query], k_per_query=5)
    retrieved_labels = [h["source_label"] for h in hits]
    relevant_labels = {c["source_label"] for c in chunks if c["source_name"] == "whales.pdf"}

    metrics = evaluation.evaluate_retrieval(retrieved_labels, relevant_labels, k=5)
    assert metrics["precision_at_k"] >= 0.6, (
        f"Precision@5 was only {metrics['precision_at_k']} for an unambiguous whale query. "
        f"Retrieved: {retrieved_labels}"
    )
