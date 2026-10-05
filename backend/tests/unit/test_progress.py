"""progress.py: listeners, stage/detail plumbing, and isolation."""

from app import evaluation, progress


def test_no_listener_is_a_no_op():
    progress.stage("extraction")
    progress.detail("anything")  # must not raise


def test_timed_stage_announces_each_stage_and_details_attach_to_it():
    events = []
    report = evaluation.EvaluationReport()
    with progress.listening(lambda stage, detail: events.append((stage, detail))):
        with report.timed_stage("extraction"):
            progress.detail("Describing Figure 1 of a.pdf (page 2)")
        with report.timed_stage("chunking"):
            pass
    assert events == [
        ("extraction", ""),
        ("extraction", "Describing Figure 1 of a.pdf (page 2)"),
        ("chunking", ""),
    ]


def test_listener_errors_never_reach_the_pipeline():
    def broken(stage, detail):
        raise ConnectionError("result backend down")
    with progress.listening(broken):
        progress.stage("extraction")  # must not raise


def test_listener_is_removed_after_the_block():
    events = []
    with progress.listening(lambda s, d: events.append(s)):
        progress.stage("a")
    progress.stage("b")
    assert events == ["a"]


def test_figure_description_reports_progress(figure_pdf, monkeypatch):
    from app import doc_parser, groq_client
    monkeypatch.setattr(groq_client, "describe_figure", lambda png, caption: "a chart")
    details = []
    with progress.listening(lambda s, d: d and details.append(d)):
        doc_parser.parse_pdf(figure_pdf)
    assert details[0] == "Describing Figure 1 of figures.pdf (page 1)"
    assert any("Figure 2" in d for d in details)
