"""
Progress reporting for a running pipeline, without threading a callback
through every function: the Celery task installs a listener for the
duration of its run, and any code below it (pipeline stages, figure
description, rate-limit waits) just calls stage() / detail().

With no listener installed (unit tests, plain run_pipeline calls) both
are no-ops. A ContextVar rather than a global, so two pipeline runs in
different threads (eager mode serves requests from a thread pool) don't
report into each other's jobs.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable

_listener: ContextVar[Callable[[str, str], None] | None] = ContextVar("progress_listener", default=None)
_draft_listener: ContextVar[Callable[[str], None] | None] = ContextVar("draft_listener", default=None)
_current_stage: ContextVar[str] = ContextVar("progress_stage", default="")


@contextmanager
def listening(callback: Callable[[str, str], None], on_draft: Callable[[str], None] | None = None):
    """callback(stage, detail) is called on every update inside this block;
    on_draft(text), if given, on every update of the report being written."""
    token = _listener.set(callback)
    draft_token = _draft_listener.set(on_draft)
    try:
        yield
    finally:
        _listener.reset(token)
        _draft_listener.reset(draft_token)


def stage(name: str) -> None:
    """A new pipeline stage started (clears the previous stage's detail)."""
    _current_stage.set(name)
    _emit(name, "")


def detail(text: str) -> None:
    """Finer-grained news within the current stage."""
    _emit(_current_stage.get(), text)


def _emit(stage_name: str, text: str) -> None:
    callback = _listener.get()
    if callback is None:
        return
    try:
        callback(stage_name, text)
    except Exception:
        pass  # a progress update must never fail the pipeline


def draft(text: str) -> None:
    """The report as written so far, while generation streams it."""
    callback = _draft_listener.get()
    if callback is None:
        return
    try:
        callback(text)
    except Exception:
        pass  # a progress update must never fail the pipeline
