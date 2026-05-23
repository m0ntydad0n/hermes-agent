"""Tests for single-query session completion bookkeeping."""

from types import SimpleNamespace

from cli import _close_single_query_session


class _SessionDB:
    def __init__(self):
        self.calls = []

    def end_session(self, session_id, end_reason):
        self.calls.append((session_id, end_reason))


def test_close_single_query_session_marks_completed_agent_session():
    db = _SessionDB()
    cli = SimpleNamespace(_session_db=db, agent=SimpleNamespace(session_id="session-1"), session_id="session-1")

    _close_single_query_session(cli, {"failed": False, "interrupted": False})

    assert db.calls == [("session-1", "single_query_complete")]


def test_close_single_query_session_marks_failed_agent_session():
    db = _SessionDB()
    cli = SimpleNamespace(_session_db=db, agent=SimpleNamespace(session_id="session-2"), session_id="session-2")

    _close_single_query_session(cli, {"failed": True})

    assert db.calls == [("session-2", "single_query_failed")]
