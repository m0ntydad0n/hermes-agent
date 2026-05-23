"""Tests for tools/local_llm_router_tool.py."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from tools.local_llm_router_tool import local_llm_router_tool


def test_local_llm_router_emits_review_packet_telemetry(monkeypatch, tmp_path):
    worker = tmp_path / "local-worker"
    worker.write_text("#!/bin/sh\nexit 0\n")
    worker.chmod(0o755)
    hermes_home = tmp_path / "hermes-home"

    def fake_run(cmd, **kwargs):
        assert cmd == [str(worker), "extract", "-"]
        assert kwargs["input"] == "private rows stay local"
        return SimpleNamespace(returncode=0, stdout='{"rows": []}', stderr="")

    monkeypatch.setenv("LOCAL_WORKER_BIN", str(worker))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = json.loads(
        local_llm_router_tool(
            {"action": "extract", "text": "private rows stay local", "raw_context_seen_by_cloud": False},
            task_id="cloud-session-1",
        )
    )

    assert payload["ok"] is True
    packet = payload["router_packet"]
    assert packet["router_packet_id"].startswith("router_")
    assert packet["consumer_session_id"] == "cloud-session-1"
    assert packet["producer_profile"] == "local-extractor"
    assert packet["action"] == "extract"
    assert packet["raw_input_estimated_tokens"] > 0
    assert packet["summary_estimated_tokens"] > 0
    assert packet["kept_local_tokens"] == packet["raw_input_estimated_tokens"]
    assert packet["completion"]["ended_at"]
    assert packet["completion"]["end_reason"] == "success"
    assert packet["completion"]["returncode"] == 0

    log_rows = (hermes_home / "router_packets.jsonl").read_text().strip().splitlines()
    assert len(log_rows) == 1
    assert json.loads(log_rows[0])["router_packet_id"] == packet["router_packet_id"]


def test_local_llm_router_records_timeout_completion(monkeypatch, tmp_path):
    worker = tmp_path / "local-worker"
    worker.write_text("#!/bin/sh\nexit 0\n")
    worker.chmod(0o755)

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=[str(worker), "draft", "-"], timeout=12)

    monkeypatch.setenv("LOCAL_WORKER_BIN", str(worker))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setattr(subprocess, "run", fake_run)

    payload = json.loads(local_llm_router_tool({"action": "draft", "text": "facts"}, task_id="cloud-session-2"))

    assert payload["ok"] is False
    assert payload["router_packet"]["completion"]["end_reason"] == "timeout"
    assert payload["router_packet"]["completion"]["ended_at"]
