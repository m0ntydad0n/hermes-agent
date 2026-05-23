"""Safe local LLM router tool for profile-scoped Hermes workers.

This tool exposes the local-worker wrapper as a narrow, review-gated interface.
It is intended for Slack/community profiles that should use local models for
classification, extraction, drafting, and code triage without granting broad
host terminal access.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from hermes_constants import get_hermes_home
from tools.registry import registry


LOCAL_LLM_ROUTER_SCHEMA = {
    "name": "local_llm_router",
    "description": (
        "Route safe draft/extract/classify/code-triage work to local LLM worker profiles. "
        "Use this for private or low-cost preliminary work only. It cannot send messages, "
        "write files, deploy, trade, or take irreversible actions; outputs must be reviewed "
        "by the main/cloud agent before any side effect."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "route",
                    "classify",
                    "extract",
                    "draft",
                    "code_plan",
                    "code_review",
                    "traceback",
                    "smoke",
                    "bench",
                ],
                "description": (
                    "Local worker action. route/classify/extract use the fast extractor/router; "
                    "draft uses the local writer; code_plan/code_review/traceback use the local coder. "
                    "smoke/bench are diagnostics and should only be used when explicitly testing the router."
                ),
            },
            "text": {
                "type": "string",
                "description": "Task text, excerpt, OCR text, draft facts, diff, or traceback to pass to the local worker.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 10,
                "maximum": 600,
                "description": "Optional local-worker timeout. Defaults to 240 seconds.",
            },
            "raw_context_seen_by_cloud": {
                "type": "boolean",
                "description": (
                    "Telemetry hint only. Leave true/default for normal cloud-planner calls because "
                    "the cloud model already saw/generated the tool argument text. Set false only for "
                    "pre-cloud/local-ingress flows where raw text was withheld from the cloud and only "
                    "the compact local output will be shown to the cloud reviewer."
                ),
            },
        },
        "required": ["action"],
    },
}


_ACTION_TO_COMMAND = {
    "route": "route",
    "classify": "classify",
    "extract": "extract",
    "draft": "draft",
    "code_plan": "code-plan",
    "code_review": "code-review",
    "traceback": "traceback",
    "smoke": "smoke",
    "bench": "bench",
}

_ACTION_TO_PROFILE_ENV = {
    "route": "LOCAL_WORKER_EXTRACTOR_PROFILE",
    "classify": "LOCAL_WORKER_EXTRACTOR_PROFILE",
    "extract": "LOCAL_WORKER_EXTRACTOR_PROFILE",
    "draft": "LOCAL_WORKER_WRITER_PROFILE",
    "code_plan": "LOCAL_WORKER_CODER_PROFILE",
    "code_review": "LOCAL_WORKER_CODER_PROFILE",
    "traceback": "LOCAL_WORKER_CODER_PROFILE",
}

_PROFILE_ENV_DEFAULTS = {
    "LOCAL_WORKER_EXTRACTOR_PROFILE": "local-extractor",
    "LOCAL_WORKER_WRITER_PROFILE": "local-writer",
    "LOCAL_WORKER_CODER_PROFILE": "local-coder",
}

_LANE_PROFILE_ENVS = {
    "local-extractor": "LOCAL_WORKER_EXTRACTOR_PROFILE",
    "local-writer": "LOCAL_WORKER_WRITER_PROFILE",
    "local-coder": "LOCAL_WORKER_CODER_PROFILE",
}


def _worker_bin() -> Path:
    return Path(os.getenv("LOCAL_WORKER_BIN", "/Users/monty/local-ai/bin/local-worker")).expanduser()


def _check_local_llm_router() -> bool:
    worker = _worker_bin()
    return worker.exists() and os.access(worker, os.X_OK)


def _json_error(message: str, **extra: Any) -> str:
    payload = {"ok": False, "error": message}
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _estimate_tokens(text: str) -> int:
    """Cheap aggregate token estimate for privacy/exposure telemetry."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _build_router_packet(
    *,
    action: str,
    text: str,
    stdout: str,
    raw_context_seen_by_cloud: bool,
    started_at: str,
    ended_at: str,
    duration_s: float,
    returncode: int | None,
    end_reason: str,
    task_id: str | None,
) -> dict[str, Any]:
    raw_tokens = _estimate_tokens(text)
    summary_tokens = _estimate_tokens(stdout)
    profile_env = _ACTION_TO_PROFILE_ENV.get(action)
    producer_profile = _profile_name_for_env(profile_env) if profile_env else None
    return {
        "router_packet_id": f"router_{uuid.uuid4().hex[:16]}",
        "consumer_session_id": task_id or os.getenv("HERMES_SESSION_ID") or None,
        "producer_profile": producer_profile,
        "action": action,
        "created_at": started_at,
        "completed_at": ended_at,
        "raw_context_seen_by_cloud": raw_context_seen_by_cloud,
        "raw_input_estimated_tokens": raw_tokens,
        "summary_estimated_tokens": summary_tokens,
        "kept_local_tokens": 0 if raw_context_seen_by_cloud else raw_tokens,
        "exposed_to_cloud_tokens": summary_tokens,
        "completion": {
            "ended_at": ended_at,
            "end_reason": end_reason,
            "returncode": returncode,
            "duration_s": round(max(0.0, duration_s), 3),
        },
    }


def _append_router_packet(packet: dict[str, Any]) -> None:
    """Persist packet metadata only; never raw input or local-worker stdout."""
    path = get_hermes_home() / "router_packets.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")


def _profile_name_for_env(env_name: str) -> str:
    return os.getenv(env_name, _PROFILE_ENV_DEFAULTS[env_name])


def _safe_profile_model(profile_name: str) -> dict[str, Any]:
    """Return non-secret model metadata for a local worker profile."""
    cfg = Path.home() / ".hermes" / "profiles" / profile_name / "config.yaml"
    meta: dict[str, Any] = {"profile": profile_name}
    try:
        data = yaml.safe_load(cfg.read_text()) or {}
        model = data.get("model") or {}
        # Do not expose api_key or host-local base_url in Slack results.
        meta.update(
            {
                "provider": model.get("provider"),
                "model": model.get("default") or model.get("model"),
                "context_length": model.get("context_length"),
            }
        )
    except Exception as exc:
        meta["metadata_warning"] = f"could not read profile model metadata: {type(exc).__name__}"
    return {k: v for k, v in meta.items() if v not in (None, "")}


def _local_model_metadata(action: str) -> dict[str, Any]:
    lane_models = {
        lane: _safe_profile_model(_profile_name_for_env(env_name))
        for lane, env_name in _LANE_PROFILE_ENVS.items()
    }
    payload: dict[str, Any] = {"lane_model_map": lane_models}
    env_name = _ACTION_TO_PROFILE_ENV.get(action)
    if env_name:
        profile = _profile_name_for_env(env_name)
        payload["worker_profile"] = profile
        payload["model_used"] = _safe_profile_model(profile)
    elif action in {"smoke", "bench"}:
        payload["model_used"] = "multiple local worker lanes"
    return payload


def local_llm_router_tool(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        action = str(args.get("action") or "").strip()
        if action not in _ACTION_TO_COMMAND:
            return _json_error("unsupported action", supported=sorted(_ACTION_TO_COMMAND))

        text = str(args.get("text") or "")
        if action not in {"smoke", "bench"} and not text.strip():
            return _json_error("text is required for this action", action=action)

        worker = _worker_bin()
        if not _check_local_llm_router():
            return _json_error("local-worker is not available or executable", path=str(worker))

        timeout = int(args.get("timeout_seconds") or os.getenv("LOCAL_WORKER_TIMEOUT", "240"))
        timeout = max(10, min(timeout, 600))

        cmd = [str(worker), _ACTION_TO_COMMAND[action]]
        input_text = None
        if action not in {"smoke", "bench"}:
            cmd.append("-")
            input_text = text

        env = os.environ.copy()
        env.setdefault("LOCAL_WORKER_TIMEOUT", str(timeout))

        started_iso = _utc_now_iso()
        started_mono = time.monotonic()
        raw_seen = bool(args.get("raw_context_seen_by_cloud", True))
        proc = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout + 15,
            env=env,
            check=False,
        )
        ended_iso = _utc_now_iso()
        duration_s = time.monotonic() - started_mono
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

        max_chars = 60000
        if len(stdout) > max_chars:
            stdout = stdout[:max_chars] + "\n...[truncated]"
        if len(stderr) > 12000:
            stderr = stderr[:12000] + "\n...[truncated]"

        payload = {
            "ok": proc.returncode == 0,
            "action": action,
            "command": _ACTION_TO_COMMAND[action],
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "safety": "local draft/extract/classify/triage only; main agent must review before side effects",
        }
        packet = _build_router_packet(
            action=action,
            text=text,
            stdout=stdout,
            raw_context_seen_by_cloud=raw_seen,
            started_at=started_iso,
            ended_at=ended_iso,
            duration_s=duration_s,
            returncode=proc.returncode,
            end_reason="success" if proc.returncode == 0 else "error",
            task_id=_kwargs.get("task_id"),
        )
        payload["router_packet"] = packet
        try:
            _append_router_packet(packet)
        except Exception as exc:
            payload["telemetry_warning"] = f"could not persist router packet: {type(exc).__name__}"
        payload.update(_local_model_metadata(action))
        return json.dumps(payload, ensure_ascii=False)
    except subprocess.TimeoutExpired as exc:
        ended_iso = _utc_now_iso()
        packet = _build_router_packet(
            action=str(args.get("action") or ""),
            text=str(args.get("text") or ""),
            stdout="",
            raw_context_seen_by_cloud=bool(args.get("raw_context_seen_by_cloud", True)),
            started_at=ended_iso,
            ended_at=ended_iso,
            duration_s=float(exc.timeout or 0.0),
            returncode=None,
            end_reason="timeout",
            task_id=_kwargs.get("task_id"),
        )
        payload = {
            "ok": False,
            "error": "local-worker timed out",
            "action": args.get("action"),
            "timeout_seconds": exc.timeout,
            "router_packet": packet,
        }
        try:
            _append_router_packet(packet)
        except Exception as persist_exc:
            payload["telemetry_warning"] = f"could not persist router packet: {type(persist_exc).__name__}"
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return _json_error(str(exc), kind=type(exc).__name__)


registry.register(
    name="local_llm_router",
    toolset="local_llm_router",
    schema=LOCAL_LLM_ROUTER_SCHEMA,
    handler=local_llm_router_tool,
    check_fn=_check_local_llm_router,
    emoji="🧭",
    max_result_size_chars=70000,
)
