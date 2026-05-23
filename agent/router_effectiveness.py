"""Local-router effectiveness telemetry helpers.

This module keeps the analysis pure: callers pass session rows already loaded
from one or more Hermes state databases plus optional gateway latency rows.
It returns JSON-serializable aggregates suitable for CLI reports, cron jobs, or
one-off local analysis without exposing raw prompts or message contents.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

DEFAULT_LOCAL_PROFILES = frozenset({"local-coder", "local-extractor", "local-writer"})


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _total_tokens(session: dict[str, Any]) -> int:
    return (
        _as_int(session.get("input_tokens"))
        + _as_int(session.get("cache_read_tokens"))
        + _as_int(session.get("cache_write_tokens"))
        + _as_int(session.get("output_tokens"))
        + _as_int(session.get("reasoning_tokens"))
    )


def _prompt_exposure(session: dict[str, Any]) -> int:
    return (
        _as_int(session.get("input_tokens"))
        + _as_int(session.get("cache_read_tokens"))
        + _as_int(session.get("cache_write_tokens"))
    )


def _median(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    return float(statistics.median(vals)) if vals else 0.0


def _percentile(values: Iterable[float], p: float) -> float:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - k) + vals[hi] * (k - lo))


def _summarize_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    api_calls = [_as_int(s.get("api_call_count")) for s in sessions]
    total_tokens = sum(_total_tokens(s) for s in sessions)
    prompt_exposure = sum(_prompt_exposure(s) for s in sessions)
    return {
        "sessions": len(sessions),
        "api_calls": sum(api_calls),
        "tool_calls": sum(_as_int(s.get("tool_call_count")) for s in sessions),
        "input_tokens": sum(_as_int(s.get("input_tokens")) for s in sessions),
        "cache_read_tokens": sum(_as_int(s.get("cache_read_tokens")) for s in sessions),
        "cache_write_tokens": sum(_as_int(s.get("cache_write_tokens")) for s in sessions),
        "output_tokens": sum(_as_int(s.get("output_tokens")) for s in sessions),
        "reasoning_tokens": sum(_as_int(s.get("reasoning_tokens")) for s in sessions),
        "total_tokens": total_tokens,
        "prompt_exposure_tokens": prompt_exposure,
        "estimated_cost_usd": sum(_as_float(s.get("estimated_cost_usd")) for s in sessions),
        "actual_cost_usd": sum(_as_float(s.get("actual_cost_usd")) for s in sessions),
        "median_api_calls": _median(api_calls),
        "max_api_calls": max(api_calls, default=0),
        "tokens_per_api_call": total_tokens / max(1, sum(api_calls)),
        "cache_read_share_of_prompt": (
            sum(_as_int(s.get("cache_read_tokens")) for s in sessions) / prompt_exposure
            if prompt_exposure
            else 0.0
        ),
    }


def _latency_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_as_float(row.get("time_s")) for row in rows]
    return {
        "count": len(values),
        "median_s": _median(values),
        "p90_s": _percentile(values, 90),
        "p95_s": _percentile(values, 95),
        "max_s": max(values, default=0.0),
    }


def _summarize_router_packets(packets: list[dict[str, Any]]) -> dict[str, Any]:
    raw_tokens = sum(_as_int(p.get("raw_input_estimated_tokens")) for p in packets)
    summary_tokens = sum(_as_int(p.get("summary_estimated_tokens")) for p in packets)
    exposed_tokens = sum(_as_int(p.get("exposed_to_cloud_tokens")) for p in packets)
    kept_tokens = sum(_as_int(p.get("kept_local_tokens")) for p in packets)
    durations = [_as_float((p.get("completion") or {}).get("duration_s")) for p in packets]
    return {
        "packets": len(packets),
        "linked_packets": sum(1 for p in packets if p.get("consumer_session_id")),
        "packets_missing_linkage": sum(1 for p in packets if not p.get("consumer_session_id")),
        "raw_input_estimated_tokens": raw_tokens,
        "summary_estimated_tokens": summary_tokens,
        "kept_local_tokens": kept_tokens,
        "exposed_to_cloud_tokens": exposed_tokens,
        "summary_compression_ratio": exposed_tokens / raw_tokens if raw_tokens else 0.0,
        "median_duration_s": _median(durations),
        "completion_end_reasons": dict(Counter(str((p.get("completion") or {}).get("end_reason") or "null") for p in packets)),
        "producer_profiles": dict(Counter(str(p.get("producer_profile") or "unknown") for p in packets)),
    }


def _api_class(api_calls: Any) -> str:
    calls = _as_int(api_calls)
    if calls <= 1:
        return "1"
    if calls <= 3:
        return "2-3"
    return "4+"


def _is_local_session(session: dict[str, Any], local_profiles: set[str]) -> bool:
    profile = str(session.get("profile") or "")
    provider = str(session.get("billing_provider") or "").lower()
    model = str(session.get("model") or "").lower()
    return (
        profile in local_profiles
        or provider in {"local", "ollama", "custom"}
        or model.endswith("-local")
    )


def analyze_router_effectiveness(
    sessions: Iterable[dict[str, Any]],
    *,
    gateway_latencies: Iterable[dict[str, Any]] | None = None,
    router_packets: Iterable[dict[str, Any]] | None = None,
    local_profiles: Iterable[str] = DEFAULT_LOCAL_PROFILES,
) -> dict[str, Any]:
    """Summarize local-router/prep effectiveness from session telemetry.

    The function intentionally uses session-level aggregates only. It never
    reads raw message content, prompts, tool outputs, paths, URLs, or secrets.
    """

    profile_set = set(local_profiles)
    tracked = [dict(s) for s in sessions if _total_tokens(s) > 0 or _as_int(s.get("api_call_count")) > 0]
    local = [s for s in tracked if _is_local_session(s, profile_set)]
    cloud = [s for s in tracked if s not in local]

    first_local_started = min(
        (_as_float(s.get("started_at")) for s in local if s.get("started_at") is not None),
        default=None,
    )
    if first_local_started is None:
        pre_local_cloud = cloud
        post_local_cloud: list[dict[str, Any]] = []
        post_local_all: list[dict[str, Any]] = []
    else:
        pre_local_cloud = [s for s in cloud if _as_float(s.get("started_at")) < first_local_started]
        post_local_cloud = [s for s in cloud if _as_float(s.get("started_at")) >= first_local_started]
        post_local_all = [s for s in tracked if _as_float(s.get("started_at")) >= first_local_started]

    totals = _summarize_sessions(tracked)
    local_summary = _summarize_sessions(local)
    cloud_summary = _summarize_sessions(cloud)
    post_all_summary = _summarize_sessions(post_local_all)

    profile_summaries = {
        profile: _summarize_sessions([s for s in tracked if (s.get("profile") or "default") == profile])
        for profile in sorted({str(s.get("profile") or "default") for s in tracked})
    }
    model_counts = Counter(str(s.get("model") or "unknown") for s in local)

    cost_values = [_as_float(s.get("estimated_cost_usd")) + _as_float(s.get("actual_cost_usd")) for s in tracked]
    local_missing_end = sum(1 for s in local if not s.get("ended_at") or not s.get("end_reason"))

    latency_rows = [dict(r) for r in (gateway_latencies or [])]
    packet_rows = [dict(p) for p in (router_packets or [])]
    packet_summary = _summarize_router_packets(packet_rows)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in latency_rows:
        by_class[_api_class(row.get("api_calls"))].append(row)

    recommendations: list[str] = []
    if not any(value > 0 for value in cost_values):
        recommendations.append("Populate estimated/actual cost telemetry or add pricing reconciliation before claiming dollar savings.")
    if local_missing_end:
        recommendations.append("Record ended_at/end_reason consistently for local one-shot workers before making reliability claims.")
    if packet_summary["packets"] and packet_summary["packets_missing_linkage"]:
        recommendations.append("Ensure every router packet includes consumer_session_id so local prep can be tied to cloud review turns.")
    if local_summary["total_tokens"] and post_all_summary["total_tokens"]:
        local_token_share = local_summary["total_tokens"] / post_all_summary["total_tokens"]
        if local_token_share < 0.10:
            recommendations.append("Increase router adoption or record raw-kept-local tokens; local lanes are not yet displacing much cloud token volume.")
    recommendations.append("Link local prep sessions to cloud consumers with router_packet_id or parent_session_id for causal before/after analysis.")

    return {
        "totals": {"tracked_sessions": len(tracked), **totals},
        "local": local_summary,
        "cloud": cloud_summary,
        "first_local_started_at": first_local_started,
        "post_local": {
            **post_all_summary,
            "local_session_share": local_summary["sessions"] / max(1, post_all_summary["sessions"]),
            "local_token_share": local_summary["total_tokens"] / max(1, post_all_summary["total_tokens"]),
            "local_api_call_share": local_summary["api_calls"] / max(1, post_all_summary["api_calls"]),
        },
        "before_after": {
            "pre_local_cloud": _summarize_sessions(pre_local_cloud),
            "post_local_cloud": _summarize_sessions(post_local_cloud),
            "post_local_all": post_all_summary,
        },
        "profiles": profile_summaries,
        "local_models": dict(model_counts),
        "instrumentation": {
            "cost_fields_populated": any(value > 0 for value in cost_values),
            "zero_cost_sessions": sum(1 for value in cost_values if value == 0),
            "local_sessions_missing_end": local_missing_end,
            "end_reasons": dict(Counter(str(s.get("end_reason") or "null") for s in tracked)),
        },
        "latency": {
            "all": _latency_summary(latency_rows),
            "by_api_call_class": {
                label: _latency_summary(by_class.get(label, []))
                for label in ("1", "2-3", "4+")
            },
        },
        "router_packets": packet_summary,
        "recommendations": recommendations,
    }
