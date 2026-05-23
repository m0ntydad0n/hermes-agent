"""Tests for local-router effectiveness telemetry summaries."""

from agent.router_effectiveness import analyze_router_effectiveness


def _session(
    sid,
    *,
    profile="default",
    started_at=1000.0,
    api_call_count=1,
    tool_call_count=0,
    input_tokens=0,
    output_tokens=0,
    cache_read_tokens=0,
    reasoning_tokens=0,
    estimated_cost_usd=0.0,
    actual_cost_usd=0.0,
    ended_at: float | None = 1100.0,
    end_reason: str | None = "complete",
    model="gpt-test",
):
    return {
        "id": sid,
        "profile": profile,
        "started_at": started_at,
        "api_call_count": api_call_count,
        "tool_call_count": tool_call_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "reasoning_tokens": reasoning_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "actual_cost_usd": actual_cost_usd,
        "ended_at": ended_at,
        "end_reason": end_reason,
        "model": model,
    }


def test_analyze_router_effectiveness_splits_local_and_cloud_sessions():
    sessions = [
        _session("cloud-before", started_at=100.0, api_call_count=8, input_tokens=800, cache_read_tokens=200, output_tokens=80),
        _session("local-one", profile="local-extractor", started_at=200.0, api_call_count=1, input_tokens=100, output_tokens=10),
        _session("cloud-after", started_at=300.0, api_call_count=4, input_tokens=400, cache_read_tokens=100, output_tokens=40),
    ]

    report = analyze_router_effectiveness(sessions)

    assert report["totals"]["tracked_sessions"] == 3
    assert report["local"]["sessions"] == 1
    assert report["cloud"]["sessions"] == 2
    assert report["local"]["total_tokens"] == 110
    assert report["post_local"]["local_session_share"] == 0.5
    assert report["post_local"]["local_token_share"] == 110 / (110 + 540)
    assert report["before_after"]["pre_local_cloud"]["median_api_calls"] == 8
    assert report["before_after"]["post_local_cloud"]["median_api_calls"] == 4


def test_analyze_router_effectiveness_surfaces_cost_and_completion_gaps():
    sessions = [
        _session("local-open", profile="local-writer", started_at=100.0, input_tokens=1000, output_tokens=100, ended_at=None, end_reason=None),
        _session("cloud-zero-cost", started_at=200.0, input_tokens=2000, output_tokens=200, estimated_cost_usd=0.0, actual_cost_usd=0.0),
    ]

    report = analyze_router_effectiveness(sessions)

    assert report["instrumentation"]["cost_fields_populated"] is False
    assert report["instrumentation"]["zero_cost_sessions"] == 2
    assert report["instrumentation"]["local_sessions_missing_end"] == 1
    assert "cost" in report["recommendations"][0].lower()


def test_analyze_router_effectiveness_includes_latency_by_api_class():
    sessions = [_session("s1", input_tokens=10, output_tokens=5)]
    latencies = [
        {"time_s": 7.0, "api_calls": 1},
        {"time_s": 9.0, "api_calls": 1},
        {"time_s": 40.0, "api_calls": 3},
        {"time_s": 120.0, "api_calls": 6},
    ]

    report = analyze_router_effectiveness(sessions, gateway_latencies=latencies)

    assert report["latency"]["all"]["count"] == 4
    assert report["latency"]["by_api_call_class"]["1"]["median_s"] == 8.0
    assert report["latency"]["by_api_call_class"]["2-3"]["median_s"] == 40.0
    assert report["latency"]["by_api_call_class"]["4+"]["median_s"] == 120.0


def test_analyze_router_effectiveness_summarizes_router_packets():
    sessions = [
        _session("cloud-consumer", started_at=300.0, input_tokens=400, output_tokens=40),
        _session("local-producer", profile="local-extractor", started_at=250.0, input_tokens=50, output_tokens=20),
    ]
    packets = [
        {
            "router_packet_id": "router_abc",
            "consumer_session_id": "cloud-consumer",
            "producer_profile": "local-extractor",
            "raw_input_estimated_tokens": 1000,
            "summary_estimated_tokens": 120,
            "kept_local_tokens": 1000,
            "exposed_to_cloud_tokens": 120,
            "completion": {"end_reason": "success", "duration_s": 7.5},
        },
        {
            "router_packet_id": "router_unlinked",
            "producer_profile": "local-writer",
            "raw_input_estimated_tokens": 200,
            "summary_estimated_tokens": 80,
            "kept_local_tokens": 0,
            "exposed_to_cloud_tokens": 80,
            "completion": {"end_reason": "error", "duration_s": 5.0},
        },
    ]

    report = analyze_router_effectiveness(sessions, router_packets=packets)

    packets_report = report["router_packets"]
    assert packets_report["packets"] == 2
    assert packets_report["linked_packets"] == 1
    assert packets_report["packets_missing_linkage"] == 1
    assert packets_report["raw_input_estimated_tokens"] == 1200
    assert packets_report["kept_local_tokens"] == 1000
    assert packets_report["exposed_to_cloud_tokens"] == 200
    assert packets_report["summary_compression_ratio"] == 200 / 1200
    assert packets_report["completion_end_reasons"] == {"success": 1, "error": 1}
