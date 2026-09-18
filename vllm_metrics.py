#!/usr/bin/env python3

import re
import sys
import time

import requests
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console(force_terminal=True)
RESET = "\x1b[0m"

# KV usage color thresholds, expressed as the vLLM metric value (0-1).
KV_GREEN = 0.80
KV_YELLOW = 0.92
KV_RED = 0.98


def get_metric(text: str, metric_name: str) -> list[float]:
    """Return all numeric values for a metric line in vLLM metrics text."""
    pattern = rf"^{re.escape(metric_name)}(?:\{{.*?\}})?\s+([0-9eE+.-]+)$"
    values = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue

        match = re.match(pattern, line)
        if match:
            values.append(float(match.group(1)))

    return values


def first(text: str, metric: str) -> float:
    values = get_metric(text, metric)
    return values[0] if values else 0.0


def total(text: str, metric: str) -> float:
    return sum(get_metric(text, metric))


def get_serving_model(base: str) -> str:
    """Fetch the serving model root path from the vLLM /v1/models endpoint."""
    try:
        resp = requests.get(f"{base}/v1/models", timeout=5)
        data = resp.json()

        for item in data.get("data", []):
            root = item.get("root", "")
            if root:
                return root

        return "?"
    except Exception:
        return "?"


def cache_tokens(text: str) -> str:
    for line in text.splitlines():
        if "vllm:cache_config_info{" not in line:
            continue

        match = re.search(
            r'kv_cache_size_tokens="([^"]+)"',
            line,
        )

        if match:
            return match.group(1)

    return "?"


def kv_cache_color(value: float) -> str:
    if value >= KV_RED:
        return "\x1b[31m"   # red: critical / nearly full
    if value >= KV_YELLOW:
        return "\x1b[33m"   # yellow: medium / high usage
    return "\x1b[32m"       # green: normal / healthy usage


def dashboard(metrics_text: str) -> Table:
    running = first(
        metrics_text,
        "vllm:num_requests_running",
    )
    waiting = first(
        metrics_text,
        "vllm:num_requests_waiting",
    )

    kv_usage = (
        first(
            metrics_text,
            "vllm:kv_cache_usage_perc",
        )
        * 100
    )

    prompt_tokens = first(
        metrics_text,
        "vllm:prompt_tokens_total",
    )

    gen_tokens = first(
        metrics_text,
        "vllm:generation_tokens_total",
    )

    prefix_hits = first(
        metrics_text,
        "vllm:prefix_cache_hits_total",
    )
    prefix_queries = first(
        metrics_text,
        "vllm:prefix_cache_queries_total",
    )

    hit_rate = (
        prefix_hits
        / max(prefix_queries, 1)
    ) * 100

    #
    # Token-level prefix cache share: fraction of prompt tokens the
    # server did not have to re-read (cached KV reuse).
    #

    prompt_cached = first(
        metrics_text,
        "vllm:prompt_tokens_cached_total",
    )

    cached_share = (
        prompt_cached
        / max(prompt_tokens, 1)
    ) * 100

    req_success = total(
        metrics_text,
        "vllm:request_success_total",
    )

    #
    # TTFT
    #

    ttft_sum = first(
        metrics_text,
        "vllm:time_to_first_token_seconds_sum",
    )

    ttft_count = first(
        metrics_text,
        "vllm:time_to_first_token_seconds_count",
    )

    avg_ttft = (
        ttft_sum / ttft_count
        if ttft_count
        else 0
    )

    #
    # E2E
    #

    e2e_sum = first(
        metrics_text,
        "vllm:e2e_request_latency_seconds_sum",
    )

    e2e_count = first(
        metrics_text,
        "vllm:e2e_request_latency_seconds_count",
    )

    avg_e2e = (
        e2e_sum / e2e_count
        if e2e_count
        else 0
    )

    #
    # TPOT
    #

    tpot_sum = first(
        metrics_text,
        "vllm:request_time_per_output_token_seconds_sum",
    )

    tpot_count = first(
        metrics_text,
        "vllm:request_time_per_output_token_seconds_count",
    )

    avg_tpot_ms = (
        tpot_sum / tpot_count * 1000
        if tpot_count
        else 0
    )

    #
    # PREFILL TPS
    #

    prefill_tokens = first(
        metrics_text,
        "vllm:request_prefill_kv_computed_tokens_sum",
    )

    prefill_time = first(
        metrics_text,
        "vllm:request_prefill_time_seconds_sum",
    )

    prefill_tps = (
        prefill_tokens / prefill_time
        if prefill_time
        else 0
    )

    #
    # DECODE TPS
    #

    decode_time = first(
        metrics_text,
        "vllm:request_decode_time_seconds_sum",
    )

    decode_tps = (
        gen_tokens / decode_time
        if decode_time
        else 0
    )

    #
    # Memory
    #

    rss = (
        first(
            metrics_text,
            "process_resident_memory_bytes",
        )
        / 1024**3
    )

    virt = (
        first(
            metrics_text,
            "process_virtual_memory_bytes",
        )
        / 1024**3
    )

    table = Table(
        title="vLLM Runtime Dashboard",
    )

    table.add_column("Metric", overflow="ellipsis")
    table.add_column("Value", overflow="ellipsis")

    table.add_row(
        "Requests Running",
        f"{running:.0f}",
    )

    table.add_row(
        "Requests Waiting",
        f"{waiting:.0f}",
    )

    kv_capacity = cache_tokens(metrics_text)
    kv_capacity_color = kv_cache_color(kv_usage / 100.0)

    table.add_row(
        "KV Cache",
        f"{kv_capacity_color}{kv_usage:.1f}% ({kv_capacity} tokens){RESET}",
    )

    table.add_row(
        "KV Capacity",
        f"{kv_capacity} tokens",
    )

    table.add_row(
        "Prefix Cache Hit",
        f"{hit_rate:.1f}%",
    )

    table.add_row(
        "Prompt Tokens Cached",
        f"{cached_share:.1f}% ({prompt_cached:,.0f}/{prompt_tokens:,.0f})",
    )

    table.add_row(
        "Successful Requests",
        f"{req_success:,.0f}",
    )

    table.add_row(
        "Average TTFT",
        f"{avg_ttft:.2f} sec",
    )

    table.add_row(
        "Average E2E",
        f"{avg_e2e:.2f} sec",
    )

    table.add_row(
        "Average TPOT",
        f"{avg_tpot_ms:.2f}ms",
    )

    table.add_row(
        "Prefill Throughput",
        f"{prefill_tps:.1f} tok/s",
    )

    table.add_row(
        "Decode Throughput",
        f"{decode_tps:.1f} tok/s",
    )

    table.add_row(
        "Prompt Tokens",
        f"{prompt_tokens:,.0f}",
    )

    table.add_row(
        "Generation Tokens",
        f"{gen_tokens:,.0f}",
    )

    table.add_row(
        "RSS Memory",
        f"{rss:.2f} GB",
    )

    table.add_row(
        "Virtual Memory",
        f"{virt:.2f} GB",
    )

    return table


def dashboard_with_model(metrics_text: str, base: str) -> Table:
    model = get_serving_model(base)
    console.print(f"[center][green]{model}[/green][/]")
    return dashboard(metrics_text)


VALID_MODES = {"--once", "--preview"}

if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in VALID_MODES):
    print(
        "Usage: python vllm_metrics.py http://host:port [--once|--preview]",
    )
    sys.exit(1)

base = sys.argv[1].rstrip("/")

metrics_url = (
    f"{base}/metrics"
)

if len(sys.argv) >= 3 and sys.argv[2] in ("--preview", "--once"):
    metrics_text = requests.get(metrics_url, timeout=5).text
    console.print(dashboard_with_model(metrics_text, base))
    sys.exit(0)

with Live(
    refresh_per_second=1,
    console=console,
) as live:

    while True:

        try:
            metrics_text = requests.get(
                metrics_url,
                timeout=5,
            ).text

            live.update(
                dashboard_with_model(metrics_text, base)
            )

        except Exception as e:
            live.update(
                Table(
                    title=f"Connection Error: {e}",
                )
            )

        time.sleep(2)