"""Monitoring and metrics collection."""

from __future__ import annotations

from collections import Counter, defaultdict
import statistics
import time
from typing import Dict, List

from common.models import InferenceResponse, SchedulingDecision, WorkerSnapshot


class MetricsCollector:
    """Collects latency, throughput, worker usage, and errors."""

    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.end_time = self.start_time
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies_ms: List[float] = []
        self.requests_per_worker = defaultdict(int)
        self.errors = Counter()

    def record(self, response: InferenceResponse) -> None:
        self.total_requests += 1
        self.end_time = time.perf_counter()
        self.latencies_ms.append(response.latency_ms)

        if response.success:
            self.successful_requests += 1
            if response.worker_id is not None:
                self.requests_per_worker[response.worker_id] += 1
        else:
            self.failed_requests += 1
            self.errors[response.error or "Unknown error"] += 1

    def summary(self, worker_snapshots: List[WorkerSnapshot]) -> Dict:
        duration = max(self.end_time - self.start_time, 0.000001)
        avg_latency = statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0
        min_latency = min(self.latencies_ms) if self.latencies_ms else 0.0
        max_latency = max(self.latencies_ms) if self.latencies_ms else 0.0
        throughput = self.total_requests / duration
        error_rate = (self.failed_requests / self.total_requests * 100) if self.total_requests else 0.0

        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": avg_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "throughput_rps": throughput,
            "error_rate_percent": error_rate,
            "requests_per_worker": dict(self.requests_per_worker),
            "errors": dict(self.errors),
            "worker_snapshots": worker_snapshots,
        }


def print_summary(decision: SchedulingDecision, summary: Dict, proxy_name: str = "external-nginx-proxy") -> None:
    """Print a clean report after each simulation."""
    print("\n" + "=" * 86)
    print(
        "Simulation summary | "
        f"master_policy={decision.policy_name} | "
        f"selected_strategy={decision.selected_strategy} | "
        f"users={decision.expected_users}"
    )
    print("=" * 86)
    print(f"External proxy      : {proxy_name}")
    print(f"Master decision     : {decision.reason}")
    print(f"Total requests      : {summary['total_requests']}")
    print(f"Successful requests : {summary['successful_requests']}")
    print(f"Failed requests     : {summary['failed_requests']}")
    print(f"Error rate          : {summary['error_rate_percent']:.2f}%")
    print(f"Average latency     : {summary['avg_latency_ms']:.2f} ms")
    print(f"Min latency         : {summary['min_latency_ms']:.2f} ms")
    print(f"Max latency         : {summary['max_latency_ms']:.2f} ms")
    print(f"Throughput          : {summary['throughput_rps']:.2f} requests/sec")

    print("\nRequests handled per worker:")
    if summary["requests_per_worker"]:
        for worker_id, count in sorted(summary["requests_per_worker"].items()):
            print(f"  - {worker_id}: {count}")
    else:
        print("  No successful worker responses.")

    print("\nWorker health snapshot:")
    for worker in summary["worker_snapshots"]:
        status = "UP" if worker.is_alive else "DOWN"
        print(
            f"  - {worker.worker_id}: {status} | "
            f"active={worker.active_tasks}/{worker.capacity} | "
            f"heartbeat_age={worker.last_heartbeat_age_sec:.2f}s | "
            f"cpu={worker.cpu_utilization * 100:.1f}% | "
            f"ram={worker.ram_utilization * 100:.1f}% | "
            f"processed={worker.processed_requests} | "
            f"failed={worker.failed_requests}"
        )

    if summary["errors"]:
        print("\nErrors:")
        for error, count in summary["errors"].items():
            print(f"  - {error}: {count}")

    print("=" * 86)
