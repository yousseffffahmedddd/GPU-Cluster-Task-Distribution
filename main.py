"""Entry point.

Run from the project folder:
    python main.py
"""

from __future__ import annotations

import asyncio
import random

from client.load_generator import LoadGenerator
from lb.load_balancer import LoadBalancer
from master.scheduler import MasterScheduler, MasterNode
from monitoring.metrics import MetricsCollector, print_summary


master_node = MasterNode()
master_scheduler = MasterScheduler(master_node)

master_scheduler.apply_strategy()

load_balancer = LoadBalancer(master_scheduler)
USER_SCENARIOS = [100, 500, 1000]

# The Master Scheduler owns the switch.
# Use "auto" to let Master choose before each traffic batch:
#   100 users  -> round_robin
#   500 users  -> least_connections
#   1000 users -> load_aware
# Or set this to one fixed strategy: "round_robin", "least_connections", or "load_aware".
MASTER_SCHEDULING_POLICY = "auto"


def build_workers() -> list:
    """Create mock GPU workers with different capacities."""
    return [
        MockGPUWorker("gpu-worker-1", capacity=30, min_delay=0.015, max_delay=0.050),
        MockGPUWorker("gpu-worker-2", capacity=35, min_delay=0.020, max_delay=0.060),
        MockGPUWorker("gpu-worker-3", capacity=25, min_delay=0.025, max_delay=0.070),
        MockGPUWorker("gpu-worker-4", capacity=40, min_delay=0.015, max_delay=0.055),
        MockGPUWorker("gpu-worker-5", capacity=20, min_delay=0.020, max_delay=0.080),
    ]


async def run_one_simulation(concurrent_users: int, fault_demo: bool = False) -> None:
    workers = build_workers()

    # The load balancer is no longer configured with a strategy.
    # It only applies the strategy selected by the Master Scheduler.
    load_balancer = LoadBalancer()

    scheduler = MasterScheduler(
        workers=workers,
        load_balancer=load_balancer,
        scheduling_policy=MASTER_SCHEDULING_POLICY,
    )

    metrics = MetricsCollector()
    load_generator = LoadGenerator(scheduler=scheduler, metrics=metrics)

    scheduler.start()

    # MASTER SWITCH POINT: this happens before traffic starts.
    decision = scheduler.prepare_for_traffic(expected_users=concurrent_users)
    print(
        f"\n[Master decision before traffic] users={concurrent_users} | "
        f"selected_strategy={decision.selected_strategy} | reason={decision.reason}"
    )

    if fault_demo:
        scheduler.simulate_worker_failure("gpu-worker-3")
        print("[Fault demo] gpu-worker-3 was intentionally marked DOWN before traffic.")

    try:
        await load_generator.run(concurrent_users)
        summary = metrics.summary(scheduler.worker_snapshots())
        print_summary(decision, summary)
    finally:
        await scheduler.stop()


async def main() -> None:
    random.seed(42)

    print("Distributed LLM Request Simulation")
    print("Your part: Client + Master Scheduler + Load Balancer + Monitoring")
    print("RAG, LLM, and real GPU inference are intentionally mocked.")
    print("Architecture correction: Master chooses/switches strategy before traffic; Load Balancer only routes.\n")

    for users in USER_SCENARIOS:
        # Fault tolerance demo is shown under the heaviest traffic scenario.
        fault_demo = users == 1000
        await run_one_simulation(concurrent_users=users, fault_demo=fault_demo)


if __name__ == "__main__":
    asyncio.run(main())
