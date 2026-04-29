"""Load balancer routing algorithms.

Architecture note
-----------------
The LoadBalancer does NOT decide when to switch between algorithms.
That decision belongs to the MasterScheduler, before a traffic batch starts.

The LoadBalancer has one job only:
    Given a selected strategy + current worker states, choose the best worker.
"""

from __future__ import annotations

import time
from typing import List, Optional


class LoadBalancer:
    def __init__(self, master_scheduler):
        self.master_scheduler = master_scheduler

    def distribute_request(self, request):
        strategy = self.master_scheduler.strategy
        # يتم تطبيق الاستراتيجية المُحددة من قبل MasterScheduler
        if strategy == "round_robin":
            return self.round_robin(request)
        elif strategy == "least_connections":
            return self.least_connections(request)

    def select_worker(self, workers: List, selected_strategy: str, heartbeat_timeout: float) -> Optional[object]:
        """Return a worker, or None if no worker can accept work.

        Parameters
        ----------
        workers:
            Current worker objects.
        selected_strategy:
            The active strategy already selected by the Master Scheduler.
        heartbeat_timeout:
            A worker with an old heartbeat is considered unhealthy.
        """
        if selected_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Invalid selected_strategy={selected_strategy!r}. "
                f"Choose from {sorted(self.VALID_STRATEGIES)}"
            )

        candidates = self._healthy_available_workers(workers, heartbeat_timeout)
        if not candidates:
            return None

        if selected_strategy == self.ROUND_ROBIN:
            return self._round_robin(candidates)

        if selected_strategy == self.LEAST_CONNECTIONS:
            return self._least_connections(candidates)

        if selected_strategy == self.LOAD_AWARE:
            return self._load_aware(candidates)

        # Defensive fallback. Normally unreachable because of validation above.
        return candidates[0]

    @staticmethod
    def any_healthy_worker(workers: List, heartbeat_timeout: float) -> bool:
        """True if at least one worker is alive, even if it is busy."""
        now = time.time()
        return any(
            worker.is_alive and (now - worker.last_heartbeat <= heartbeat_timeout)
            for worker in workers
        )

    @staticmethod
    def _healthy_available_workers(workers: List, heartbeat_timeout: float) -> List:
        """Remove failed, stale-heartbeat, or capacity-full workers."""
        now = time.time()
        available = []

        for worker in workers:
            heartbeat_fresh = (now - worker.last_heartbeat) <= heartbeat_timeout
            if worker.is_alive and heartbeat_fresh and worker.has_capacity():
                available.append(worker)

        return available

    def _round_robin(self, workers: List) -> object:
        """Circular selection across currently available workers."""
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker

    @staticmethod
    def _least_connections(workers: List) -> object:
        """Choose the worker with the smallest active task count."""
        return min(workers, key=lambda worker: worker.active_tasks)

    def _load_aware(self, workers: List) -> object:
        """Choose the worker with the lowest weighted load score.

        Load-aware logic:
        - Failed workers are removed before scoring.
        - Workers with full capacity are removed before scoring.
        - Each remaining worker gets a score based on:
              active_tasks / capacity
              simulated CPU utilization
              simulated RAM utilization
        - The worker with the lowest score receives the request.

        Formula:
            score = active_ratio * 0.60 + cpu * 0.25 + ram * 0.15

        This is more intelligent than Least Connections because it considers
        resource pressure, not only the number of active tasks.
        """
        return min(workers, key=self._load_score)

    @staticmethod
    def _load_score(worker) -> float:
        active_ratio = worker.active_tasks / max(worker.capacity, 1)
        cpu = worker.get_cpu_utilization()
        ram = worker.get_ram_utilization()
        return (active_ratio * 0.60) + (cpu * 0.25) + (ram * 0.15)
