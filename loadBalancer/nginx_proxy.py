"""External NGINX-style reverse proxy/load balancer.

This module intentionally models the Load Balancer as a separate infrastructure
component instead of hiding the routing decision inside the Master Scheduler.

Why this exists
---------------
In production, NGINX normally sits in front of backend application/model servers:

    Client  ->  NGINX Reverse Proxy  ->  Backend workers/controllers

For a university project that must still run with a single command
(`python main.py`), starting a real NGINX process would make the demo harder to
run on any machine. Therefore, this class simulates the behavior and interface
of an external NGINX proxy while staying dependency-free.

The included `nginx/nginx.conf.example` shows how the same idea would be mapped
to a real NGINX deployment.
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Protocol

from common.models import InferenceRequest, InferenceResponse


class BackendController(Protocol):
    """Interface expected from the backend controller behind the proxy.

    The proxy should not know the internal details of the Master Scheduler. It
    only needs a stable interface to ask for routable backends and to forward a
    request to one selected backend.
    """

    heartbeat_timeout: float
    queue_timeout: float

    def get_routable_workers(self) -> List[object]:
        ...

    def any_healthy_worker(self) -> bool:
        ...

    async def execute_on_worker(
        self,
        request: InferenceRequest,
        worker: object,
        selected_strategy: str,
        request_started_at: float,
    ) -> InferenceResponse:
        ...

    def clean_failure_response(
        self,
        request: InferenceRequest,
        request_started_at: float,
        error: str,
        selected_strategy: str,
    ) -> InferenceResponse:
        ...


class NginxReverseProxy:
    """External reverse proxy that chooses the backend worker.

    Supported strategies mirror the project requirements:
    - round_robin: NGINX default upstream behavior.
    - least_connections: similar to NGINX `least_conn` upstream method.
    - load_aware: project-specific extension that scores active_tasks, capacity,
      CPU, and RAM. Real NGINX Open Source does not directly inspect Python
      worker CPU/RAM, so the simulator uses worker health metadata exposed by
      the backend controller.

    The Master Scheduler may configure the active method before traffic starts,
    but actual request routing is performed here, outside the application-level
    scheduler logic.
    """

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LOAD_AWARE = "load_aware"

    VALID_STRATEGIES = {ROUND_ROBIN, LEAST_CONNECTIONS, LOAD_AWARE}

    def __init__(self, name: str = "external-nginx-proxy") -> None:
        self.name = name
        self.active_strategy = self.ROUND_ROBIN
        self._round_robin_index = 0
        self._lock = asyncio.Lock()

    def configure_strategy(self, strategy: str) -> None:
        """Configure the active upstream selection method.

        In a real NGINX deployment this would correspond to changing/reloading
        the upstream configuration. Here it is an in-memory change so the
        project remains runnable via `python main.py`.
        """
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid NGINX proxy strategy {strategy!r}. Choose from {sorted(self.VALID_STRATEGIES)}")
        self.active_strategy = strategy

    async def submit_request(self, request: InferenceRequest, backend: BackendController) -> InferenceResponse:
        """Entry point used by the Client layer.

        This simulates the client hitting an external proxy first. The proxy then
        selects a backend worker and forwards the request to the Master backend
        controller for execution. If a selected worker fails during execution,
        the proxy attempts to reroute until the scheduler queue timeout expires.
        """
        start = time.perf_counter()
        deadline = start + backend.queue_timeout
        last_error: Optional[str] = None

        while True:
            workers = backend.get_routable_workers()
            if not workers:
                if not backend.any_healthy_worker():
                    return backend.clean_failure_response(
                        request,
                        start,
                        "NGINX circuit breaker: no healthy upstream workers are available",
                        self.active_strategy,
                    )

                if time.perf_counter() >= deadline:
                    return backend.clean_failure_response(
                        request,
                        start,
                        "NGINX upstream timeout: workers stayed busy too long",
                        self.active_strategy,
                    )

                await asyncio.sleep(0.001)
                continue

            worker = await self._select_worker(workers)

            try:
                return await backend.execute_on_worker(
                    request=request,
                    worker=worker,
                    selected_strategy=self.active_strategy,
                    request_started_at=start,
                )
            except Exception as exc:  # defensive; execute_on_worker should normally return a clean response
                last_error = str(exc)
                if time.perf_counter() >= deadline:
                    return backend.clean_failure_response(
                        request,
                        start,
                        f"NGINX failed to reroute request before timeout: {last_error}",
                        self.active_strategy,
                    )
                await asyncio.sleep(0.001)

    async def _select_worker(self, workers: List[object]) -> object:
        """Select one worker using the configured external proxy strategy."""
        if self.active_strategy == self.ROUND_ROBIN:
            async with self._lock:
                worker = workers[self._round_robin_index % len(workers)]
                self._round_robin_index += 1
                return worker

        if self.active_strategy == self.LEAST_CONNECTIONS:
            return min(workers, key=lambda worker: worker.active_tasks)

        if self.active_strategy == self.LOAD_AWARE:
            return min(workers, key=self._load_score)

        # Defensive fallback. Validation in configure_strategy should prevent this.
        return workers[0]

    @staticmethod
    def _load_score(worker: object) -> float:
        """Weighted score used by the project-specific load-aware strategy.

        Formula:
            score = active_ratio * 0.60 + cpu * 0.25 + ram * 0.15

        Lower score wins. Workers that are dead, stale, or full are filtered by
        the Master backend before reaching this method.
        """
        active_ratio = worker.active_tasks / max(worker.capacity, 1)
        cpu = worker.get_cpu_utilization()
        ram = worker.get_ram_utilization()
        return (active_ratio * 0.60) + (cpu * 0.25) + (ram * 0.15)
