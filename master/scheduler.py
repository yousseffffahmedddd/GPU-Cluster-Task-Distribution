"""Master scheduler and mock GPU worker.

Corrected distributed architecture
----------------------------------
The Load Balancer is no longer treated as an application-level scheduling helper.
It is represented by an external NGINX-style reverse proxy in `lb/nginx_proxy.py`.

Runtime flow:
    Client Load Generator
        -> External NGINX Reverse Proxy
        -> Master Scheduler backend controller
        -> Mock GPU Worker

The Master Scheduler is still important, but its responsibility is different:
- choose/configure the proxy strategy before a traffic batch starts;
- maintain the global worker view;
- detect failed workers using heartbeat timestamps;
- execute the selected request on the worker chosen by the proxy;
- return clean responses when the system is overloaded or unavailable.
"""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import suppress
from typing import List, Optional

from common.models import InferenceRequest, InferenceResponse, SchedulingDecision, WorkerSnapshot
from lb.nginx_proxy import NginxReverseProxy


class MockGPUWorker:
    """Temporary mock with the same interface expected from a real worker.

    Future real-worker interface:
        worker.id
        worker.is_alive
        worker.active_tasks
        worker.last_heartbeat
        worker.capacity
        worker.has_capacity()
        worker.get_cpu_utilization()
        worker.get_ram_utilization()
        await worker.process(request)

    The mock simulates processing delay, active tasks, capacity, heartbeat, and
    optional failure. It does not run a real LLM/RAG pipeline.
    """

    def __init__(
        self,
        worker_id: str,
        capacity: int = 25,
        min_delay: float = 0.015,
        max_delay: float = 0.060,
        failure_probability: float = 0.0,
    ) -> None:
        self.id = worker_id
        self.capacity = capacity
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.failure_probability = failure_probability

        self.is_alive = True
        self.last_heartbeat = time.time()
        self.active_tasks = 0
        self.processed_requests = 0
        self.failed_requests = 0

    def heartbeat(self) -> None:
        """Simulate a heartbeat message from this worker."""
        if self.is_alive:
            self.last_heartbeat = time.time()

    def fail(self) -> None:
        """Manually mark this worker as failed."""
        self.is_alive = False

    def recover(self) -> None:
        """Manually recover this worker."""
        self.is_alive = True
        self.last_heartbeat = time.time()

    def has_capacity(self) -> bool:
        """Return True when the worker can accept another request."""
        return self.is_alive and self.active_tasks < self.capacity

    def get_cpu_utilization(self) -> float:
        """Return simulated CPU utilization between 0 and 1."""
        active_ratio = self.active_tasks / max(self.capacity, 1)
        noise = random.uniform(0.02, 0.12)
        return min(1.0, active_ratio * 0.85 + noise)

    def get_ram_utilization(self) -> float:
        """Return simulated RAM utilization between 0 and 1."""
        active_ratio = self.active_tasks / max(self.capacity, 1)
        base_model_memory = 0.25
        noise = random.uniform(0.01, 0.08)
        return min(1.0, base_model_memory + active_ratio * 0.60 + noise)

    async def process(self, request: InferenceRequest) -> str:
        """Simulate GPU worker processing.

        Replace only this method later with the real RAG + LLM pipeline.
        """
        if not self.is_alive:
            self.failed_requests += 1
            raise RuntimeError(f"Worker {self.id} is down")

        if self.active_tasks >= self.capacity:
            self.failed_requests += 1
            raise RuntimeError(f"Worker {self.id} is overloaded")

        self.active_tasks += 1
        try:
            if random.random() < self.failure_probability:
                self.fail()
                raise RuntimeError(f"Worker {self.id} failed during processing")

            await asyncio.sleep(random.uniform(self.min_delay, self.max_delay))
            self.processed_requests += 1
            return f"Mock answer for request {request.request_id} from {self.id}"
        except Exception:
            self.failed_requests += 1
            raise
        finally:
            self.active_tasks = max(0, self.active_tasks - 1)

    def snapshot(self, heartbeat_timeout: float) -> WorkerSnapshot:
        """Return monitoring data for this worker."""
        now = time.time()
        heartbeat_age = now - self.last_heartbeat
        return WorkerSnapshot(
            worker_id=self.id,
            is_alive=self.is_alive and heartbeat_age <= heartbeat_timeout,
            active_tasks=self.active_tasks,
            capacity=self.capacity,
            last_heartbeat_age_sec=heartbeat_age,
            cpu_utilization=self.get_cpu_utilization(),
            ram_utilization=self.get_ram_utilization(),
            processed_requests=self.processed_requests,
            failed_requests=self.failed_requests,
        )


class MasterScheduler:
    """Backend controller behind the external NGINX-style proxy."""

    AUTO_POLICY = "auto"

    def __init__(
        self,
        workers: List[MockGPUWorker],
        proxy: NginxReverseProxy,
        scheduling_policy: str = AUTO_POLICY,
        heartbeat_interval: float = 0.5,
        heartbeat_timeout: float = 2.0,
        queue_timeout: float = 5.0,
    ) -> None:
        valid_policies = {self.AUTO_POLICY, *NginxReverseProxy.VALID_STRATEGIES}
        if scheduling_policy not in valid_policies:
            raise ValueError(f"Invalid scheduling_policy={scheduling_policy!r}. Choose from {sorted(valid_policies)}")

        self.workers = workers
        self.proxy = proxy
        self.scheduling_policy = scheduling_policy
        self.active_strategy: Optional[str] = None
        self.last_decision: Optional[SchedulingDecision] = None

        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.queue_timeout = queue_timeout

        self._stop_event: Optional[asyncio.Event] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

    def prepare_for_traffic(self, expected_users: int) -> SchedulingDecision:
        """Master decision point before traffic reaches backend workers.

        The Master chooses the strategy. The external proxy performs the actual
        per-request routing.
        """
        if self.scheduling_policy != self.AUTO_POLICY:
            selected = self.scheduling_policy
            reason = f"Fixed project configuration: configure external NGINX proxy to use {selected}."
        elif expected_users <= 100:
            selected = NginxReverseProxy.ROUND_ROBIN
            reason = "Low traffic: configure external NGINX proxy with Round Robin."
        elif expected_users <= 500:
            selected = NginxReverseProxy.LEAST_CONNECTIONS
            reason = "Medium traffic: configure external NGINX proxy with Least Connections."
        else:
            selected = NginxReverseProxy.LOAD_AWARE
            reason = "High traffic: configure external NGINX proxy with Load-Aware routing."

        self.active_strategy = selected
        self.proxy.configure_strategy(selected)
        self.last_decision = SchedulingDecision(
            policy_name=self.scheduling_policy,
            selected_strategy=selected,
            expected_users=expected_users,
            reason=reason,
        )
        return self.last_decision

    def start(self) -> None:
        """Start the async heartbeat monitor."""
        self._stop_event = asyncio.Event()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop heartbeat monitoring safely."""
        if self._stop_event is not None:
            self._stop_event.set()

        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task

    async def _heartbeat_loop(self) -> None:
        """Simulate repeated worker heartbeat updates."""
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            for worker in self.workers:
                worker.heartbeat()
            self.detect_failed_workers()
            await asyncio.sleep(self.heartbeat_interval)

    def detect_failed_workers(self) -> None:
        """Mark workers as down if heartbeat is older than the timeout."""
        now = time.time()
        for worker in self.workers:
            if worker.is_alive and (now - worker.last_heartbeat > self.heartbeat_timeout):
                worker.fail()

    def simulate_worker_failure(self, worker_id: str) -> None:
        """Used in the demo to prove that the proxy skips failed upstreams."""
        worker = self._find_worker(worker_id)
        if worker is not None:
            worker.fail()

    def recover_worker(self, worker_id: str) -> None:
        """Recover a previously failed worker."""
        worker = self._find_worker(worker_id)
        if worker is not None:
            worker.recover()

    def _find_worker(self, worker_id: str) -> Optional[MockGPUWorker]:
        for worker in self.workers:
            if worker.id == worker_id:
                return worker
        return None

    async def submit_request(self, request: InferenceRequest) -> InferenceResponse:
        """Compatibility entry point.

        The preferred runtime path is Client -> NginxReverseProxy.submit_request()
        -> MasterScheduler.execute_on_worker(). This method is kept so older code
        can still call the scheduler directly, but it still goes through the
        external proxy object.
        """
        if self.active_strategy is None:
            self.prepare_for_traffic(expected_users=1)
        return await self.proxy.submit_request(request, backend=self)

    def get_routable_workers(self) -> List[MockGPUWorker]:
        """Return workers that the external proxy may route to."""
        self.detect_failed_workers()
        now = time.time()
        routable = []
        for worker in self.workers:
            heartbeat_fresh = (now - worker.last_heartbeat) <= self.heartbeat_timeout
            if worker.is_alive and heartbeat_fresh and worker.has_capacity():
                routable.append(worker)
        return routable

    def any_healthy_worker(self) -> bool:
        """True if at least one worker is alive, even if it is busy."""
        self.detect_failed_workers()
        now = time.time()
        return any(
            worker.is_alive and (now - worker.last_heartbeat <= self.heartbeat_timeout)
            for worker in self.workers
        )

    async def execute_on_worker(
        self,
        request: InferenceRequest,
        worker: MockGPUWorker,
        selected_strategy: str,
        request_started_at: float,
    ) -> InferenceResponse:
        """Execute a request on the worker chosen by the external proxy."""
        try:
            if self.active_strategy is None:
                self.prepare_for_traffic(expected_users=1)

            if not worker.is_alive:
                raise RuntimeError(f"Selected upstream {worker.id} is already down")

            result = await worker.process(request)
            latency_ms = (time.perf_counter() - request_started_at) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                success=True,
                worker_id=worker.id,
                message=result,
                latency_ms=latency_ms,
                strategy=selected_strategy,
            )
        except Exception as exc:
            # Mark the worker down if it failed during execution. The proxy can
            # then select a different upstream on its next loop iteration.
            if "failed" in str(exc).lower() or "down" in str(exc).lower():
                worker.fail()
            raise

    def clean_failure_response(
        self,
        request: InferenceRequest,
        request_started_at: float,
        error: str,
        selected_strategy: str,
    ) -> InferenceResponse:
        """Return a clean error response instead of crashing the program."""
        latency_ms = (time.perf_counter() - request_started_at) * 1000
        return InferenceResponse(
            request_id=request.request_id,
            success=False,
            worker_id=None,
            message="Request failed cleanly",
            latency_ms=latency_ms,
            error=error,
            strategy=selected_strategy,
        )

    def worker_snapshots(self) -> List[WorkerSnapshot]:
        """Return all worker health states for reporting."""
        return [worker.snapshot(self.heartbeat_timeout) for worker in self.workers]
