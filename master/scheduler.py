"""Master scheduler and mock GPU worker.

The real RAG, LLM, Hugging Face API, FAISS, and GPU code are intentionally not
implemented here. This file focuses only on distributed coordination:

- Master Scheduler chooses the active scheduling/routing strategy before traffic.
- Load Balancer applies that chosen strategy to select a worker.
- Workers expose a stable interface that real GPU/RAG/LLM workers can replace.
- Heartbeats and fault tolerance prevent routing to failed workers.
"""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import suppress
from typing import List, Optional

from common.models import InferenceRequest, InferenceResponse, SchedulingDecision, WorkerSnapshot
from lb.load_balancer import LoadBalancer


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
    optional failure. It does not run a real LLM.
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
    def __init__(self, master_node):
        self.master_node = master_node
        self.strategy = None  # سيحصل على الاستراتيجية من الـ Master Node

    def apply_strategy(self):
        """
        استلام الاستراتيجية من الـ Master Node وتنفيذها على الـ workers.
        """
        self.strategy = self.master_node.choose_strategy(workers_health=None)  # الـ workers_health هتكون معلومات الحالة الحالية للـ workers
        # تنفيذ الاستراتيجية بناءً على القيمة المحصلة من الـ Master Node
        if self.strategy == "round_robin":
            return self.round_robin()
        elif self.strategy == "least_connections":
            return self.least_connections()

    def prepare_for_traffic(self, expected_users: int) -> SchedulingDecision:
        """Master decision point before traffic reaches the load balancer.

        If scheduling_policy is fixed, the Master always selects that strategy.
        If scheduling_policy is 'auto', the Master switches strategy according
        to expected traffic volume.
        """
        if self.scheduling_policy != self.AUTO_POLICY:
            selected = self.scheduling_policy
            reason = f"Fixed project configuration: always use {selected}."
        elif expected_users <= 100:
            selected = LoadBalancer.ROUND_ROBIN
            reason = "Low traffic: Round Robin is simple and fair enough."
        elif expected_users <= 500:
            selected = LoadBalancer.LEAST_CONNECTIONS
            reason = "Medium traffic: Least Connections reacts to active tasks."
        else:
            selected = LoadBalancer.LOAD_AWARE
            reason = "High traffic: Load-Aware considers active tasks, capacity, CPU, and RAM."

        self.active_strategy = selected
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
        """Stop heartbeat monitoring safely.

        The stop event asks the loop to finish naturally. We also cancel the
        heartbeat task to avoid waiting for a sleeping background task. This is
        safer for repeated simulations in one program run.
        """
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
        """Used in the demo to prove that failed workers are skipped."""
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
        """Dispatch one request and return a safe response object."""
        start = time.perf_counter()
        deadline = start + self.queue_timeout

        if self.active_strategy is None:
            # Safe default if the caller forgot to call prepare_for_traffic().
            self.prepare_for_traffic(expected_users=1)

        assert self.active_strategy is not None

        try:
            while True:
                self.detect_failed_workers()

                worker = self.load_balancer.select_worker(
                    self.workers,
                    selected_strategy=self.active_strategy,
                    heartbeat_timeout=self.heartbeat_timeout,
                )

                if worker is not None:
                    result = await worker.process(request)
                    latency_ms = (time.perf_counter() - start) * 1000
                    return InferenceResponse(
                        request_id=request.request_id,
                        success=True,
                        worker_id=worker.id,
                        message=result,
                        latency_ms=latency_ms,
                        strategy=self.active_strategy,
                    )

                # If no worker is alive, fail immediately. If workers are alive
                # but busy, wait briefly. This simulates queueing in a real master.
                if not self.load_balancer.any_healthy_worker(self.workers, self.heartbeat_timeout):
                    return self._error_response(request, start, "No healthy worker is currently available")

                if time.perf_counter() >= deadline:
                    return self._error_response(request, start, "Queue timeout: workers stayed busy too long")

                await asyncio.sleep(0.001)

        except Exception as exc:
            return self._error_response(request, start, str(exc))

    def _error_response(self, request: InferenceRequest, start: float, error: str) -> InferenceResponse:
        latency_ms = (time.perf_counter() - start) * 1000
        return InferenceResponse(
            request_id=request.request_id,
            success=False,
            worker_id=None,
            message="Request failed cleanly",
            latency_ms=latency_ms,
            error=error,
            strategy=self.active_strategy,
        )

    def worker_snapshots(self) -> List[WorkerSnapshot]:
        """Return all worker health states for reporting."""
        return [worker.snapshot(self.heartbeat_timeout) for worker in self.workers]


class MasterNode:
    def __init__(self):
        self.selected_strategy = None  # الحقل اللي هيخزن الاستراتيجية

    def choose_strategy(self, workers_health):
        """
        يقوم باختيار الاستراتيجية بناءً على الحالة العامة للـ workers أو التحميل الحالي.
        """
        # مثال: استراتيجيات مختلفة بناءً على حالة الـ workers
        if workers_health:  # لو في عدد كافي من الـ workers الأحياء
            self.selected_strategy = "round_robin"  # على سبيل المثال
        else:
            self.selected_strategy = "least_connections"
        return self.selected_strategy