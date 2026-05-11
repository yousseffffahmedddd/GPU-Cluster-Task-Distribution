"""Master scheduler and real GPU worker."""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import suppress
from typing import List, Optional

from common.models import InferenceRequest, InferenceResponse, SchedulingDecision, WorkerSnapshot
from lb.nginx_proxy import NginxReverseProxy


class GPUWorker:
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
        if self.is_alive:
            self.last_heartbeat = time.time()

    def fail(self) -> None:
        self.is_alive = False

    def recover(self) -> None:
        self.is_alive = True
        self.last_heartbeat = time.time()

    def has_capacity(self) -> bool:
        return self.is_alive and self.active_tasks < self.capacity

    def get_ram_utilization(self) -> float:
        """Return simulated RAM utilization between 0 and 1."""
        active_ratio = self.active_tasks / max(self.capacity, 1)
        base_model_memory = 0.25
        noise = random.uniform(0.01, 0.08)
        return min(1.0, base_model_memory + active_ratio * 0.60 + noise)

    async def process(self, request: InferenceRequest) -> str:
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

            import aiohttp
            from LLM.inference import llm_generate
            from Rag.retriever import RAGRetriever

            if not hasattr(self, "_session") or self._session is None:
                self._session = aiohttp.ClientSession()
            if not hasattr(self, "_retriever"):
                self._retriever = RAGRetriever()

            context = ""
            try:
                chunks = await self._retriever.retrieve(self._session, request.prompt, top_k=3)
                if chunks:
                    context = "\n".join(chunks)
            except Exception:
                pass

            augmented_prompt = (
                f"Context:\n{context}\n\nQuestion: {request.prompt}"
                if context else request.prompt
            )

            result = await llm_generate(self._session, augmented_prompt)
            if result.get("status", "") != "ok":
                raise RuntimeError(f"LLM Error: {result.get('status')}")

            self.processed_requests += 1
            return result.get("response", "")
        except Exception:
            self.failed_requests += 1
            raise
        finally:
            self.active_tasks = max(0, self.active_tasks - 1)

    def snapshot(self, heartbeat_timeout: float) -> WorkerSnapshot:
        now = time.time()
        heartbeat_age = now - self.last_heartbeat
        return WorkerSnapshot(
            worker_id=self.id,
            is_alive=self.is_alive and heartbeat_age <= heartbeat_timeout,
            active_tasks=self.active_tasks,
            capacity=self.capacity,
            last_heartbeat_age_sec=heartbeat_age,
            ram_utilization=self.get_ram_utilization(),
            processed_requests=self.processed_requests,
            failed_requests=self.failed_requests,
        )


class MasterScheduler:
    AUTO_POLICY = "auto"

    def __init__(
        self,
        workers: List[GPUWorker],
        proxy: NginxReverseProxy,
        scheduling_policy: str = AUTO_POLICY,
        heartbeat_interval: float = 0.5,
        heartbeat_timeout: float = 2.0,
        queue_timeout: float = 5.0,
    ) -> None:
        valid_policies = {self.AUTO_POLICY, *NginxReverseProxy.VALID_STRATEGIES}
        if scheduling_policy not in valid_policies:
            raise ValueError(f"Invalid scheduling_policy={scheduling_policy!r}.")
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
        if self.scheduling_policy != self.AUTO_POLICY:
            selected = self.scheduling_policy
            reason = f"Fixed policy: {selected}."
        elif expected_users <= 100:
            selected = NginxReverseProxy.ROUND_ROBIN
            reason = "Low traffic: Round Robin."
        elif expected_users <= 500:
            selected = NginxReverseProxy.LEAST_CONNECTIONS
            reason = "Medium traffic: Least Connections."
        else:
            selected = NginxReverseProxy.LOAD_AWARE
            reason = "High traffic: Load-Aware routing."

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
        self._stop_event = asyncio.Event()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat_task

    async def _heartbeat_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            for worker in self.workers:
                worker.heartbeat()
            self.detect_failed_workers()
            await asyncio.sleep(self.heartbeat_interval)

    def detect_failed_workers(self) -> None:
        now = time.time()
        for worker in self.workers:
            if worker.is_alive and (now - worker.last_heartbeat > self.heartbeat_timeout):
                worker.fail()

    def simulate_worker_failure(self, worker_id: str) -> None:
        worker = self._find_worker(worker_id)
        if worker is not None:
            worker.fail()

    def recover_worker(self, worker_id: str) -> None:
        worker = self._find_worker(worker_id)
        if worker is not None:
            worker.recover()

    def _find_worker(self, worker_id: str) -> Optional[GPUWorker]:
        for worker in self.workers:
            if worker.id == worker_id:
                return worker
        return None

    async def submit_request(self, request: InferenceRequest) -> InferenceResponse:
        if self.active_strategy is None:
            self.prepare_for_traffic(expected_users=1)
        return await self.proxy.submit_request(request, backend=self)

    def get_routable_workers(self) -> List[GPUWorker]:
        self.detect_failed_workers()
        now = time.time()
        return [
            w for w in self.workers
            if w.is_alive
            and (now - w.last_heartbeat) <= self.heartbeat_timeout
            and w.has_capacity()
        ]

    def any_healthy_worker(self) -> bool:
        self.detect_failed_workers()
        now = time.time()
        return any(
            w.is_alive and (now - w.last_heartbeat <= self.heartbeat_timeout)
            for w in self.workers
        )

    async def execute_on_worker(
        self,
        request: InferenceRequest,
        worker: GPUWorker,
        selected_strategy: str,
        request_started_at: float,
    ) -> InferenceResponse:
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
        return [worker.snapshot(self.heartbeat_timeout) for worker in self.workers]