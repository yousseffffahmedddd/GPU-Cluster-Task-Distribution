"""Client layer that simulates concurrent users with asyncio."""

from __future__ import annotations

import asyncio
from typing import List, Optional
import uuid

from common.models import InferenceRequest, InferenceResponse
from monitoring.metrics import MetricsCollector


class LoadGenerator:
    """Generates requests and sends them to the master scheduler concurrently."""

    def __init__(self, scheduler, metrics: MetricsCollector) -> None:
        self.scheduler = scheduler
        self.metrics = metrics

    def _build_request(self, user_id: int) -> InferenceRequest:
        return InferenceRequest(
            request_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            prompt=f"User {user_id} asks a simulated LLM question.",
            metadata={"source": "load_generator"},
        )

    async def _send_one_request(self, user_id: int, semaphore: Optional[asyncio.Semaphore]) -> InferenceResponse:
        if semaphore is None:
            request = self._build_request(user_id)
            response = await self.scheduler.submit_request(request)
            self.metrics.record(response)
            return response

        async with semaphore:
            request = self._build_request(user_id)
            response = await self.scheduler.submit_request(request)
            self.metrics.record(response)
            return response

    async def run(self, concurrent_users: int, max_in_flight: Optional[int] = None) -> List[InferenceResponse]:
        """Run the load simulation.

        concurrent_users is the number of user requests created for the scenario.
        max_in_flight can be used to limit pressure on very weak machines. By
        default, all requests are launched as asyncio tasks, which can easily
        handle 1000+ simulated users locally.
        """
        semaphore = asyncio.Semaphore(max_in_flight) if max_in_flight is not None else None
        tasks = [self._send_one_request(user_id, semaphore) for user_id in range(1, concurrent_users + 1)]
        return await asyncio.gather(*tasks)
