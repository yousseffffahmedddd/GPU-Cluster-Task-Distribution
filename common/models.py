"""Common dataclasses used across the distributed simulation project.

These models are intentionally small and stable. The real RAG/LLM worker can
reuse the same request/response objects later without changing the scheduler.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


@dataclass(frozen=True)
class InferenceRequest:
    """One simulated user request sent to the distributed inference platform."""

    request_id: str
    user_id: int
    prompt: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResponse:
    """Response returned by the scheduler after success or clean failure."""

    request_id: str
    success: bool
    message: str
    latency_ms: float
    worker_id: Optional[str] = None
    error: Optional[str] = None
    strategy: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SchedulingDecision:
    """Decision made by the Master Scheduler before traffic is sent.

    Important architecture point:
    - The Master Scheduler chooses the active routing strategy.
    - The Load Balancer only applies that strategy to pick a worker.
    """

    policy_name: str
    selected_strategy: str
    expected_users: int
    reason: str
    created_at: float = field(default_factory=time.time)


@dataclass
class WorkerSnapshot:
    """Read-only worker state used by the monitoring report."""

    worker_id: str
    is_alive: bool
    active_tasks: int
    capacity: int
    last_heartbeat_age_sec: float
    cpu_utilization: float
    ram_utilization: float
    processed_requests: int
    failed_requests: int
