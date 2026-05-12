# GPU Cluster Task Distribution
### Efficient Load Balancing for 100+ Concurrent LLM Requests

> **CSE354: Distributed Computing — Ain Shams University, Faculty of Engineering**
> **2nd Semester 2025/2026 | Branch: `integrated_1`**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/LLM-llama3.2%3A1b-orange)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/VectorDB-FAISS-green)](https://github.com/facebookresearch/faiss)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20T4-yellow?logo=googlecolab)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Module Descriptions](#4-module-descriptions)
   - [common/models.py](#41-commonmodelspy)
   - [LLM/inference.py](#42-llminferencepy)
   - [Rag/retriever.py](#43-ragretrieverpy)
   - [lb/nginx_proxy.py](#44-lbnginx_proxypy)
   - [master/scheduler.py](#45-masterschedulerpy)
   - [client/load_generator.py](#46-clientload_generatorpy)
   - [metrics.py](#47-metricspy)
   - [main.py](#48-mainpy)
5. [Scheduling Strategies](#5-scheduling-strategies)
6. [RAG Pipeline](#6-rag-pipeline)
7. [Fault Tolerance](#7-fault-tolerance)
8. [Quick Start](#8-quick-start)
9. [Google Colab Setup](#9-google-colab-setup)
10. [Running Tests](#10-running-tests)
11. [Performance Metrics](#11-performance-metrics)
12. [Sample Output](#12-sample-output)
13. [API Reference](#13-api-reference)
14. [Common Mistakes](#14-common-mistakes)
15. [Dependencies](#15-dependencies)
16. [References](#16-references)

---

## 1. Overview

This project implements a **distributed inference platform** that routes and executes
Large Language Model (LLM) requests across a cluster of simulated GPU worker nodes.
Each request passes through a full **RAG → LLM pipeline**:

1. User sends a natural language prompt
2. An **NGINX-style reverse proxy** routes it to the best available worker
3. The worker retrieves relevant context from a **FAISS vector database**
4. A **llama3.2:1b** model served via **Ollama** generates the final answer
5. Metrics (latency, throughput, RAM) are collected and visualised

Key capabilities:

| Feature | Detail |
|---|---|
| **Concurrency** | 100+ async requests via `asyncio.gather()` |
| **Load Balancing** | Round Robin, Least Connections, Load-Aware |
| **LLM** | llama3.2:1b via Ollama (per-worker isolated instances) |
| **RAG** | FAISS IndexFlatL2 + nomic-embed-text (768-dim) |
| **Fault Tolerance** | Heartbeat detection, auto-rerouting, circuit breaker |
| **Metrics** | Avg/min/max latency, throughput, per-worker distribution |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       CLIENT LAYER                           │
│           client/load_generator.py  (LoadGenerator)         │
│     asyncio.gather() — 100+ concurrent InferenceRequests   │
└───────────────────────────┬──────────────────────────────────┘
                            │  submit_request()
┌───────────────────────────▼──────────────────────────────────┐
│              NGINX REVERSE PROXY  (lb/nginx_proxy.py)        │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Strategy: round_robin | least_connections |        │     │
│  │            load_aware  (score = 0.70×load+0.30×ram) │     │
│  │  Circuit Breaker: returns clean failure if all down  │     │
│  │  Retry Loop: reroutes until queue_timeout=5.0s       │     │
│  └─────────────────────────────────────────────────────┘     │
└───────────────────────────┬──────────────────────────────────┘
                            │  execute_on_worker()
┌───────────────────────────▼──────────────────────────────────┐
│             MASTER SCHEDULER  (master/scheduler.py)          │
│  • Auto-selects strategy (RR ≤100 / LC ≤500 / LA 500+)      │
│  • Heartbeat loop every 0.5s — marks dead workers            │
│  • get_routable_workers() filters is_alive=False             │
└────────┬──────────────┬──────────────┬────────────┬──────────┘
         │              │              │            │
┌────────▼──┐  ┌────────▼──┐  ┌───────▼───┐  ┌────▼──────┐
│  gpu-0    │  │  gpu-1    │  │  gpu-2    │  │  gpu-3    │
│ :11434    │  │ :11435    │  │ :11436    │  │ :11437    │
│ cap=25    │  │ cap=25    │  │ cap=25    │  │ cap=25    │
└────┬──────┘  └────┬──────┘  └─────┬─────┘  └────┬──────┘
     │              │               │              │
     └──────────────┴───────────────┴──────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     RAG PIPELINE           │
              │  Rag/retriever.py          │
              │  FAISS + nomic-embed-text  │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │     LLM INFERENCE          │
              │  LLM/inference.py          │
              │  llama3.2:1b via Ollama    │
              └───────────────────────────┘
```

### Data Flow Summary

```
InferenceRequest  →  NginxReverseProxy  →  MasterScheduler
    →  GPUWorker.process()
        ├── RAGRetriever.retrieve()     (FAISS top-k chunks)
        ├── prompt augmentation         (context + question)
        └── llm_generate()             (Ollama /api/generate)
    →  InferenceResponse  →  MetricsCollector.record()
```

---

## 3. Project Structure

```
GPU-Cluster-Task-Distribution/
├── LLM/
│   ├── __init__.py
│   └── inference.py          # llm_generate() — async Ollama API, per-worker port routing
├── Rag/
│   ├── __init__.py
│   └── retriever.py          # RAGRetriever — FAISS build_index() + retrieve()
├── client/
│   ├── __init__.py
│   └── load_generator.py     # LoadGenerator — async concurrent user simulation
├── common/
│   ├── __init__.py
│   └── models.py             # Dataclasses: InferenceRequest, InferenceResponse, etc.
├── lb/
│   ├── __init__.py
│   └── nginx_proxy.py        # NginxReverseProxy — 3 strategies + circuit breaker
├── master/
│   ├── __init__.py
│   └── scheduler.py          # GPUWorker + MasterScheduler + heartbeat loop
├── workers/
│   └── __init__.py
├── metrics.py                # MetricsCollector + print_summary()
├── main.py                   # System entry point
├── requirements.txt          # Python dependencies
├── TEAM_GUIDE.md             # Developer integration guide
└── TEST.ipynb                # Google Colab full test suite
```

---

## 4. Module Descriptions

### 4.1 `common/models.py`

Shared frozen/mutable dataclasses used across all layers.

```python
@dataclass(frozen=True)
class InferenceRequest:
    request_id: str       # UUID-based unique ID
    user_id:    int       # Simulated user identifier
    prompt:     str       # Raw user question
    created_at: float     # Submission timestamp (time.time())
    metadata:   dict      # Optional extra fields

@dataclass
class InferenceResponse:
    request_id: str
    success:    bool
    message:    str         # LLM response text (if success=True)
    latency_ms: float       # Wall-clock time from dispatch to response
    worker_id:  str | None  # Which GPU node handled this request
    error:      str | None  # Error detail (if success=False)
    strategy:   str | None  # Routing strategy used

@dataclass(frozen=True)
class SchedulingDecision:
    policy_name:       str   # "auto" or explicit policy
    selected_strategy: str   # "round_robin" | "least_connections" | "load_aware"
    expected_users:    int
    reason:            str

@dataclass
class WorkerSnapshot:
    worker_id:              str
    is_alive:               bool
    active_tasks:           int
    capacity:               int
    last_heartbeat_age_sec: float
    ram_utilization:        float   # 0.0 – 1.0
    processed_requests:     int
    failed_requests:        int
```

---

### 4.2 `LLM/inference.py`

Async LLM generation via Ollama. Each worker has its own dedicated port.

```python
WORKER_PORTS = {
    "gpu-0": 11434,
    "gpu-1": 11435,
    "gpu-2": 11436,
    "gpu-3": 11437,
}
MODEL = "llama3.2:1b"

async def llm_generate(
    session:     aiohttp.ClientSession,
    prompt:      str,
    num_predict: int   = 150,
    temperature: float = 0.0,
    worker_id:   str   = None,
) -> dict:
    # Returns: {"response": str, "tokens": int, "total_duration": int, "status": "ok"|"error:..."}
```

**Key design choices:**
- `stream=False` — full response returned atomically (no streaming chunks)
- `temperature=0.0` — deterministic outputs for reproducible tests
- `timeout=300s` — generous timeout for slow T4 GPU inference
- Port isolation — killing one Ollama instance does not affect others

---

### 4.3 `Rag/retriever.py`

FAISS-backed Retrieval-Augmented Generation using `nomic-embed-text`.

```python
EMBED_URL   = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM   = 768    # nomic-embed-text output dimension

class RAGRetriever:
    async def build_index(session, documents: list[str]) -> None
    # Call ONCE at startup. Embeds all document chunks into FAISS.

    async def retrieve(session, query: str, top_k: int = 3) -> list[str]
    # Call per request. Returns top-k most relevant text chunks.
```

**Pipeline:**
1. Documents are split on `.` into sentence-level chunks (~200 chars)
2. Each chunk is embedded via `nomic-embed-text` (POST `/api/embeddings`)
3. Vectors stored in `faiss.IndexFlatL2(768)`
4. At query time: embed query → `index.search()` → return top-k chunks

> ⚠️ Never call `build_index()` inside the request loop — call it once at startup.

---

### 4.4 `lb/nginx_proxy.py`

Simulates an external NGINX upstream reverse proxy.

```python
class NginxReverseProxy:
    ROUND_ROBIN       = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LOAD_AWARE        = "load_aware"

    async def submit_request(request, backend) -> InferenceResponse
    # Entry point for all client requests.
    # Applies strategy, retries on failure, enforces queue_timeout=5.0s.

    def configure_strategy(strategy: str) -> None
    # Called by MasterScheduler.prepare_for_traffic()
```

**Load-Aware score function:**
```python
def _load_score(worker) -> float:
    active_ratio = worker.active_tasks / max(worker.capacity, 1)
    ram          = worker.get_ram_utilization()
    return (active_ratio * 0.70) + (ram * 0.30)
# Lower score = less loaded = preferred target
```

**Circuit Breaker behaviour:**
- If no routable workers: retry loop (1ms sleep) until `queue_timeout` expires
- Returns `InferenceResponse(success=False, error="NGINX circuit breaker: ...")` — no exceptions escape

---

### 4.5 `master/scheduler.py`

Contains both `GPUWorker` and `MasterScheduler`.

#### `GPUWorker`

```python
class GPUWorker:
    def __init__(self,
        worker_id:           str,
        capacity:            int   = 25,
        min_delay:           float = 0.015,
        max_delay:           float = 0.060,
        failure_probability: float = 0.0,
    )

    # Key properties
    worker.id               # str — "gpu-0" .. "gpu-3"
    worker.capacity         # int — max concurrent tasks (default 25)
    worker.active_tasks     # int — currently in-flight requests
    worker.is_alive         # bool — False after failure
    worker.processed_requests  # int — lifetime success count
    worker.failed_requests     # int — lifetime failure count

    def has_capacity() -> bool        # is_alive AND active_tasks < capacity
    def get_ram_utilization() -> float # 0.25 + active_ratio*0.60 + noise
    def fail()                         # mark as dead
    def recover()                      # mark as alive again
    async def process(request) -> str  # full RAG + LLM pipeline
    def snapshot(heartbeat_timeout) -> WorkerSnapshot
```

**RAM utilisation model:**
```
RAM = min(1.0, 0.25 + (active_tasks/capacity)×0.60 + uniform(0.01, 0.08))
      ↑                ↑                              ↑
  base model       linear load                     noise
  (weights in      component                     (±4%)
   VRAM ~25%)
```

#### `MasterScheduler`

```python
class MasterScheduler:
    def __init__(self,
        workers:             list[GPUWorker],
        proxy:               NginxReverseProxy,
        scheduling_policy:   str   = "auto",
        heartbeat_interval:  float = 0.5,
        heartbeat_timeout:   float = 2.0,
        queue_timeout:       float = 5.0,
    )

    def prepare_for_traffic(expected_users: int) -> SchedulingDecision
    # Auto-selects strategy:
    #   ≤100  users  → round_robin
    #   ≤500  users  → least_connections
    #   500+  users  → load_aware

    def start()                              # launch background heartbeat task
    async def stop()                         # cancel heartbeat, clean shutdown
    def simulate_worker_failure(worker_id)   # hard-crash for fault tests
    def recover_worker(worker_id)            # restore crashed worker
    def get_routable_workers() -> list       # alive + has_capacity
    def worker_snapshots() -> list[WorkerSnapshot]
```

---

### 4.6 `client/load_generator.py`

Async concurrent user simulator.

```python
class LoadGenerator:
    def __init__(self, proxy, backend, metrics: MetricsCollector)

    async def run(
        concurrent_users: int,
        max_in_flight:    int | None = None,  # semaphore limit
    ) -> list[InferenceResponse]
    # Launches all requests as asyncio tasks via asyncio.gather()
    # max_in_flight=8 used in fault tolerance tests to match Ollama parallelism
```

Each request builds an `InferenceRequest` with:
- `request_id`: 8-char UUID prefix
- `user_id`: sequential integer
- `prompt`: `"User {id} asks a simulated LLM question."`

---

### 4.7 `metrics.py`

```python
class MetricsCollector:
    def record(response: InferenceResponse) -> None
    # Called after every request completes

    def summary(worker_snapshots: list[WorkerSnapshot]) -> dict
    # Returns:
    # {
    #   "total_requests":       int,
    #   "successful_requests":  int,
    #   "failed_requests":      int,
    #   "avg_latency_ms":       float,
    #   "min_latency_ms":       float,
    #   "max_latency_ms":       float,
    #   "throughput_rps":       float,   # total / elapsed_seconds
    #   "error_rate_percent":   float,
    #   "requests_per_worker":  dict,
    #   "errors":               dict,    # Counter of error strings
    #   "worker_snapshots":     list,
    # }

def print_summary(decision, summary, proxy_name) -> None
# Pretty-prints the full simulation summary to stdout
```

---

### 4.8 `main.py`

System entry point. Wires all components and runs a 50-user load test by default.

```python
async def main():
    proxy     = NginxReverseProxy()
    workers   = [GPUWorker(worker_id=f"GPU-{i}", capacity=20) for i in range(4)]
    scheduler = MasterScheduler(workers=workers, proxy=proxy)
    scheduler.start()                                   # heartbeat loop

    decision  = scheduler.prepare_for_traffic(50)      # auto-selects strategy

    metrics   = MetricsCollector()
    load_gen  = LoadGenerator(proxy=proxy, backend=scheduler, metrics=metrics)

    async with aiohttp.ClientSession() as session:
        retriever = RAGRetriever()
        await retriever.build_index(session, DOCS)     # build FAISS index once

        for worker in workers:
            worker._retriever = retriever              # share index across workers

        await load_gen.run(concurrent_users=50, max_in_flight=50)

    await scheduler.stop()
    print_summary(decision, metrics.summary(scheduler.worker_snapshots()), proxy.name)
```

---

## 5. Scheduling Strategies

| Strategy | Trigger (auto mode) | Selection Logic | Use Case |
|---|---|---|---|
| `round_robin` | `expected_users ≤ 100` | Cycle workers in fixed order (modulo index, async-locked) | Low load, uniform request duration |
| `least_connections` | `expected_users ≤ 500` | `min(workers, key=lambda w: w.active_tasks)` | Medium load, variable duration |
| `load_aware` | `expected_users > 500` (default for all tests) | `min(workers, key=lambda w: w.active_tasks/cap×0.70 + ram×0.30)` | High load, capacity-aware |

To override auto-selection:
```python
scheduler = MasterScheduler(workers, proxy, scheduling_policy="round_robin")
```

---

## 6. RAG Pipeline

```
Query prompt
    │
    ▼
nomic-embed-text (768-dim)
    │  POST http://localhost:11434/api/embeddings
    ▼
FAISS IndexFlatL2.search(query_vec, top_k=3)
    │  Returns indices of closest chunks (L2 distance)
    ▼
Retrieved chunks list[str]
    │
    ▼
Augmented prompt:
    "Context:\n{chunk1}\n{chunk2}\n{chunk3}\n\nQuestion: {prompt}"
    │
    ▼
llm_generate(session, augmented_prompt, worker_id=self.id)
    │  POST http://localhost:{port}/api/generate
    ▼
LLM response text
```

**Graceful degradation:** If the FAISS index is empty or embedding fails, `retrieve()` returns `[]` and the raw prompt is sent to the LLM directly (RAG step is skipped silently).

---

## 7. Fault Tolerance

### Detection

The `MasterScheduler._heartbeat_loop()` runs as a background `asyncio.Task` every `heartbeat_interval=0.5s`. Workers whose `last_heartbeat` age exceeds `heartbeat_timeout=2.0s` are auto-marked `is_alive=False`.

### Injection (for tests)

```python
scheduler.simulate_worker_failure("gpu-3")   # hard crash
scheduler.recover_worker("gpu-3")             # restore
```

### Rerouting

`NginxReverseProxy.get_routable_workers()` filters dead workers **before every selection**. The `load_aware` strategy never assigns a request to an `is_alive=False` worker.

### Circuit Breaker

If all workers are simultaneously down:
```
NginxReverseProxy → no routable workers
    → any_healthy_worker() = False
    → return InferenceResponse(success=False,
          error="NGINX circuit breaker: no healthy upstream workers are available")
```

### Fault Tolerance Test Results

| Metric | Normal (4 workers) | Fault (gpu-3 killed) |
|---|---|---|
| Workers active | 4 | 3 |
| Total capacity | 100 slots | 75 slots (−25%) |
| Requests sent | 75 | 75 |
| Succeeded | 75 ✅ | 75 ✅ |
| Failed | 0 | 0 |
| gpu-3 requests | ~19 | **0** |
| Avg latency | ~10,800 ms | ~11,200 ms (+3.7%) |
| System survived | ✅ YES | ✅ YES |

---

## 8. Quick Start

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed and accessible at `http://localhost:11434`
- Models pulled: `llama3.2:1b` and `nomic-embed-text`

### Installation

```bash
git clone https://github.com/yousseffffahmedddd/GPU-Cluster-Task-Distribution.git
cd GPU-Cluster-Task-Distribution
git checkout integrated_1
pip install -r requirements.txt
```

### Pull Ollama models

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

### Run

```bash
python main.py
```

---

## 9. Google Colab Setup

Run these cells at the top of every Colab session:

#### Cell 1 — Clone & install

```python
!git clone https://github.com/yousseffffahmedddd/GPU-Cluster-Task-Distribution.git
%cd GPU-Cluster-Task-Distribution
!git checkout integrated_1
!pip install aiohttp faiss-cpu nest-asyncio numpy -q
```

#### Cell 2 — Install & start Ollama

```python
import subprocess, os, time

# Install Ollama
os.system("curl -fsSL https://ollama.com/install.sh | sh")

# Start Ollama server
subprocess.Popen(
    ["/usr/local/bin/ollama", "serve"],
    stdout=open("ollama.log", "w"),
    stderr=open("ollama_error.log", "w"),
)
time.sleep(6)
print("✅ Ollama started")
```

#### Cell 3 — Pull models

```python
!ollama pull llama3.2:1b
!ollama pull nomic-embed-text
```

#### Cell 4 — Run

```python
import nest_asyncio, asyncio
nest_asyncio.apply()
exec(open("main.py").read())
```

#### Fault tolerance / full test suite

Open and run `TEST.ipynb` — it handles the full setup automatically.

---

## 10. Running Tests

### Default run (50 users, auto strategy)

```bash
python main.py
```

### Custom load test

```python
from master.scheduler import GPUWorker, MasterScheduler
from lb.nginx_proxy import NginxReverseProxy
from client.load_generator import LoadGenerator
from metrics import MetricsCollector, print_summary
import asyncio, aiohttp

async def test():
    proxy     = NginxReverseProxy()
    workers   = [GPUWorker(f"gpu-{i}", capacity=25) for i in range(4)]
    scheduler = MasterScheduler(workers, proxy, scheduling_policy="load_aware")
    scheduler.start()
    decision  = scheduler.prepare_for_traffic(100)

    metrics  = MetricsCollector()
    load_gen = LoadGenerator(proxy, scheduler, metrics)

    async with aiohttp.ClientSession() as session:
        from Rag.retriever import RAGRetriever
        docs = ["Cairo is the capital of Egypt.", "Paris is the capital of France."]
        r = RAGRetriever()
        await r.build_index(session, docs)
        for w in workers: w._retriever = r
        await load_gen.run(concurrent_users=100, max_in_flight=8)

    await scheduler.stop()
    print_summary(decision, metrics.summary(scheduler.worker_snapshots()), proxy.name)

asyncio.run(test())
```

### Fault tolerance test

```python
# Kill gpu-3 at t=5s
async def fault_test():
    # ... setup as above ...
    scheduler.start()
    await asyncio.sleep(5)
    scheduler.simulate_worker_failure("gpu-3")  # hard crash
    print("💥 gpu-3 killed — load_aware now routes to gpu-0/1/2 only")
    await load_gen.run(concurrent_users=75, max_in_flight=8)
    await scheduler.stop()
```

---

## 11. Performance Metrics

All metrics are collected by `MetricsCollector` and reported via `print_summary()`.

| Metric | Formula | Where |
|---|---|---|
| Avg latency | `mean(latencies_ms)` | `metrics.summary()` |
| Min / Max latency | `min/max(latencies_ms)` | `metrics.summary()` |
| Throughput | `total_requests / elapsed_seconds` (req/s) | `metrics.summary()` |
| Error rate | `failed / total × 100` (%) | `metrics.summary()` |
| Worker distribution | `requests_per_worker` dict | `metrics.summary()` |
| RAM utilisation | `0.25 + load×0.60 + noise` | `GPUWorker.get_ram_utilization()` |
| Heartbeat age | `now - last_heartbeat` (s) | `WorkerSnapshot` |

### Interpreting Results

- **Latency 10,000–50,000 ms** — expected on single T4 GPU with 4 Ollama instances sharing VRAM
- **Throughput ~0.006–0.07 req/s** — bottlenecked by sequential Ollama inference per worker
- **RAM 25–45%** — base model weights + linear load component
- **0% error rate** — system correctly routes around failed workers

---

## 12. Sample Output

```
====================================================================================
Simulation summary | master_policy=auto | selected_strategy=load_aware | users=40
====================================================================================
External proxy      : external-nginx-proxy
Master decision     : High traffic: Load-Aware routing.
Total requests      : 40
Successful requests : 40
Failed requests     : 0
Error rate          : 0.00%
Average latency     : 36989.50 ms
Min latency         : 33550.12 ms
Max latency         : 47908.44 ms
Throughput          : 0.006 requests/sec

Requests handled per worker:
  - gpu-0: 10
  - gpu-1: 10
  - gpu-2: 10
  - gpu-3: 10

Worker health snapshot:
  - gpu-0: UP | active=0/25 | heartbeat_age=0.01s | ram=27.3% | processed=10 | failed=0
  - gpu-1: UP | active=0/25 | heartbeat_age=0.01s | ram=26.8% | processed=10 | failed=0
  - gpu-2: UP | active=0/25 | heartbeat_age=0.01s | ram=25.4% | processed=10 | failed=0
  - gpu-3: UP | active=0/25 | heartbeat_age=0.01s | ram=28.1% | processed=10 | failed=0
====================================================================================
```

**Fault tolerance run (gpu-3 killed):**

```
=================================================================
PHASE 1 — Fault run: gpu-3 killed BEFORE dispatch
          Capacity: 75 slots (3×25) — 25% power removed
=================================================================
  💥 [5.0s] gpu-3 HARD CRASHED — load_aware rerouting to gpu-0/1/2
  ✅ 75/75 ok | ❌ 0 failed | Avg: 11200.4ms | 0.067 req/s
  Distribution: {gpu-0: 25, gpu-1: 25, gpu-2: 25, gpu-3: 0}

FAULT TOLERANCE VERDICT
  Capacity after failure   : 75% (3/4 workers × 25 slots)
  Requests redistributed   : 75 — all handled by surviving workers
  Latency impact           : +400.3ms vs normal (+3.7%)
  System survived          : ✅ YES
=================================================================
```

---

## 13. API Reference

### `NginxReverseProxy`

| Method | Signature | Description |
|---|---|---|
| `configure_strategy` | `(strategy: str)` | Set routing method |
| `submit_request` | `async (request, backend) → InferenceResponse` | Route and execute one request |

### `MasterScheduler`

| Method | Signature | Description |
|---|---|---|
| `prepare_for_traffic` | `(expected_users: int) → SchedulingDecision` | Auto-select strategy |
| `start` | `()` | Launch heartbeat background task |
| `stop` | `async ()` | Stop heartbeat, clean shutdown |
| `simulate_worker_failure` | `(worker_id: str)` | Hard-crash a worker (tests) |
| `recover_worker` | `(worker_id: str)` | Restore crashed worker |
| `get_routable_workers` | `() → list[GPUWorker]` | Alive + has capacity |
| `worker_snapshots` | `() → list[WorkerSnapshot]` | Current state of all workers |

### `GPUWorker`

| Method / Property | Type | Description |
|---|---|---|
| `has_capacity()` | `bool` | `is_alive and active_tasks < capacity` |
| `get_ram_utilization()` | `float` | Simulated RAM 0.0–1.0 |
| `fail()` | — | Mark as dead |
| `recover()` | — | Mark as alive |
| `process(request)` | `async → str` | Execute RAG + LLM, returns response text |
| `snapshot(timeout)` | `WorkerSnapshot` | Read-only state snapshot |

### `RAGRetriever`

| Method | Signature | Description |
|---|---|---|
| `build_index` | `async (session, documents: list[str])` | Embed docs + build FAISS index |
| `retrieve` | `async (session, query, top_k=3) → list[str]` | Return top-k relevant chunks |

### `MetricsCollector`

| Method | Signature | Description |
|---|---|---|
| `record` | `(response: InferenceResponse)` | Record one response |
| `summary` | `(worker_snapshots) → dict` | Aggregated metrics dict |

---

## 14. Common Mistakes

| # | Mistake | Fix |
|---|---|---|
| 1 | Calling `build_index()` inside the request loop | Call it **once before** dispatching any requests |
| 2 | Forgetting `max_in_flight` on slow machines | Set `max_in_flight=8` to match real Ollama parallelism |
| 3 | Missing `return_exceptions=True` in `gather()` | One failure kills all tasks without it |
| 4 | Not filtering `is_alive` in load balancer | Will dispatch to dead workers and hang |
| 5 | Redefining Ollama port constants | Already defined in `LLM/inference.py` — do not redefine |
| 6 | `nest_asyncio.apply()` missing in Colab | Colab has a running event loop — apply before `asyncio.run()` |
| 7 | Stale sys.modules cache in Colab | Clear with `del sys.modules[k]` for all project modules |
| 8 | Worker._retriever not set before load test | Share retriever with `worker._retriever = retriever` after `build_index()` |

---

## 15. Dependencies

```
aiohttp>=3.9          # Async HTTP client for Ollama API calls
faiss-cpu>=1.7        # Facebook AI Similarity Search (vector DB)
nest-asyncio>=1.6     # Allow asyncio.run() inside Jupyter/Colab
numpy>=1.24           # Required by FAISS for float32 arrays
```

Install:
```bash
pip install -r requirements.txt
```

### External services required

| Service | Version | Purpose |
|---|---|---|
| Ollama | Latest | LLM serving (`llama3.2:1b`) + embeddings (`nomic-embed-text`) |
| llama3.2:1b | via Ollama | Language model for inference |
| nomic-embed-text | via Ollama | 768-dim text embeddings for RAG |

---

## 16. References

- [Ollama](https://ollama.com) — Local LLM serving engine
- [FAISS](https://github.com/facebookresearch/faiss) — Johnson et al., "Billion-scale similarity search with GPUs", IEEE TBD 2019
- [Lewis et al., 2020](https://arxiv.org/abs/2005.11401) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020
- [vLLM](https://github.com/vllm-project/vllm) — Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention", SOSP 2023
- [NGINX Load Balancing](https://nginx.org/en/docs/http/load_balancing.html) — Upstream module documentation
- [Python asyncio](https://docs.python.org/3/library/asyncio.html) — Async I/O framework
- [aiohttp](https://docs.aiohttp.org) — Async HTTP client/server
- CSE354 Project Specification — Ain Shams University, Faculty of Engineering, 2025/2026

---

*Ain Shams University · Faculty of Engineering · CSE354 Distributed Computing · 2025/2026*
