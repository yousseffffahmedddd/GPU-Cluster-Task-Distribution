# CSE354 — Distributed LLM System
## Team Integration Guide

> **Ain Shams University · Faculty of Engineering**
> **CSE354: Distributed Computing — 2nd Semester 2025/2026**
> This guide covers the three completed core modules.
> Use it as your reference when building the Load Balancer, Master Scheduler, Client, and Main entry point.

---

## Table of Contents
1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Shared Constants](#3-shared-constants)
4. [GPUWorker API](#4-gpuworker-api)
5. [RAGRetriever API](#5-ragretriever-api)
6. [llm_generate API](#6-llm_generate-api)
7. [Result Dictionary Shape](#7-result-dictionary-shape)
8. [Ready-to-Paste Stubs](#8-ready-to-paste-stubs)
   - [LoadBalancer](#81-loadbalancer)
   - [MasterScheduler](#82-masterscheduler)
   - [LoadGenerator](#83-loadgenerator)
   - [main.py](#84-mainpy)
9. [Performance Metrics](#9-performance-metrics)
10. [Common Mistakes](#10-common-mistakes)

---

## 1. Project Structure

```
RAG_Ditributed/
├── main.py                        ← Entry point              (TODO)
├── TEAM_GUIDE.md                  ← This file
├── common/
│   ├── __init__.py
│   └── models.py                  ← Shared dataclasses       (TODO)
├── lb/
│   ├── __init__.py
│   └── load_balancer.py           ← Load Balancer            (TODO)
├── master/
│   ├── __init__.py
│   └── scheduler.py               ← Master Scheduler         (TODO)
├── workers/
│   ├── __init__.py
│   └── gpu_worker.py              ← ✅ DONE — RAG + LLM pipeline
├── rag/
│   ├── __init__.py
│   └── retriever.py               ← ✅ DONE — FAISS + nomic-embed-text
├── llm/
│   ├── __init__.py
│   └── inference.py               ← ✅ DONE — llama3.2:1b via Ollama
├── client/
│   ├── __init__.py
│   └── load_generator.py          ← 50-user simulator        (TODO)
└── tests/
    ├── __init__.py
    └── test_load.py               ← Load + fault tests       (TODO)
```

---

## 2. Environment Setup

Run this block **at the top of every Colab session** before any imports:

```python
import os, sys

PROJECT_ROOT = "/content/drive/MyDrive/RAG_Ditributed"

# Clear any stale cached modules
for key in list(sys.modules.keys()):
    if key.startswith(("llm", "rag", "workers", "lb", "master", "client", "common")):
        del sys.modules[key]

# Set correct path
sys.path = [p for p in sys.path if "RAG" not in p]
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
```

### Install Dependencies (once per session)
```bash
pip install aiohttp faiss-cpu nest_asyncio -q
```

### Start Ollama Server (once per session)
```python
import subprocess, os, time

os.environ["PATH"] += ":/usr/local/bin"
subprocess.Popen(
    ["/usr/local/bin/ollama", "serve"],
    stdout=open("ollama.log", "w"),
    stderr=open("ollama_error.log", "w")
)
time.sleep(6)
print("✅ Ollama server started")
```

---

## 3. Shared Constants

These are **already defined** inside the core modules. Do not redefine them in your files.

| Constant | Value | Used In |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | `gpu_worker.py`, `inference.py` |
| `EMBED_URL` | `http://localhost:11434/api/embeddings` | `retriever.py` |
| `MODEL` | `llama3.2:1b` | `gpu_worker.py`, `inference.py` |
| `EMBED_MODEL` | `nomic-embed-text` | `retriever.py` |
| `EMBED_DIM` | `768` | `retriever.py` (FAISS dimension) |
| `GPU_CONCURRENCY` | `4` | `gpu_worker.py` (T4 GPU cap) |
| `USER_CONCURRENCY` | `50` | `gpu_worker.py` (max concurrent users) |

---

## 4. GPUWorker API

### Import
```python
from workers.gpu_worker import GPUWorker
```

### Initialize
```python
workers = [GPUWorker(i) for i in range(4)]   # 4 logical workers on 1x T4
```

### `worker.process()` — Main Dispatch Method
```python
result = await worker.process(
    session,                                       # aiohttp.ClientSession
    request_id  = 1,                               # int — unique request ID
    prompt_obj  = {"prompt": "...", "num_predict": 100},
    retriever   = retriever                        # RAGRetriever | None
)
```

| Parameter | Type | Description |
|---|---|---|
| `session` | `aiohttp.ClientSession` | Shared HTTP session |
| `request_id` | `int` | Unique ID for this request |
| `prompt_obj` | `dict` | Must have `"prompt"` key; optional `"num_predict"` (default 150) |
| `retriever` | `RAGRetriever` or `None` | Pass `None` to skip RAG and call LLM directly |

### Properties for Load Balancer
```python
worker.active_requests   # int   — requests currently in-flight
worker.avg_latency       # float — running average latency (seconds)
worker.is_healthy        # bool  — False after 3 missed heartbeats
worker.id                # int   — worker identifier (0, 1, 2, 3)
worker.total_requests    # int   — lifetime request count
worker.total_latency     # float — lifetime latency sum
```

### Methods for Master Scheduler
```python
# Ping Ollama — auto-marks worker failed after 3 misses
is_alive = await worker.heartbeat(session)   # returns bool

# Manual control
worker.mark_failed()     # force-mark as dead (for fault simulation)
worker.mark_recovered()  # mark as back online
```

---

## 5. RAGRetriever API

### Import
```python
from rag.retriever import RAGRetriever
```

### Initialize
```python
retriever = RAGRetriever()
```

### `build_index()` — Run ONCE at startup
```python
DOCS = [
    "Paris is the capital of France.",
    "Cairo is the capital of Egypt.",
    "Neural networks learn patterns from data.",
    # Add more knowledge base documents here
]

await retriever.build_index(session, DOCS)
# Prints: [RAG] Index ready — N vectors stored.
```

> ⚠️ **Never call `build_index()` inside the request loop.** Call it once before dispatching any requests.

### `retrieve()` — Called per request
```python
chunks = await retriever.retrieve(
    session,
    query = "What is the capital of France?",
    top_k = 3              # how many relevant chunks to return
)
# Returns: list[str] — top-k most relevant text chunks
```

### Internals (for reference)
- Embeddings: `nomic-embed-text` via `POST /api/embeddings` → 768-dim vectors
- Vector store: `faiss.IndexFlatL2(768)` — cosine-like L2 search
- Chunking: documents are split on `.` into sentence-level chunks
- Graceful degradation: returns `[]` if index empty or embedding fails

---

## 6. llm_generate API

### Import
```python
from llm.inference import llm_generate
```

### Call
```python
result = await llm_generate(
    session,
    prompt      = "What is the capital of Egypt?",
    num_predict = 100,          # max tokens to generate
    temperature = 0.0           # 0 = deterministic output
)
```

### Returns
```python
{
    "response":       str,   # generated text from llama3.2:1b
    "tokens":         int,   # eval_count (tokens generated)
    "total_duration": int,   # nanoseconds (Ollama internal)
    "status":         str    # "ok" or "error: <message>"
}
```

> **Note:** You normally do **not** call this directly. `GPUWorker.process()` calls it internally as Step 2 of the RAG → LLM pipeline. Use it only for standalone LLM-only tests.

---

## 7. Result Dictionary Shape

Every call to `worker.process()` returns this exact dictionary:

```python
{
    "id":           int,    # request number
    "worker":       int,    # worker ID that handled it  (0–3)
    "prompt":       str,    # first 40 chars of the prompt
    "response":     str,    # full LLM answer text
    "latency":      float,  # end-to-end seconds (RAG + LLM)
    "tokens":       int,    # tokens generated by the LLM
    "status":       str,    # "✅ OK"  or  "❌ <error message>"
    "context_used": bool    # True if RAG returned at least 1 chunk
}
```

Use `result["status"]` to detect failures:
```python
ok_count = sum(1 for r in results if "OK" in r["status"])
```

---

## 8. Ready-to-Paste Stubs

### 8.1 LoadBalancer

**File:** `lb/load_balancer.py`

```python
from workers.gpu_worker import GPUWorker

class LoadBalancer:
    def __init__(self, workers: list, strategy: str = "least_conn"):
        self.workers   = workers
        self.strategy  = strategy
        self._rr_index = 0

    def get_worker(self) -> GPUWorker:
        healthy = [w for w in self.workers if w.is_healthy]
        if not healthy:
            raise RuntimeError("No healthy workers available")

        if self.strategy == "round_robin":
            w = healthy[self._rr_index % len(healthy)]
            self._rr_index += 1
            return w

        elif self.strategy == "least_conn":
            return min(healthy, key=lambda w: w.active_requests)

        elif self.strategy == "load_aware":
            return min(healthy, key=lambda w: w.active_requests + w.avg_latency * 0.1)

        return healthy[0]
```

### 8.2 MasterScheduler

**File:** `master/scheduler.py`

```python
import asyncio
from workers.gpu_worker import GPUWorker

class MasterScheduler:
    def __init__(self, workers: list):
        self.workers = workers

    async def monitor_heartbeats(self, session, interval: int = 5):
        """Run as background task — detects and marks failed workers."""
        while True:
            for w in self.workers:
                alive = await w.heartbeat(session)
                status = "✅ healthy" if alive else f"⚠️  miss {w._missed_heartbeats}/3"
                print(f"  [Scheduler] Worker {w.id}: {status}")
            await asyncio.sleep(interval)

    def log_stats(self):
        """Print per-worker performance snapshot."""
        print("\n📋 WORKER STATS")
        print("─" * 50)
        for w in self.workers:
            print(f"  Worker {w.id}: {w.total_requests:>4} reqs | "
                  f"avg {w.avg_latency:.2f}s | healthy={w.is_healthy}")

    async def dispatch(self, session, request_id, prompt_obj, retriever, load_balancer):
        """Pick a worker via load balancer and process the request."""
        worker = load_balancer.get_worker()
        return await worker.process(session, request_id, prompt_obj, retriever)
```

### 8.3 LoadGenerator

**File:** `client/load_generator.py`

```python
import asyncio, aiohttp, time, statistics

NUM_USERS = 50

PROMPTS = [
    {"prompt": "What is the capital of France?",         "num_predict": 30},
    {"prompt": "What is the capital of Egypt?",          "num_predict": 20},
    {"prompt": "Explain neural networks briefly.",       "num_predict": 80},
    {"prompt": "What is FAISS used for?",                "num_predict": 60},
    {"prompt": "What GPU is used for ML in the cloud?",  "num_predict": 40},
    {"prompt": "Summarize the French Revolution.",       "num_predict": 200},
    {"prompt": "What are pros and cons of renewables?",  "num_predict": 150},
    {"prompt": "Who wrote Romeo and Juliet?",            "num_predict": 20},
    {"prompt": "Explain the difference between RAM/ROM.","num_predict": 100},
    {"prompt": "Name one planet in our solar system.",   "num_predict": 10},
]

async def run_load_test(workers, retriever, num_users=NUM_USERS):
    start = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [
            workers[i % len(workers)].process(
                session, i + 1,
                PROMPTS[i % len(PROMPTS)],
                retriever=retriever
            )
            for i in range(num_users)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - start

    results   = [r for r in results if isinstance(r, dict)]
    latencies = [r["latency"] for r in results]
    ok        = sum(1 for r in results if "OK" in r["status"])
    s         = sorted(latencies)

    print(f"\n{'='*50}")
    print(f"📊 LOAD TEST — {num_users} concurrent users")
    print(f"{'='*50}")
    print(f"  Successful   : {ok}/{num_users}")
    print(f"  Wall time    : {elapsed:.2f}s")
    print(f"  Avg latency  : {statistics.mean(latencies):.2f}s")
    print(f"  P50/P90/P99  : {s[int(.50*len(s))]:.2f}s / "
          f"{s[int(.90*len(s))]:.2f}s / {s[int(.99*len(s))]:.2f}s")
    print(f"  Throughput   : {ok/elapsed:.2f} req/s")
    print(f"{'='*50}")
    return results
```

### 8.4 main.py

**File:** `main.py`

```python
import asyncio, aiohttp, nest_asyncio
import os, sys
nest_asyncio.apply()

PROJECT_ROOT = "/content/drive/MyDrive/RAG_Ditributed"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from workers.gpu_worker      import GPUWorker
from rag.retriever           import RAGRetriever
from lb.load_balancer        import LoadBalancer
from master.scheduler        import MasterScheduler
from client.load_generator   import run_load_test

DOCS = [
    "Paris is the capital of France. It is known for the Eiffel Tower.",
    "Cairo is the capital of Egypt. It lies on the Nile River.",
    "Neural networks are inspired by the human brain and learn from data.",
    "FAISS is a library by Meta for fast similarity search of dense vectors.",
    "The T4 GPU by NVIDIA is used for machine learning inference in the cloud.",
]

async def main():
    # 1. Init components
    workers   = [GPUWorker(i) for i in range(4)]
    retriever = RAGRetriever()
    balancer  = LoadBalancer(workers, strategy="least_conn")
    scheduler = MasterScheduler(workers)

    async with aiohttp.ClientSession() as session:
        # 2. Build RAG index once
        await retriever.build_index(session, DOCS)

        # 3. Start heartbeat monitor in background
        monitor = asyncio.create_task(
            scheduler.monitor_heartbeats(session, interval=5)
        )

        # 4. Run load test
        results = await run_load_test(workers, retriever, num_users=50)

        # 5. Print worker stats
        scheduler.log_stats()

        # 6. Cancel monitor
        monitor.cancel()

asyncio.run(main())
```

---

## 9. Performance Metrics

Your tests must report the following (per project spec):

| Metric | Formula |
|---|---|
| Avg / Min / Max latency | `statistics.mean / min / max(latencies)` |
| P50 / P90 / P99 | `sorted(lat)[int(0.50/0.90/0.99 * N)]` |
| Throughput | `ok_count / total_wall_clock_time` (req/s) |
| Token rate | `sum(r["tokens"] for r in results) / elapsed` (tok/s) |
| Worker distribution | `Counter(r["worker"] for r in results)` |
| GPU utilization | `!nvidia-smi` polled every 2s during test |
| Failure recovery time | Time from `mark_failed()` to task reassigned |

---

## 10. Common Mistakes

| # | Mistake | Fix |
|---|---|---|
| 1 | Calling `build_index()` inside the request loop | Call it **once** before dispatching any requests |
| 2 | Redefining `GPU_SEMAPHORE` or `USER_SEMAPHORE` | These are module-level singletons — never redefine |
| 3 | Missing `return_exceptions=True` in `gather()` | One failure kills all tasks without it |
| 4 | Not filtering `is_healthy` in load balancer | Will dispatch to dead workers and hang |
| 5 | Forgetting Drive mount | Run `from google.colab import drive; drive.mount('/content/drive')` first |
| 6 | Wrong `sys.path` | Always insert `/content/drive/MyDrive/RAG_Ditributed`, not `/content` |
| 7 | Stale module cache | Clear with `del sys.modules[key]` for all project modules at session start |

---

*CSE354 Distributed Computing Project — Ain Shams University, Faculty of Engineering — 2025/2026*
