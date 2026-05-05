# workers/gpu_worker.py
import asyncio
import aiohttp
import time

OLLAMA_URL      = "http://localhost:11434/api/generate"
EMBED_URL       = "http://localhost:11434/api/embeddings"
MODEL           = "llama3.2:1b"
EMBED_MODEL     = "nomic-embed-text"
GPU_CONCURRENCY = 4                      # T4 GPU parallel inference cap
USER_CONCURRENCY = 50                    # max concurrent users

GPU_SEMAPHORE  = asyncio.Semaphore(GPU_CONCURRENCY)
USER_SEMAPHORE = asyncio.Semaphore(USER_CONCURRENCY)


class GPUWorker:
    def __init__(self, worker_id: int):
        self.id               = worker_id
        self.active_requests  = 0
        self.total_requests   = 0
        self.total_latency    = 0.0
        self._failed          = False
        self._missed_heartbeats = 0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.total_requests if self.total_requests > 0 else 0.0

    @property
    def is_healthy(self) -> bool:
        return not self._failed

    def mark_failed(self):
        self._failed = True

    def mark_recovered(self):
        self._failed = False
        self._missed_heartbeats = 0

    async def heartbeat(self, session: aiohttp.ClientSession) -> bool:
        """Ping Ollama; 3 missed → mark failed."""
        try:
            async with session.get(
                "http://localhost:11434/",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as r:
                if r.status == 200:
                    self._missed_heartbeats = 0
                    self.mark_recovered()
                    return True
        except Exception:
            pass
        self._missed_heartbeats += 1
        if self._missed_heartbeats >= 3:
            self.mark_failed()
        return False

    async def process(
        self,
        session: aiohttp.ClientSession,
        request_id: int,
        prompt_obj: dict,
        retriever=None          # rag.retriever.RAGRetriever instance (optional)
    ) -> dict:
        """
        Full pipeline:
          1. RAG  – embed query → FAISS top-3 context  (if retriever supplied)
          2. LLM  – llama3.2:1b with augmented prompt
        Both semaphores gate entry: USER_SEMAPHORE (50) → GPU_SEMAPHORE (4).
        """
        async with USER_SEMAPHORE:
            async with GPU_SEMAPHORE:
                self.active_requests += 1
                start = time.perf_counter()

                # ── Step 1: RAG context retrieval ────────────────────────────
                context = ""
                if retriever is not None:
                    try:
                        chunks = await retriever.retrieve(
                            session, prompt_obj["prompt"], top_k=3
                        )
                        context = "\n".join(chunks)
                    except Exception as e:
                        context = ""  # degrade gracefully

                augmented_prompt = (
                    f"Context:\n{context}\n\nQuestion: {prompt_obj['prompt']}"
                    if context else prompt_obj["prompt"]
                )

                # ── Step 2: LLM inference ────────────────────────────────────
                payload = {
                    "model": MODEL,
                    "prompt": augmented_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": prompt_obj.get("num_predict", 150)
                    }
                }
                try:
                    async with session.post(
                        OLLAMA_URL, json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as resp:
                        result  = await resp.json()
                        latency = time.perf_counter() - start
                        self.total_latency  += latency
                        self.total_requests += 1
                        self.active_requests -= 1
                        return {
                            "id": request_id, "worker": self.id,
                            "prompt":   prompt_obj["prompt"][:40],
                            "response": result.get("response", "").strip(),
                            "latency":  latency,
                            "tokens":   result.get("eval_count", 0),
                            "status":   "✅ OK",
                            "context_used": bool(context)
                        }
                except Exception as e:
                    latency = time.perf_counter() - start
                    self.total_latency  += latency
                    self.total_requests += 1
                    self.active_requests -= 1
                    return {
                        "id": request_id, "worker": self.id,
                        "prompt":   prompt_obj["prompt"][:40],
                        "response": str(e)[:40], "latency": latency,
                        "tokens": 0, "status": f"❌ {str(e)[:60]}",
                        "context_used": False
                    }