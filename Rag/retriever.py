# rag/retriever.py
import asyncio
import aiohttp
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("Run: pip install faiss-cpu")

EMBED_URL   = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM   = 768   # nomic-embed-text output dimension


class RAGRetriever:
    def __init__(self):
        self.index  = faiss.IndexFlatL2(EMBED_DIM)
        self.chunks: list[str] = []

    # ── Offline index build ───────────────────────────────────────────────────
    async def build_index(self, session: aiohttp.ClientSession, documents: list[str]):
        """Chunk documents and embed them into FAISS at startup."""
        print(f"[RAG] Building FAISS index for {len(documents)} documents …")
        all_chunks  = []
        all_vectors = []

        for doc in documents:
            # simple sentence-level chunking (~200 chars)
            sentences = [s.strip() for s in doc.split(".") if len(s.strip()) > 10]
            for chunk in sentences:
                vec = await self._embed(session, chunk)
                if vec is not None:
                    all_chunks.append(chunk)
                    all_vectors.append(vec)

        if all_vectors:
            matrix = np.array(all_vectors, dtype="float32")
            self.index.add(matrix)
            self.chunks = all_chunks
            print(f"[RAG] Index ready — {self.index.ntotal} vectors stored.")
        else:
            print("[RAG] ⚠️  No vectors embedded; RAG will return empty context.")

    # ── Online query retrieval ────────────────────────────────────────────────
    async def retrieve(
        self, session: aiohttp.ClientSession, query: str, top_k: int = 3
    ) -> list[str]:
        """Embed query → FAISS search → return top-k text chunks."""
        if self.index.ntotal == 0:
            return []
        vec = await self._embed(session, query)
        if vec is None:
            return []
        q = np.array([vec], dtype="float32")
        _, indices = self.index.search(q, min(top_k, self.index.ntotal))
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    # ── Embedding helper (nomic-embed-text via Ollama) ───────────────────────
    async def _embed(
        self, session: aiohttp.ClientSession, text: str
    ) -> list[float] | None:
        payload = {"model": EMBED_MODEL, "prompt": text}
        try:
            async with session.post(
                EMBED_URL, json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                data = await resp.json()
                return data.get("embedding")
        except Exception as e:
            print(f"[RAG] Embed error: {e}")
            return None