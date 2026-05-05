# llm/inference.py
import aiohttp
import asyncio

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2:1b"


async def llm_generate(
    session: aiohttp.ClientSession,
    prompt: str,
    num_predict: int = 150,
    temperature: float = 0.0
) -> dict:
    """
    Async call to llama3.2:1b via Ollama.
    Returns { response, eval_count, total_duration } or error dict.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict}
    }
    try:
        async with session.post(
            OLLAMA_URL, json=payload,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            data = await resp.json()
            return {
                "response":       data.get("response", "").strip(),
                "tokens":         data.get("eval_count", 0),
                "total_duration": data.get("total_duration", 0),
                "status":         "ok"
            }
    except Exception as e:
        return {"response": "", "tokens": 0, "status": f"error: {e}"}


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    async def _test():
        async with aiohttp.ClientSession() as session:
            r = await llm_generate(session, "What is the capital of Egypt?", num_predict=20)
            print(r)
    asyncio.run(_test())