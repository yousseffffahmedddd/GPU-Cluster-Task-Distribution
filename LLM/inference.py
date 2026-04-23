import time

def run_llm(query,context):
    #sim delay
    time.sleep(0.2)

    return f"LLM answer to '{query}' using [{context}]"