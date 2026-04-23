import time 
from LLM.inference import run_llm
from Rag.retriever import retrieve_context

class GPUWorker:
    def __init__(self,id):
        self.id=id
    def process(self,request):
        start=time.time()
        print(f"[worker] {self.id} processing request {request.id}")

        #rag step
        context=retrieve_context(request.query)

        #llm step
        result=run_llm(request.query,context)

        latency=time.time()-start

        return{
            "id":request.id,
            "result":result,
            "latency":latency
        }