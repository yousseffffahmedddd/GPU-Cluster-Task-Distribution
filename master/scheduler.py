class Scheduler:
    def __init__(self,load_balancer):
        self.load_balancer=load_balancer

    def handle_request(self,request):
        print(f"[Scheduler] Dispatching request {request.id}")
        response=self.load_balancer.dispatch(request)
        return response