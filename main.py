from workers.gpu_worker import GPUWorker
from loadBalancer.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test
def main():
    workers=[GPUWorker(i) for i in range(4)] #4 gpus
    loadbalancer=LoadBalancer(workers)

    scheduler=Scheduler(loadbalancer)

    run_load_test(scheduler,num_users=10)

if __name__ == "__main__":
    main()