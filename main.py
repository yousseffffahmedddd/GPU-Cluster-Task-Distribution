import asyncio
import nest_asyncio

# Required for Colab to run asyncio in a Jupyter environment
nest_asyncio.apply()

from master.scheduler import GPUWorker, MasterScheduler
from lb.nginx_proxy import NginxReverseProxy
from client.load_generator import LoadGenerator
from metrics import MetricsCollector, print_summary

async def main():
    print("🚀 Initializing components...")
    
    # 1. Initialize NGINX proxy
    proxy = NginxReverseProxy()
    
    # 2. Initialize GPU Workers
    workers = [
        GPUWorker(worker_id="GPU-0", capacity=20),
        GPUWorker(worker_id="GPU-1", capacity=20),
        GPUWorker(worker_id="GPU-2", capacity=20),
        GPUWorker(worker_id="GPU-3", capacity=20),
    ]
    
    # 3. Initialize Master Scheduler (which now controls the backend)
    scheduler = MasterScheduler(workers=workers, proxy=proxy, scheduling_policy="auto")
    
    # Start worker heartbeat monitoring
    scheduler.start()
    
    # 4. Prepare for traffic (Scheduler decides the load balancer strategy)
    expected_users = 50
    decision = scheduler.prepare_for_traffic(expected_users=expected_users)
    
    # 5. Initialize Client Load Generator & Metrics
    metrics = MetricsCollector()
    load_gen = LoadGenerator(proxy=proxy, backend=scheduler, metrics=metrics)
    
    print(f"🚦 Starting Load Test with {expected_users} concurrent users...")
    print(f"   Strategy: {decision.selected_strategy}")
    
    # Run the load test
    await load_gen.run(concurrent_users=expected_users, max_in_flight=50)
    
    # Gracefully shut down the scheduler
    await scheduler.stop()
    
    # Print the results
    summary = metrics.summary(scheduler.worker_snapshots())
    print_summary(decision, summary, proxy.name)

if __name__ == "__main__":
    asyncio.run(main())
