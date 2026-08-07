import os
from experiment_utils import GPUJobScheduler

def main():
    manifest_path = "run_multiseed_manifest.json"
    scheduler = GPUJobScheduler(manifest_path=manifest_path)
    
    seeds = [42, 123, 3407, 6666, 8888]
    
    for seed in seeds:
        job_id = f"multiseed_seed_{seed}"
        command = f"python train.py --seed {seed}"
        scheduler.add_job(job_id, command)
            
    scheduler.run()

if __name__ == "__main__":
    main()
