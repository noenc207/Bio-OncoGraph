import os
from experiment_utils import GPUJobScheduler

def main():
    manifest_path = "run_experiments_manifest.json"
    scheduler = GPUJobScheduler(manifest_path=manifest_path)
    
    models = ["model_a", "model_b"]
    learning_rates = ["1e-3", "1e-4"]
    
    for model in models:
        for lr in learning_rates:
            job_id = f"exp_{model}_lr{lr}"
            command = f"python train.py --model {model} --lr {lr}"
            scheduler.add_job(job_id, command)
            
    scheduler.run()

if __name__ == "__main__":
    main()
