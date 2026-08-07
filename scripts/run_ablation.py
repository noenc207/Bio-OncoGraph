import os
from experiment_utils import GPUJobScheduler

def main():
    manifest_path = "run_ablation_manifest.json"
    scheduler = GPUJobScheduler(manifest_path=manifest_path)
    
    components_to_remove = ["none", "attention", "augmentation", "residual_connections"]
    
    for component in components_to_remove:
        job_id = f"ablation_without_{component}"
        command = f"python train.py --remove_component {component}"
        scheduler.add_job(job_id, command)
            
    scheduler.run()

if __name__ == "__main__":
    main()
