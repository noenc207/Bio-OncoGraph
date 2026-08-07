import os
import json
import time
import subprocess
import multiprocessing as mp
from queue import Empty
from typing import List, Dict, Any, Optional

try:
    import torch
    NUM_GPUS = torch.cuda.device_count()
except ImportError:
    NUM_GPUS = 0

class GPUJobScheduler:
    def __init__(self, manifest_path: str, max_gpus: Optional[int] = None):
        self.manifest_path = manifest_path
        self.is_cpu_only = NUM_GPUS == 0
        if max_gpus is not None:
            self.num_workers = max_gpus
        else:
            self.num_workers = mp.cpu_count() if self.is_cpu_only else max(1, NUM_GPUS)
            
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {"jobs": [], "completed": [], "failed": []}

    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=4)

    def add_job(self, job_id: str, command: str):
        if job_id not in self.manifest["completed"] and job_id not in [j["id"] for j in self.manifest["jobs"]]:
            self.manifest["jobs"].append({"id": job_id, "command": command})
            self.manifest["failed"] = [j for j in self.manifest["failed"] if j != job_id]
            self._save_manifest()

    def _worker(self, worker_id: int, queue: mp.Queue, result_queue: mp.Queue):
        while True:
            try:
                job = queue.get(timeout=3)
            except Empty:
                break
            
            job_id = job["id"]
            command = job["command"]
            
            env = os.environ.copy()
            if not self.is_cpu_only:
                gpu_id = worker_id % NUM_GPUS if NUM_GPUS > 0 else 0
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            print(f"Worker {worker_id} starting job {job_id}")
            try:
                process = subprocess.Popen(command, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                stdout, _ = process.communicate()
                
                if process.returncode == 0:
                    result_queue.put({"id": job_id, "status": "completed", "stdout": stdout})
                else:
                    result_queue.put({"id": job_id, "status": "failed", "stdout": stdout})
            except Exception as e:
                result_queue.put({"id": job_id, "status": "failed", "stdout": str(e)})

    def run(self):
        pending_jobs = [j for j in self.manifest["jobs"] if j["id"] not in self.manifest["completed"]]
        if not pending_jobs:
            print("No pending jobs to run.")
            return

        print(f"Starting {len(pending_jobs)} jobs with {self.num_workers} workers.")
        
        manager = mp.Manager()
        queue = manager.Queue()
        result_queue = manager.Queue()
        
        for job in pending_jobs:
            queue.put(job)
            
        workers = []
        for i in range(self.num_workers):
            p = mp.Process(target=self._worker, args=(i, queue, result_queue))
            p.start()
            workers.append(p)
            
        active_workers = len(workers)
        
        while active_workers > 0 or not result_queue.empty():
            try:
                result = result_queue.get(timeout=1)
                job_id = result["id"]
                if result["status"] == "completed":
                    print(f"Job {job_id} completed successfully.")
                    if job_id not in self.manifest["completed"]:
                        self.manifest["completed"].append(job_id)
                    self.manifest["jobs"] = [j for j in self.manifest["jobs"] if j["id"] != job_id]
                else:
                    print(f"Job {job_id} failed.")
                    print(f"Output: {result['stdout']}")
                    if job_id not in self.manifest["failed"]:
                        self.manifest["failed"].append(job_id)
                        
                self._save_manifest()
            except Empty:
                active_workers = sum(1 for p in workers if p.is_alive())
                
        for p in workers:
            p.join()
            
        print("All workers finished.")


def check_patient_leakage(train_patients: List[str], val_patients: List[str], test_patients: List[str]):
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)
    
    if not train_set.isdisjoint(val_set):
        raise ValueError(f"Patient leakage detected between train and val: {train_set.intersection(val_set)}")
    if not train_set.isdisjoint(test_set):
        raise ValueError(f"Patient leakage detected between train and test: {train_set.intersection(test_set)}")
    if not val_set.isdisjoint(test_set):
        raise ValueError(f"Patient leakage detected between val and test: {val_set.intersection(test_set)}")
