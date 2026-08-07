import time
import threading
import csv
import psutil
import subprocess
from datetime import datetime
import os

class ResourceMonitor:
    def __init__(self, log_file='resource_history.csv', interval=1.0):
        self.log_file = log_file
        self.interval = interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.epoch_start_time = None
        self.current_epoch = 0
        
        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'Timestamp', 'Epoch', 'Epoch_Duration_sec', 
                    'CPU_Usage_Percent', 'RAM_Usage_MB', 
                    'GPU_Util_Percent', 'GPU_Mem_MB'
                ])
                
    def get_gpu_stats(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                output = result.stdout.strip().split('\n')[0]
                gpu_util, gpu_mem = map(float, output.split(','))
                return gpu_util, gpu_mem
        except Exception:
            pass
        return 0.0, 0.0

    def _monitor_loop(self):
        while self.is_monitoring:
            timestamp = datetime.now().isoformat()
            cpu_usage = psutil.cpu_percent(interval=None)
            ram_usage = psutil.virtual_memory().used / (1024 * 1024)
            gpu_util, gpu_mem = self.get_gpu_stats()
            
            epoch_duration = 0.0
            if self.epoch_start_time is not None:
                epoch_duration = time.time() - self.epoch_start_time
                
            with open(self.log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, 
                    self.current_epoch, 
                    f"{epoch_duration:.2f}", 
                    f"{cpu_usage:.1f}", 
                    f"{ram_usage:.1f}", 
                    f"{gpu_util:.1f}", 
                    f"{gpu_mem:.1f}"
                ])
                
            time.sleep(self.interval)

    def start_epoch(self, epoch):
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def end_epoch(self):
        pass

    def start(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            psutil.cpu_percent(interval=None)
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()

    def stop(self):
        self.is_monitoring = False
        if self.monitor_thread is not None:
            self.monitor_thread.join()
