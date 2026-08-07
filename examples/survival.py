import numpy as np

def run_survival():
    times = np.random.rand(10) * 100
    events = np.random.randint(0, 2, 10)
    print(f"Survival analysis on {len(times)} samples.")

if __name__ == "__main__":
    run_survival()
