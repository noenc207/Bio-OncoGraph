import numpy as np

def run_classification():
    data = np.random.rand(10, 5)
    labels = np.random.randint(0, 2, 10)
    print(f"Classification run on data shape {data.shape} with labels {labels}")

if __name__ == "__main__":
    run_classification()
