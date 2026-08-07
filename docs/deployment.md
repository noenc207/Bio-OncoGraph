# Deployment

Deploying BioOncoGraph models to production allows for real-time inference and integration into larger clinical or research pipelines. We provide several deployment strategies.

## 1. Local/Server Deployment (Python API)

For simple applications, you can load the model directly within a Python backend (e.g., FastAPI or Flask).

```python
# server.py (FastAPI example)
from fastapi import FastAPI
import biooncograph as bog
from pydantic import BaseModel

app = FastAPI()
model = bog.models.BioModel.from_pretrained('my-trained-model')
model.eval()
graph = bog.datasets.load_production_graph()

class Query(BaseModel):
    drug_id: str
    target_id: str

@app.post("/predict")
def predict(query: Query):
    q = bog.Query(
        subject_type='drug', subject_id=query.drug_id,
        relation='targets',
        object_type='gene', object_id=query.target_id
    )
    with bog.inference_context():
        result = model.predict(graph, q)
    return {"score": float(result.score)}
```

## 2. Dockerized Serving

We strongly recommend deploying using Docker to ensure environment consistency.

```dockerfile
# Dockerfile
FROM biooncograph/biooncograph:v2.0-runtime
COPY my-trained-model /models/my-trained-model
COPY server.py /app/server.py
WORKDIR /app
RUN pip install fastapi uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 3. High-Performance Serving (TorchServe)

For high-throughput, production-grade deployment, BioOncoGraph integrates with TorchServe.

### Step A: Export the Model

First, trace or script your model using TorchScript to decouple it from Python.

```python
import torch
import biooncograph as bog

model = bog.models.BioModel.from_pretrained('my-trained-model')
model.eval()

# Example input for tracing
dummy_input = (torch.randint(0, 100, (10,)), torch.randint(0, 100, (2, 20)))
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('model.pt')
```

### Step B: Create a Model Archive (.mar)

Use the TorchServe model archiver, providing the exported model and a custom handler provided by BioOncoGraph.

```bash
torch-model-archiver --model-name bog_model --version 2.0 \
  --serialized-file model.pt \
  --handler biooncograph.deploy.torchserve_handler
```

### Step C: Run TorchServe

```bash
torchserve --start --ncs --model-store . --models bog_model=bog_model.mar
```

## Scaling Considerations

*   **Graph Storage**: For production, loading massive graphs into memory might be infeasible. Use the `biooncograph.storage.LMDBStore` backend to serve graphs directly from disk with minimal latency overhead.
*   **Batching**: Enable dynamic batching in TorchServe to maximize GPU utilization during concurrent inference requests.
