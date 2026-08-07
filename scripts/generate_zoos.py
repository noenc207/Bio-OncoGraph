import os
import json

def generate_zoos():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    docs_dir = os.path.join(base_dir, 'docs')
    models_dir = os.path.join(base_dir, 'models')
    datasets_dir = os.path.join(base_dir, 'datasets')
    weights_dir = os.path.join(base_dir, 'weights')

    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Model Zoo Markdown
    with open(os.path.join(docs_dir, 'model_zoo.md'), 'w') as f:
        f.write("# Model Zoo\n\nThis document lists all available models.\n\n## Bio-OncoGraph Models\n\n* **Model A**: Graph-based oncology model.\n")

    # Dataset Zoo Markdown
    with open(os.path.join(docs_dir, 'dataset_zoo.md'), 'w') as f:
        f.write("# Dataset Zoo\n\nThis document lists all available datasets.\n\n## Bio-OncoGraph Datasets\n\n* **Dataset A**: Multi-omics oncology dataset.\n")

    # Manifests
    model_manifest = {"models": [{"name": "Model A", "version": "1.0.0", "path": "models/model_a"}]}
    with open(os.path.join(models_dir, 'manifest.json'), 'w') as f:
        json.dump(model_manifest, f, indent=4)

    dataset_manifest = {"datasets": [{"name": "Dataset A", "version": "1.0.0", "path": "datasets/dataset_a"}]}
    with open(os.path.join(datasets_dir, 'manifest.json'), 'w') as f:
        json.dump(dataset_manifest, f, indent=4)
        
    weight_manifest = {"weights": [{"name": "Weight A", "version": "1.0.0", "path": "weights/weight_a.pt"}]}
    with open(os.path.join(weights_dir, 'manifest.json'), 'w') as f:
        json.dump(weight_manifest, f, indent=4)

if __name__ == "__main__":
    generate_zoos()
