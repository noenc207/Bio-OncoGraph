# Evaluation

Rigorous evaluation is critical in oncological research. BioOncoGraph v2.0 provides a comprehensive suite of metrics and evaluation strategies tailored for heterogeneous graphs and specific biomedical tasks.

## Supported Metrics

BioOncoGraph categorizes metrics based on the task type:

### Link Prediction (e.g., Drug-Target, Gene-Disease)
*   **AUROC** (Area Under the Receiver Operating Characteristic curve)
*   **AUPRC** (Area Under the Precision-Recall Curve): Crucial for highly imbalanced datasets common in biology.
*   **MRR** (Mean Reciprocal Rank)
*   **Hits@K**: Percentage of true positives in the top K predictions.

### Node Classification (e.g., Cell Type Annotation, Patient Stratification)
*   **Accuracy**, **Precision**, **Recall**, **F1-Score** (Macro and Micro)
*   **Confusion Matrix**

### Graph Regression (e.g., Drug Response Prediction, IC50)
*   **MSE** (Mean Squared Error), **RMSE**
*   **MAE** (Mean Absolute Error)
*   **Pearson / Spearman Correlation**

## Using the Evaluator

The `Evaluator` class provides a unified interface for computing these metrics.

```python
import biooncograph as bog
from biooncograph.engine import Evaluator

# Initialize the evaluator for link prediction
evaluator = Evaluator(task='link_prediction', metrics=['auroc', 'auprc', 'hits@10'])

# During inference
predictions = model.predict(test_loader)
labels = test_loader.get_labels()

# Compute metrics
results = evaluator.compute(predictions, labels)

print(f"Test AUROC: {results['auroc']:.4f}")
print(f"Test AUPRC: {results['auprc']:.4f}")
print(f"Test Hits@10: {results['hits@10']:.4f}")
```

## Cross-Validation Strategies

Standard random splits often lead to data leakage in biological graphs (e.g., structurally similar drugs). BioOncoGraph provides advanced splitting strategies:

1.  **Scaffold Split**: Splits drugs based on their Murcko scaffolds, testing the model's ability to generalize to novel chemical structures.
2.  **Cold-Start Split**: Ensures that nodes in the test set (e.g., new proteins or diseases) were not seen during training.
3.  **Temporal Split**: Uses historical data for training and future data for testing, mimicking real-world discovery timelines.

```python
from biooncograph.transforms import ScaffoldSplit

# Apply scaffold splitting to the drug nodes
transform = ScaffoldSplit(node_type='drug', train_ratio=0.8, val_ratio=0.1)
train_g, val_g, test_g = transform(graph)
```

## Analyzing Results

BioOncoGraph includes visualization tools to interpret evaluation results.

```python
from biooncograph.viz import plot_roc_curve, plot_pr_curve

# Generate evaluation plots
plot_roc_curve(predictions, labels, save_path='./results/roc.png')
plot_pr_curve(predictions, labels, save_path='./results/pr.png')
```

For benchmark comparisons against standard baselines, please refer to the [Benchmarks](benchmark.md) section.
