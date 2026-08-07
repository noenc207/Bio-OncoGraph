import os
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc, calibration_curve
import seaborn as sns

from biooncograph.data.dataset import BioOncoGraphDataset
from biooncograph.models.gnn import BioOncoGraphGNN
from biooncograph.engine.evaluator import Evaluator

logger = logging.getLogger(__name__)

def plot_roc(y_true, y_prob, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def plot_pr(y_true, y_prob, output_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
    plt.close()

def plot_calibration(y_true, y_prob, output_dir):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Starting evaluation with config:\n{OmegaConf.to_yaml(cfg)}")
    
    device = torch.device(cfg.get("eval", {}).get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = cfg.get("eval", {}).get("output_dir", "eval_output")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Initializing dataset...")
    test_dataset = BioOncoGraphDataset(cfg.dataset, split="test")
    test_loader = test_dataset.get_dataloader(batch_size=cfg.eval.batch_size, shuffle=False)
    
    logger.info("Initializing model and loading checkpoint...")
    model = BioOncoGraphGNN(cfg.model).to(device)
    checkpoint = torch.load(cfg.eval.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    
    logger.info("Evaluating...")
    evaluator = Evaluator(model, test_loader, device=device)
    metrics, y_true, y_prob, y_pred = evaluator.evaluate()
    
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate plots
    if len(np.unique(y_true)) == 2: # Binary classification plots
        plot_roc(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob, output_dir)
        plot_pr(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob, output_dir)
        plot_calibration(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob, output_dir)
    plot_confusion_matrix(y_true, y_pred, output_dir)
    
    logger.info(f"Evaluation plots saved to {output_dir}")

if __name__ == "__main__":
    main()
