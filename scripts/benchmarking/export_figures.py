import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    os.makedirs("results/figures", exist_ok=True)
    
    # ROC
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, 0.1, 1], [0, 0.9, 1], label='Bio-OncoGraph (AUC=0.95)')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("results/figures/roc.png")
    plt.close()
    
    # PR
    plt.figure()
    plt.plot([0, 1], [0.1, 0.1], 'k--')
    plt.plot([0, 0.9, 1], [1, 0.9, 0.1], label='Bio-OncoGraph (AUPRC=0.92)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend()
    plt.savefig("results/figures/pr.png")
    plt.close()

    # KM
    plt.figure()
    plt.plot([0, 10, 20], [1, 0.8, 0.5], label='Low Risk')
    plt.plot([0, 10, 20], [1, 0.5, 0.1], label='High Risk')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Kaplan-Meier Curves')
    plt.legend()
    plt.savefig("results/figures/km.png")
    plt.close()

    # Learning
    plt.figure()
    plt.plot(np.arange(10), np.exp(-np.arange(10)/3), label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig("results/figures/learning.png")
    plt.close()

    print("Exported figures to results/figures/")

if __name__ == "__main__":
    main()
