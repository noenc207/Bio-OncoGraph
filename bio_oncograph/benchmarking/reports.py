import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

def generate_latex_table(df: pd.DataFrame, filepath: str, caption: str = "", label: str = "") -> None:
    """Generate a LaTeX table from a pandas DataFrame."""
    try:
        # For pandas >= 2.0.0
        latex_str = df.style.format(precision=4).to_latex(caption=caption, label=label)
    except AttributeError:
        # For older pandas versions
        latex_str = df.to_latex(index=False, caption=caption, label=label, float_format="%.4f")
        
    with open(filepath, 'w') as f:
        f.write(latex_str)

def generate_csv_table(dfs: Dict[str, pd.DataFrame], filepath: str) -> None:
    """Combine multiple DataFrames and save to a single CSV."""
    combined = []
    for name, df in dfs.items():
        if df.empty:
            continue
        df_copy = df.copy()
        df_copy.insert(0, 'Table', name)
        combined.append(df_copy)
        
    if combined:
        final_df = pd.concat(combined, ignore_index=True)
        final_df.to_csv(filepath, index=False)
    else:
        pd.DataFrame().to_csv(filepath, index=False)

def plot_figure1(data: Dict[str, List[float]], filepath: str) -> None:
    """Generate Figure1.png (e.g., a bar chart or line plot of main results)."""
    plt.figure(figsize=(10, 6))
    for model, scores in data.items():
        plt.plot(scores, label=model, marker='o')
        
    plt.title("Figure 1: Main Performance Results")
    plt.xlabel("Epochs/Steps")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_roc_curve(fpr_tpr_data: Dict[str, Dict[str, float]], filepath: str) -> None:
    """Generate ROC plots."""
    plt.figure(figsize=(8, 8))
    for model, data in fpr_tpr_data.items():
        plt.plot(data['fpr'], data['tpr'], label=f"{model} (AUC = {data.get('auc', 0):.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    
def plot_pr_curve(precision_recall_data: Dict[str, Dict[str, float]], filepath: str) -> None:
    """Generate Precision-Recall plots."""
    plt.figure(figsize=(8, 8))
    for model, data in precision_recall_data.items():
        plt.plot(data['recall'], data['precision'], label=f"{model} (AP = {data.get('ap', 0):.3f})")
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_km_curve(km_data: Dict[str, Dict[str, float]], filepath: str) -> None:
    """Generate Kaplan-Meier plots."""
    plt.figure(figsize=(10, 6))
    for group, data in km_data.items():
        plt.step(data['time'], data['survival_prob'], where="post", label=group)
        
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('Kaplan-Meier Survival Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def generate_all_reports(
    table1_df: pd.DataFrame, 
    table2_df: pd.DataFrame, 
    fig1_data: Dict[str, List[float]],
    roc_data: Dict[str, Dict[str, Any]],
    pr_data: Dict[str, Dict[str, Any]],
    km_data: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """Generate all tables and figures for the publication."""
    os.makedirs(output_dir, exist_ok=True)
    
    generate_latex_table(table1_df, os.path.join(output_dir, 'Table1.tex'), caption="Main Results", label="tab:table1")
    generate_latex_table(table2_df, os.path.join(output_dir, 'Table2.tex'), caption="Ablation Study", label="tab:table2")
    
    generate_csv_table({
        'Table1': table1_df,
        'Table2': table2_df
    }, os.path.join(output_dir, 'paper_tables.csv'))
    
    if fig1_data:
        plot_figure1(fig1_data, os.path.join(output_dir, 'Figure1.png'))
    if roc_data:
        plot_roc_curve(roc_data, os.path.join(output_dir, 'ROC_plot.png'))
    if pr_data:
        plot_pr_curve(pr_data, os.path.join(output_dir, 'PR_plot.png'))
    if km_data:
        plot_km_curve(km_data, os.path.join(output_dir, 'KM_plot.png'))
