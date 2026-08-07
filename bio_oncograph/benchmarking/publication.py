import os
import json
import pandas as pd
from typing import Dict, Any, List

from .statistics import compare_models
from .leaderboard import generate_leaderboard, format_leaderboard_for_display
from .artifacts import ReproducibilityPackage
from .reports import generate_all_reports

def prepare_publication(
    results_data: List[Dict[str, Any]],
    ablation_data: List[Dict[str, Any]],
    metrics_dict: Dict[str, Any],
    baseline_model: str,
    datasets: Dict[str, str],
    fig1_data: Dict[str, List[float]],
    roc_data: Dict[str, Dict[str, Any]],
    pr_data: Dict[str, Dict[str, Any]],
    km_data: Dict[str, Dict[str, Any]],
    output_dir: str
) -> None:
    """
    Coordinates the generation of the reproducibility package, leaderboards, statistics, and reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Reproducibility Package
    repro_pkg = ReproducibilityPackage(os.path.join(output_dir, 'reproducibility'))
    repro_pkg.build(datasets)
    
    # 2. Leaderboard & Stats for Table 1
    leaderboard_df = generate_leaderboard(results_data, primary_metric='f1_score', ascending=False)
    formatted_table1 = format_leaderboard_for_display(leaderboard_df)
    
    # 3. Ablation Study for Table 2
    ablation_df = generate_leaderboard(ablation_data, primary_metric='f1_score', ascending=False)
    formatted_table2 = format_leaderboard_for_display(ablation_df)
    
    # 4. Statistical comparisons
    if metrics_dict and baseline_model:
        try:
            stats_results = compare_models(metrics_dict, baseline_model)
            with open(os.path.join(output_dir, 'statistical_tests.json'), 'w') as f:
                json.dump(stats_results, f, indent=4)
        except Exception as e:
            print(f"Statistical comparison failed: {e}")
            
    # 5. Generate all reports (LaTeX, CSV, Plots)
    reports_dir = os.path.join(output_dir, 'reports')
    generate_all_reports(
        table1_df=formatted_table1,
        table2_df=formatted_table2,
        fig1_data=fig1_data,
        roc_data=roc_data,
        pr_data=pr_data,
        km_data=km_data,
        output_dir=reports_dir
    )
