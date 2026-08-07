import pandas as pd
from typing import Dict, Any, List

def generate_leaderboard(results_data: List[Dict[str, Any]], primary_metric: str = 'f1_score', ascending: bool = False) -> pd.DataFrame:
    """
    Generate a leaderboard from a list of model results.
    """
    if not results_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(results_data)
    
    if primary_metric not in df.columns:
        raise ValueError(f"Primary metric '{primary_metric}' not found in results data.")
        
    # Aggregate over multiple seeds if 'seed' column exists and 'model_name' exists
    if 'model_name' in df.columns and 'seed' in df.columns:
        numeric_cols = df.select_dtypes(include='number').columns
        agg_funcs = {col: ['mean', 'std'] for col in numeric_cols if col != 'seed'}
        
        aggregated_df = df.groupby('model_name').agg(agg_funcs).reset_index()
        
        # Flatten multi-level columns
        aggregated_df.columns = ['_'.join(col).strip('_') for col in aggregated_df.columns.values]
        
        sort_col = f"{primary_metric}_mean"
        if sort_col in aggregated_df.columns:
            aggregated_df = aggregated_df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
            aggregated_df.index += 1
            aggregated_df.index.name = 'Rank'
            return aggregated_df
            
    # Simple sort if no aggregation needed
    df = df.sort_values(by=primary_metric, ascending=ascending).reset_index(drop=True)
    df.index += 1
    df.index.name = 'Rank'
    return df

def format_leaderboard_for_display(leaderboard_df: pd.DataFrame, metric_cols: List[str] = None) -> pd.DataFrame:
    """
    Format the leaderboard for display (e.g. rounding, formatting mean/std).
    """
    if leaderboard_df.empty:
        return leaderboard_df
        
    formatted_df = leaderboard_df.copy()
    
    if metric_cols is None:
        metric_cols = [col.replace('_mean', '') for col in formatted_df.columns if col.endswith('_mean')]
        
    for metric in metric_cols:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        
        if mean_col in formatted_df.columns and std_col in formatted_df.columns:
            formatted_df[metric] = formatted_df.apply(
                lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}" if pd.notnull(row[std_col]) else f"{row[mean_col]:.4f}", axis=1
            )
            formatted_df = formatted_df.drop(columns=[mean_col, std_col])
            
    return formatted_df
