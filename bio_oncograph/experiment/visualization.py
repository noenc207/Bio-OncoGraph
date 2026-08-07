"""Module for generating visualizations via Optuna and Matplotlib."""
import os
import optuna
import matplotlib.pyplot as plt

def plot_optimization_history(study: optuna.Study, save_path: str = "optimization_history.png") -> None:
    """Generates optimization history plot.
    
    Args:
        study: Optuna study object.
        save_path: Path to save the plot.
    """
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_parallel_coordinate(study: optuna.Study, save_path: str = "parallel_coordinate.png") -> None:
    """Generates parallel coordinate plot.
    
    Args:
        study: Optuna study object.
        save_path: Path to save the plot.
    """
    fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_parameter_importance(study: optuna.Study, save_path: str = "parameter_importance.png") -> None:
    """Generates parameter importance plot.
    
    Args:
        study: Optuna study object.
        save_path: Path to save the plot.
    """
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_slice(study: optuna.Study, save_path: str = "slice_plot.png") -> None:
    """Generates slice plot.
    
    Args:
        study: Optuna study object.
        save_path: Path to save the plot.
    """
    fig = optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_contour(study: optuna.Study, save_path: str = "contour_plot.png") -> None:
    """Generates contour plot.
    
    Args:
        study: Optuna study object.
        save_path: Path to save the plot.
    """
    fig = optuna.visualization.matplotlib.plot_contour(study)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def generate_all_plots(study: optuna.Study, output_dir: str) -> None:
    """Generates all optuna plots.
    
    Args:
        study: Optuna study object.
        output_dir: Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_optimization_history(study, os.path.join(output_dir, "optimization_history.png"))
    plot_parallel_coordinate(study, os.path.join(output_dir, "parallel_coordinate.png"))
    plot_parameter_importance(study, os.path.join(output_dir, "parameter_importance.png"))
    plot_slice(study, os.path.join(output_dir, "slice_plot.png"))
    plot_contour(study, os.path.join(output_dir, "contour_plot.png"))
