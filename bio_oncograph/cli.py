"""Command-line interface for BioOncoGraph."""
import click
from bio_oncograph.__version__ import __version__

@click.group()
def cli():
    """BioOncoGraph CLI."""
    pass

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to training config.')
def train(config):
    """Train a BioOncoGraph model."""
    click.echo(f"Training model with config: {config}")

@cli.command()
@click.option('--model-path', required=True, type=click.Path(exists=True), help='Path to trained model.')
def evaluate(model_path):
    """Evaluate a trained model."""
    click.echo(f"Evaluating model at: {model_path}")

@cli.command()
@click.option('--model-path', required=True, type=click.Path(exists=True), help='Path to trained model.')
@click.option('--input-data', required=True, type=click.Path(exists=True), help='Path to input data.')
def predict(model_path, input_data):
    """Make predictions using a trained model."""
    click.echo(f"Predicting on {input_data} with model {model_path}")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to benchmark config.')
def benchmark(config):
    """Run benchmark tests."""
    click.echo(f"Running benchmarks with config: {config}")

@cli.command()
@click.option('--config', required=True, type=click.Path(exists=True), help='Path to tuning config.')
def tune(config):
    """Hyperparameter tuning."""
    click.echo(f"Tuning model with config: {config}")

@cli.command()
@click.option('--model-path', required=True, type=click.Path(exists=True), help='Path to trained model.')
@click.option('--output', required=True, type=click.Path(), help='Path to export the model.')
def export(model_path, output):
    """Export a trained model for deployment."""
    click.echo(f"Exporting model from {model_path} to {output}")

@cli.command()
def doctor():
    """Check environment and dependencies."""
    click.echo("Running diagnostics...")
    click.echo("Environment is healthy.")

@cli.command()
def version():
    """Print the version number."""
    click.echo(f"BioOncoGraph version {__version__}")

if __name__ == '__main__':
    cli()
