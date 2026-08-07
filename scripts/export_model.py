"""
Script to export BioOncoGraph models to standard formats (ONNX, TorchScript).
"""
import argparse
import logging

from biooncograph.export import ONNXExporter, TorchScriptExporter
from biooncograph.models import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def export_model(model_path: str, output_path: str, export_format: str) -> None:
    """Exports a trained model to a specific format.

    Args:
        model_path: Path to the saved framework model.
        output_path: Destination path for exported file.
        export_format: Format to export ('onnx' or 'torchscript').
    """
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    if export_format.lower() == "onnx":
        logger.info(f"Exporting model to ONNX at {output_path}")
        exporter = ONNXExporter()
    elif export_format.lower() == "torchscript":
        logger.info(f"Exporting model to TorchScript at {output_path}")
        exporter = TorchScriptExporter()
    else:
        raise ValueError(f"Unsupported export format: {export_format}")
        
    exporter.export(model, output_path)
    logger.info("Export completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Model")
    parser.add_argument("--model_path", type=str, required=True, help="Input model path")
    parser.add_argument("--output_path", type=str, required=True, help="Output file path")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript"], default="onnx", help="Export format")
    args = parser.parse_args()
    export_model(args.model_path, args.output_path, args.format)
