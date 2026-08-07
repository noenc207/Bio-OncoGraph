import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pandas as pd
from datetime import datetime

from biooncograph.research.evaluation.benchmark_runner import BenchmarkRunner
from biooncograph.ecosystem.integration import EcosystemManager

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Starting benchmark with config:\n{OmegaConf.to_yaml(cfg)}")
    
    output_dir = cfg.get("benchmark", {}).get("output_dir", "benchmark_output")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Initializing EcosystemManager...")
    ecosystem = EcosystemManager(cfg.ecosystem)
    
    logger.info("Initializing BenchmarkRunner...")
    runner = BenchmarkRunner(cfg.benchmark, ecosystem)
    
    logger.info("Running benchmarks...")
    results = runner.run_all()
    
    # Generate CSV tables
    logger.info("Generating CSV reports...")
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate Markdown report
    logger.info("Generating Markdown report...")
    md_path = os.path.join(output_dir, "benchmark_report.md")
    with open(md_path, 'w') as f:
        f.write("# BioOncoGraph Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n\n")
        f.write("This report contains benchmark results across different models and datasets in the BioOncoGraph ecosystem.\n\n")
        f.write("## Results Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Configuration Used\n\n```yaml\n")
        f.write(OmegaConf.to_yaml(cfg.benchmark))
        f.write("\n```\n")
        
    logger.info(f"Benchmark reports saved to {output_dir}")

if __name__ == "__main__":
    main()
