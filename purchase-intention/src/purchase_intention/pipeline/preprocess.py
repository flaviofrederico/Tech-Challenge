"""DVC pipeline stage: clean and validate the raw dataset.

This stage reads the raw CSV, removes exact duplicate rows, and writes a
processed CSV that downstream stages (training) depend on. Keeping this as
a separate DVC stage lets DVC skip re-running training when only the raw
data changes in ways that do not affect the cleaned output, and vice versa.
"""
from __future__ import annotations

import argparse

from purchase_intention.config import PipelineConfig
from purchase_intention.data.loader import load_raw_dataset


def run_preprocess(config_path: str) -> None:
    """Load the raw dataset, clean it, and persist the processed version.

    Args:
        config_path: Path to the pipeline YAML configuration file.
    """
    config = PipelineConfig.from_yaml(config_path)
    dataframe = load_raw_dataset(config.data.raw_path)
    cleaned = dataframe.drop_duplicates().reset_index(drop=True)

    output_dir = config.data.processed_path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset.csv"
    cleaned.to_csv(output_path, index=False)
    print(f"Processed dataset written to {output_path} ({len(cleaned)} rows).")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this pipeline stage."""
    parser = argparse.ArgumentParser(description="Preprocess the raw dataset.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_preprocess(args.config)