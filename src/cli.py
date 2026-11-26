"""Command-line entry point for the appointment study."""

import argparse
import json

from src.config import PipelineConfig
from src.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare clubs before and after a manager change.")
    parser.add_argument("--use-existing-parquet", action="store_true",
                        help="Read the four existing tables. The default generates and saves simulated seasons.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generating seasons.")
    args = parser.parse_args()
    artifacts = run_full_pipeline(PipelineConfig(random_seed=args.seed), args.use_existing_parquet)
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
