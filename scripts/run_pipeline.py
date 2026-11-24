"""Pipeline orchestration entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_eda import run_eda
from src import config
from src.data.download import download_dataset
from src.models.train import run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the end-to-end readmission pipeline.")
    parser.add_argument("--skip-download", action="store_true", help="Reuse the cached dataset.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA figure generation.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training.")
    parser.add_argument("--skip-keras", action="store_true", help="Skip the TensorFlow model.")
    parser.add_argument("--quick", action="store_true", help="Train on a 30k-row sample.")
    parser.add_argument("--force-download", action="store_true", help="Force dataset re-download.")
    arguments = parser.parse_args(argv)

    config.ensure_directories()

    if not arguments.skip_download:
        download_dataset(force=arguments.force_download)
    if not arguments.skip_eda:
        run_eda()
    if not arguments.skip_train:
        run_training(skip_keras=arguments.skip_keras, quick=arguments.quick)

    print("Pipeline finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
