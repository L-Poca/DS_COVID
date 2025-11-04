#!/usr/bin/env python
"""
Main script to run the complete EDA pipeline
Can be run locally or on Colab
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.explorationdata.pipeline.pipeline_runner import EDAPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive EDA pipeline on COVID-19 dataset"
    )

    parser.add_argument(
        '--base-path',
        type=str,
        required=True,
        help='Path to COVID-19_Radiography_Dataset/COVID-19_Radiography_Dataset'
    )

    parser.add_argument(
        '--metadata-path',
        type=str,
        required=True,
        help='Path to metadata directory'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', None],
        help='Device to use for deep learning (auto-detect if not specified)'
    )

    parser.add_argument(
        '--max-images-per-class',
        type=int,
        default=None,
        help='Maximum number of images per class (None = all)'
    )

    args = parser.parse_args()

    # Validate paths
    base_path = Path(args.base_path)
    metadata_path = Path(args.metadata_path)

    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        sys.exit(1)

    if not metadata_path.exists():
        print(f"Error: Metadata path does not exist: {metadata_path}")
        sys.exit(1)

    # Create and run pipeline
    print("Initializing EDA pipeline...")
    pipeline = EDAPipeline(
        base_path=str(base_path),
        metadata_path=str(metadata_path),
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        max_images_per_class=args.max_images_per_class
    )

    print("Running full pipeline...")
    success = pipeline.run_full_pipeline()

    if success:
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print(f"Results saved to: {pipeline.output_dir}")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("Pipeline failed. Check logs for details.")
        print(f"Partial results may be in: {pipeline.output_dir}")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
