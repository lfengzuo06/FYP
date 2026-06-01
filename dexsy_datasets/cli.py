"""
CLI interface for dexsy_datasets.

Invoked as: python -m dexsy_datasets.cli <subcommand>

Subcommands:
    create-dataset   Create a new dataset
    load-dataset     Load and verify a dataset
    list-datasets    List available datasets
    extend-dataset  Extend dataset with more training samples
    info             Show dataset information
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from dexsy_datasets import (
    create_dataset,
    load_dataset,
    extend_dataset,
    list_datasets,
    dataset_exists,
    get_dataset_info,
    DatasetConfig,
)


def cmd_create_dataset(args: argparse.Namespace) -> int:
    """Create a new dataset."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    output_dir = args.output or "datasets"

    print(f"Loading config from: {config_path}")
    print(f"Output directory: {output_dir}")

    try:
        dataset = create_dataset(
            config=str(config_path),
            output_dir=output_dir,
            generator_version=args.version,
        )
        print(f"\nDataset created successfully!")
        print(f"  Dataset ID: {dataset.dataset_id}")
        print(f"  Task type: {dataset.task_type}")
        print(f"  Total samples: {len(dataset.signals)}")
        print(f"  Train: {dataset.n_train}, Val: {dataset.n_val}, Test: {dataset.n_test}")
        print(f"  Saved to: {Path(output_dir) / dataset.dataset_id}")
        return 0
    except Exception as e:
        print(f"Error creating dataset: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_load_dataset(args: argparse.Namespace) -> int:
    """Load and verify a dataset."""
    if not dataset_exists(args.dataset_id, args.base_path):
        print(f"Error: Dataset not found: {args.dataset_id}", file=sys.stderr)
        print(f"  Searched in: {Path(args.base_path)}")
        return 1

    try:
        dataset = load_dataset(args.dataset_id, args.base_path)
        print(f"Dataset loaded successfully!")
        print(f"\n{dataset.summary()}")

        if args.verify:
            try:
                dataset.validate()
                print("\nValidation: PASSED")
            except Exception as e:
                print(f"\nValidation: FAILED - {e}")
                return 1

        return 0
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_list_datasets(args: argparse.Namespace) -> int:
    """List available datasets."""
    datasets = list_datasets(args.base_path, include_metadata=args.verbose)

    if not datasets:
        print(f"No datasets found in: {Path(args.base_path)}")
        return 0

    print(f"Datasets in {args.base_path}:")
    print("-" * 80)

    for ds in datasets:
        if args.verbose:
            print(f"\nID: {ds.get('dataset_id', 'unknown')}")
            print(f"  Task: {ds.get('task_type')}")
            print(f"  Model: {ds.get('model_type')}")
            print(f"  Created: {ds.get('created_at')}")
            print(f"  Generator: {ds.get('generator_version')}")
        else:
            print(f"  {ds}")

    print(f"\nTotal: {len(datasets)} dataset(s)")
    return 0


def cmd_extend_dataset(args: argparse.Namespace) -> int:
    """Extend dataset with more training samples."""
    if not dataset_exists(args.dataset_id, args.base_path):
        print(f"Error: Base dataset not found: {args.dataset_id}", file=sys.stderr)
        return 1

    output_dir = args.output or args.base_path

    print(f"Extending dataset: {args.dataset_id}")
    print(f"Adding {args.add_train} training samples")
    print(f"Output directory: {output_dir}")

    try:
        dataset = extend_dataset(
            base_dataset_id=args.dataset_id,
            n_add_train=args.add_train,
            output_dir=output_dir,
        )
        print(f"\nDataset extended successfully!")
        print(f"  New Dataset ID: {dataset.dataset_id}")
        print(f"  New train size: {dataset.n_train}")
        print(f"  Val/Test unchanged: {dataset.n_val}/{dataset.n_test}")
        print(f"  Saved to: {Path(output_dir) / dataset.dataset_id}")
        return 0
    except Exception as e:
        print(f"Error extending dataset: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Show detailed dataset information."""
    try:
        info = get_dataset_info(args.dataset_id, args.base_path)
        print(f"Dataset: {info['dataset_id']}")
        print(f"  Path: {info['path']}")
        print(f"  Task type: {info['task_type']}")
        print(f"  Model type: {info['model_type']}")
        print(f"  Samples: {info['n_samples']}")
        print(f"  Grid size (n_b): {info['n_b']}")
        print(f"  Created: {info['created_at']}")
        print(f"  Generator version: {info['generator_version']}")
        print(f"  Checksum: {info['checksum']}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m dexsy_datasets.cli",
        description="Immutable dataset management for DEXSY experiments",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    create_parser = subparsers.add_parser(
        "create-dataset",
        help="Create a new dataset from config",
    )
    create_parser.add_argument(
        "--config", "-c", required=True, help="Path to config YAML file"
    )
    create_parser.add_argument(
        "--output", "-o", default="datasets", help="Output directory (default: datasets)"
    )
    create_parser.add_argument(
        "--version", default="1.0.0", help="Generator version (default: 1.0.0)"
    )
    create_parser.set_defaults(func=cmd_create_dataset)

    load_parser = subparsers.add_parser(
        "load-dataset",
        help="Load and verify a dataset",
    )
    load_parser.add_argument("dataset_id", help="Dataset ID to load")
    load_parser.add_argument(
        "--base-path", "-b", default="datasets", help="Base datasets directory"
    )
    load_parser.add_argument(
        "--verify", "-v", action="store_true", help="Verify dataset integrity"
    )
    load_parser.add_argument(
        "--verbose", action="store_true", help="Verbose error output"
    )
    load_parser.set_defaults(func=cmd_load_dataset)

    list_parser = subparsers.add_parser(
        "list-datasets",
        aliases=["ls"],
        help="List available datasets",
    )
    list_parser.add_argument(
        "--base-path", "-b", default="datasets", help="Base datasets directory"
    )
    list_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full metadata"
    )
    list_parser.set_defaults(func=cmd_list_datasets)

    extend_parser = subparsers.add_parser(
        "extend-dataset",
        help="Extend dataset with more training samples",
    )
    extend_parser.add_argument("dataset_id", help="Base dataset ID to extend")
    extend_parser.add_argument(
        "--add-train", "-n", type=int, required=True, help="Number of samples to add to train"
    )
    extend_parser.add_argument(
        "--output", "-o", help="Output directory (default: same as base-path)"
    )
    extend_parser.add_argument(
        "--base-path", "-b", default="datasets", help="Base datasets directory"
    )
    extend_parser.add_argument(
        "--verbose", action="store_true", help="Verbose error output"
    )
    extend_parser.set_defaults(func=cmd_extend_dataset)

    info_parser = subparsers.add_parser(
        "info",
        help="Show dataset information",
    )
    info_parser.add_argument("dataset_id", help="Dataset ID")
    info_parser.add_argument(
        "--base-path", "-b", default="datasets", help="Base datasets directory"
    )
    info_parser.set_defaults(func=cmd_info)

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
