#!/usr/bin/env python3
"""
Compare all available 2D DEXSY models on a standardized benchmark.

This script loads all trained model checkpoints and evaluates them on the same
test dataset with consistent metrics.

Reference: Paper Section 2.4 - Model Benchmarks

Usage:
    # Run full benchmark with all available models
    python -m benchmarks_2d.compare_models

    # Run with custom test size
    python -m benchmarks_2d.compare_models --n-test 500

    # Skip specific models
    python -m benchmarks_2d.compare_models --skip attention_unet --skip plain_unet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks_2d.evaluator import (
    BenchmarkEvaluator,
    evaluate_all_models,
    ModelBenchmarkResult,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare all available 2D DEXSY models on a standardized benchmark."
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=500,
        help="Number of test samples (default: 500 for statistical significance)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: outputs/benchmark/YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        dest="skip_models",
        help="Model names to skip (can be repeated)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to evaluate (default: all available)",
    )
    parser.add_argument(
        "--noise-sigma-range",
        type=float,
        nargs=2,
        default=(0.005, 0.015),
        metavar=("LOW", "HIGH"),
        help="Noise sigma range for test data (default: 0.005 0.015)",
    )
    parser.add_argument(
        "--n-compartments",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of compartments in test data (default: 2)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "outputs" / "benchmark" / timestamp
    
    print("=" * 70)
    print("2D DEXSY Model Benchmark")
    print("=" * 70)
    print(f"Test samples:     {args.n_test}")
    print(f"Random seed:      {args.seed}")
    print(f"Noise range:      {args.noise_sigma_range}")
    print(f"Compartments:     {args.n_compartments}")
    print(f"Output directory: {output_dir}")
    print(f"Skip models:      {args.skip_models or 'None'}")
    print("=" * 70)
    print()
    
    # Run evaluation
    try:
        results = evaluate_all_models(
            n_test=args.n_test,
            seed=args.seed,
            output_dir=output_dir if not args.no_save else None,
            skip_models=args.skip_models,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Create evaluator to generate table
    evaluator = BenchmarkEvaluator(
        n_test=args.n_test,
        seed=args.seed,
        noise_sigma_range=tuple(args.noise_sigma_range),
        n_compartments=args.n_compartments,
        output_dir=output_dir if not args.no_save else None,
    )
    
    # Re-add results to evaluator
    for model_name, result in results.items():
        evaluator.model_results[model_name] = result
    
    # Print comparison table
    table = evaluator.generate_comparison_table()
    print()
    print(table.to_string(index=False))
    print()
    
    # Save results
    if not args.no_save:
        saved_path = evaluator.save_results(output_dir)
        print(f"Results saved to: {saved_path}")
    
    print()
    print("Benchmark complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
