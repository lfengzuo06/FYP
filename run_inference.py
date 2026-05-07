#!/usr/bin/env python3
"""Run single-sample or batch 2D DEXSY inference from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improved_2d_dexsy import (  # noqa: E402
    DEXSYInferencePipeline,
    ForwardModel2D,
    create_forward_model,
    InferenceConfig,
    available_models,
    create_run_output_dir,
    list_available_checkpoints,
    load_matrix,
    load_named_matrices_from_directory,
    save_batch_results,
    save_prediction_result,
    save_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 2D DEXSY inverse-model inference on one sample, a directory, or synthetic data."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=str, help="Path to one signal matrix (.npy, .npz, .csv, .txt).")
    mode.add_argument("--input-dir", type=str, help="Directory containing many signal matrices.")
    mode.add_argument("--synthetic-count", type=int, help="Generate N synthetic samples and reconstruct them.")

    parser.add_argument("--true-spectrum", type=str, help="Optional ground-truth spectrum for --input.")
    parser.add_argument(
        "--true-spectra-dir",
        type=str,
        help="Optional directory of ground-truth spectra matched by stem for --input-dir.",
    )
    parser.add_argument("--pattern", type=str, default="*", help="Glob pattern used with --input-dir.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="attention_unet",
        choices=available_models(),
        help="Model family to use.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint override. Defaults to the bundled best model for the selected family.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to use: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory.")
    parser.add_argument("--no-figure", action="store_true", help="Skip PNG figure generation.")
    parser.add_argument(
        "--n-compartments",
        type=int,
        default=2,
        choices=[2, 3],
        help="Synthetic generation mode only.",
    )
    parser.add_argument(
        "--noise-sigma-range",
        type=float,
        nargs=2,
        default=(0.005, 0.015),
        metavar=("LOW", "HIGH"),
        help="Synthetic generation noise range.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=None,
        help="Grid size for inference (16 or 64). Uses checkpoint config if not specified.",
    )
    return parser


def _candidate_paths_for_stem(directory: Path, stem: str) -> list[Path]:
    suffixes = [".npy", ".npz", ".csv", ".txt"]
    return [directory / f"{stem}{suffix}" for suffix in suffixes]


def _load_optional_ground_truth(stem: str, directory: Path | None):
    if directory is None:
        return None
    for candidate in _candidate_paths_for_stem(directory, stem):
        if candidate.exists():
            return load_matrix(candidate)
    return None


def _print_single_result(result, output_dir: Path):
    print(f"Saved output to: {output_dir}")
    print(f"DEI: {result.dei:.6f}")
    for key in ["mse", "mae", "prediction_mass", "prediction_peak"]:
        if key in result.summary_metrics:
            print(f"{key}: {float(result.summary_metrics[key]):.6f}")


def _print_batch_summary(results: list, output_dir: Path):
    dei_values = np.array([result.dei for result in results], dtype=np.float64)
    print(f"Saved batch outputs to: {output_dir}")
    print(f"Samples processed: {len(results)}")
    print(f"Mean predicted DEI: {dei_values.mean():.6f}")
    print(f"Std predicted DEI: {dei_values.std():.6f}")
    with_ground_truth = [result for result in results if result.ground_truth_spectrum is not None]
    if with_ground_truth:
        mse_values = np.array([result.summary_metrics["mse"] for result in with_ground_truth], dtype=np.float64)
        mae_values = np.array([result.summary_metrics["mae"] for result in with_ground_truth], dtype=np.float64)
        print(f"Mean MSE: {mse_values.mean():.6f}")
        print(f"Mean MAE: {mae_values.mean():.6f}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = InferenceConfig(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        include_figures=not args.no_figure,
    )
    if config.model_name in {"deeponet"} and config.resolved_checkpoint_path is None:
        parser.error(
            f"Model '{config.model_name}' requires a checkpoint. "
            "Provide --checkpoint-path or train the model first."
        )

    # Determine grid size for forward model
    grid_size = args.grid_size

    pipeline = DEXSYInferencePipeline(
        model_name=config.model_name,
        checkpoint_path=config.resolved_checkpoint_path,
        device=config.resolved_device,
        grid_size=grid_size,
    )

    if args.input:
        run_dir = create_run_output_dir(config.output_dir, prefix="single")
        signal_path = Path(args.input)
        signal = load_matrix(signal_path)
        true_spectrum = load_matrix(args.true_spectrum) if args.true_spectrum else None
        result = pipeline.predict_from_signal(
            signal,
            true_spectrum=true_spectrum,
            include_figure=config.include_figures,
            source_name=signal_path.stem,
        )
        save_prediction_result(
            result,
            run_dir,
            stem=signal_path.stem,
            save_figure=config.include_figures,
        )
        save_json(vars(args), run_dir / "run_args.json")
        _print_single_result(result, run_dir)
        return 0

    if args.input_dir:
        run_dir = create_run_output_dir(config.output_dir, prefix="batch")
        signal_items = load_named_matrices_from_directory(args.input_dir, pattern=args.pattern)
        if not signal_items:
            parser.error(f"No supported signal files found in {args.input_dir!r}.")

        # Use forward model's grid size for zero padding
        fm = pipeline.forward_model
        zero_shape = (fm.n_d, fm.n_b)

        names = [name for name, _, _ in signal_items]
        signals = np.stack([signal for _, signal, _ in signal_items], axis=0)
        truth_dir = Path(args.true_spectra_dir) if args.true_spectra_dir else None
        truth_list = [_load_optional_ground_truth(name, truth_dir) for name in names]
        true_spectra = None if all(item is None for item in truth_list) else np.stack([
            np.zeros(zero_shape, dtype=np.float32) if item is None else np.asarray(item, dtype=np.float32)
            for item in truth_list
        ], axis=0)

        results = pipeline.predict_batch_from_signals(
            signals,
            true_spectra=true_spectra,
            include_figures=config.include_figures,
            source_names=names,
            batch_size=config.batch_size,
        )

        # Keep "missing GT" explicit in the per-item metadata and drop synthetic zero placeholders.
        for result, gt in zip(results, truth_list):
            if gt is None:
                result.ground_truth_spectrum = None
                result.metadata["ground_truth_available"] = False
                result.summary_metrics.pop("mse", None)
                result.summary_metrics.pop("mae", None)
                result.summary_metrics.pop("ground_truth_dei", None)
            else:
                result.metadata["ground_truth_available"] = True

        save_batch_results(
            results,
            run_dir,
            stems=names,
            save_figures=config.include_figures,
        )
        save_json(vars(args), run_dir / "run_args.json")
        _print_batch_summary(results, run_dir)
        return 0

    run_dir = create_run_output_dir(config.output_dir, prefix="synthetic")
    ground_truths, signals, params_list, _ = pipeline.forward_model.generate_batch(
        n_samples=args.synthetic_count,
        n_compartments=args.n_compartments,
        noise_sigma=None,
        noise_sigma_range=tuple(args.noise_sigma_range),
        return_reference_signal=True,
    )
    source_names = [f"synthetic_{idx:03d}" for idx in range(args.synthetic_count)]
    results = pipeline.predict_batch_from_signals(
        signals,
        true_spectra=ground_truths,
        include_figures=config.include_figures,
        source_names=source_names,
        batch_size=config.batch_size,
    )
    for result, params in zip(results, params_list):
        result.metadata["synthetic_params"] = params
    save_batch_results(
        results,
        run_dir,
        stems=source_names,
        save_figures=config.include_figures,
    )
    save_json(params_list, run_dir / "synthetic_params.json")
    save_json(vars(args), run_dir / "run_args.json")
    _print_batch_summary(results, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
