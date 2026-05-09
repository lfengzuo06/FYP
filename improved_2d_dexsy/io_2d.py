"""I/O helpers for 2D DEXSY signals, predictions, and saved artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


SUPPORTED_MATRIX_SUFFIXES = {".npy", ".npz", ".csv", ".txt"}


def to_serializable(value: Any) -> Any:
    """Convert numpy-heavy structures into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_matrix(path: str | Path) -> np.ndarray:
    """Load one 2D signal matrix from ``.npy``, ``.npz``, ``.csv`` or ``.txt``."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_MATRIX_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}' for {path.name}. "
            f"Supported: {sorted(SUPPORTED_MATRIX_SUFFIXES)}."
        )

    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        if "signal" in archive.files:
            array = archive["signal"]
        elif len(archive.files) == 1:
            array = archive[archive.files[0]]
        else:
            raise ValueError(
                f"{path.name} contains multiple arrays. Add a 'signal' key or use a single-array archive."
            )
    else:
        array = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)

    return np.asarray(array, dtype=np.float32)


def load_named_matrices_from_directory(
    directory: str | Path,
    pattern: str = "*",
) -> list[tuple[str, np.ndarray, Path]]:
    """Load supported signal matrices from one directory."""
    directory = Path(directory)
    files = sorted(
        path for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_MATRIX_SUFFIXES
    )
    return [(path.stem, load_matrix(path), path) for path in files]


def save_json(data: Any, path: str | Path) -> Path:
    """Save JSON-serializable content with indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_serializable(data), indent=2))
    return path


def save_prediction_result(
    result,
    output_dir: str | Path,
    stem: str = "prediction",
    *,
    save_figure: bool = True,
    save_input_signal: bool = True,
) -> dict[str, Path]:
    """Save one prediction result bundle to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    if save_input_signal:
        paths["signal"] = output_dir / f"{stem}_signal.npy"
        np.save(paths["signal"], np.asarray(result.signal, dtype=np.float32))

    paths["prediction"] = output_dir / f"{stem}_prediction.npy"
    np.save(paths["prediction"], np.asarray(result.reconstructed_spectrum, dtype=np.float32))

    ground_truth = getattr(result, "ground_truth_spectrum", None)
    if ground_truth is not None:
        paths["ground_truth"] = output_dir / f"{stem}_ground_truth.npy"
        np.save(paths["ground_truth"], np.asarray(ground_truth, dtype=np.float32))

    paths["summary"] = save_json(result.summary_metrics, output_dir / f"{stem}_summary.json")
    paths["metadata"] = save_json(result.metadata, output_dir / f"{stem}_metadata.json")

    if save_figure and getattr(result, "figure", None) is not None:
        paths["figure"] = output_dir / f"{stem}_figure.png"
        result.figure.savefig(paths["figure"], dpi=150, bbox_inches="tight")

    return paths


def save_batch_results(
    results: list,
    output_dir: str | Path,
    *,
    stems: list[str] | None = None,
    save_figures: bool = True,
) -> dict[str, Any]:
    """Save a batch of prediction results and one CSV summary table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stems is None:
        stems = [
            getattr(result, "source_name", None) or f"sample_{idx:03d}"
            for idx, result in enumerate(results)
        ]

    saved_items: list[dict[str, Path]] = []
    rows: list[dict[str, Any]] = []

    for stem, result in zip(stems, results):
        item_dir = output_dir / stem
        saved_items.append(
            save_prediction_result(
                result,
                item_dir,
                stem=stem,
                save_figure=save_figures,
            )
        )

        row = {"sample": stem}
        row.update(to_serializable(result.summary_metrics))
        rows.append(row)

    summary_csv = output_dir / "batch_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable_row = {key: row.get(key) for key in fieldnames}
            writer.writerow(serializable_row)

    return {
        "items": saved_items,
        "summary_csv": summary_csv,
        "output_dir": output_dir,
    }


def create_output_archive(
    source_dir: str | Path,
    archive_stem: str | Path | None = None,
) -> Path:
    """Zip a saved output directory and return the archive path."""
    source_dir = Path(source_dir).resolve()
    if archive_stem is None:
        archive_stem = source_dir.parent / source_dir.name
    archive_stem = Path(archive_stem).resolve()

    # Avoid creating the archive inside the source directory itself.
    # Otherwise the zip can include its own growing output and hang.
    try:
        archive_stem.relative_to(source_dir)
    except ValueError:
        pass
    else:
        archive_stem = source_dir.parent / archive_stem.name

    archive_path = shutil.make_archive(str(archive_stem), "zip", root_dir=str(source_dir))
    return Path(archive_path)
