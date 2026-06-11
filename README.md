# DEXSY Inverse Modelling Workbench

This repository contains the code used for synthetic DEXSY inverse-modelling
experiments. It combines physics-based forward simulators, reproducible dataset
generation, neural inverse models, ILT baselines, command-line inference, and
Gradio interfaces for running and comparing experiments.

The main workflow is:

1. generate synthetic DEXSY signals from a forward model
2. train or load an inverse model
3. reconstruct a diffusion-exchange spectrum or pathway weights
4. evaluate the prediction with DEI and reconstruction metrics
5. inspect or export figures, metrics, checkpoints, and run manifests

## What Is Included

- 2-compartment and 3-compartment Gaussian DEXSY forward models
- N-compartment Gaussian forward model experiments
- 3-compartment non-Gaussian restricted-compartment forward model
- Attention U-Net, plain U-Net, PINN, deep-unfolding, FNO, DeepONet, and CNN
  inverse models
- ILT baseline code for classical comparison
- Immutable dataset creation and verification utilities
- Inference CLI for single files, batches, and synthetic demos
- Gradio apps for quick reconstruction, step-by-step inference, dataset
  creation, model training, and fair comparison
- Unit tests for forward models, inference paths, shape validation, and smoke
  training checks

## Repository Layout

```text
.
├── app.py                     # Minimal 2D Gradio reconstruction app
├── app_3step.py               # Step-by-step data/model/result interface
├── app_training.py            # Dataset, training, comparison, and history app
├── run_inference.py           # CLI for single, batch, and synthetic inference
├── dexsy_core/                # Shared physics, metrics, and preprocessing
├── dexsy_datasets/            # Immutable dataset configs, generation, storage
├── models_2d/                 # 2-compartment inverse models
├── models_3d/                 # 3-compartment inverse models
├── models_nd/                 # N-compartment attention U-Net experiments
├── models_nonGaussian/        # Non-Gaussian pathway-regression models
├── benchmarks_2d/             # 2D ILT and evaluation helpers
├── benchmarks_3d/             # 3C evaluation helpers
├── checkpoints_2d/            # Bundled 2C checkpoints
├── checkpoints_3d/            # Bundled 3C checkpoints
├── checkpoints_nd/            # N-compartment checkpoints
├── checkpoints_nonGaussian/   # Non-Gaussian model checkpoints
├── checkpoints_other/         # User-trained/custom model checkpoints
├── notebooks/                 # Validation and Colab-style notebooks
├── outputs/                   # Generated inference/evaluation artifacts
├── tests/                     # Unit and smoke tests
└── requirements.txt
```

The `models_3d/` name refers to 3-compartment DEXSY experiments. The inputs are
still DEXSY signal grids, with model-specific channels added during
preprocessing.

## Installation

Run commands from the repository root:

```bash
cd FYP
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project is written as local source modules rather than an installed Python
package, so scripts should be run from this directory.

Core dependencies are PyTorch, NumPy, SciPy, Matplotlib, and Gradio. CUDA is
used automatically when available; otherwise inference and tests run on CPU.

## Quick Start

Launch the minimal reconstruction app:

```bash
python app.py
```

Launch the fuller research interface:

```bash
python app_training.py
```

Run a small synthetic inference batch from the command line:

```bash
python run_inference.py --synthetic-count 4 --model-name attention_unet --grid_size 64
```

Outputs are written to timestamped folders under `outputs/inference/` unless
`--output-dir` is provided.

## Command-Line Inference

Single matrix input:

```bash
python run_inference.py --input path/to/signal.npy --model-name attention_unet
```

Single input with a ground-truth spectrum:

```bash
python run_inference.py \
  --input path/to/signal.npy \
  --true-spectrum path/to/spectrum.npy \
  --model-name attention_unet
```

Batch inference over a directory:

```bash
python run_inference.py \
  --input-dir path/to/signals \
  --pattern "*.npy" \
  --model-name plain_unet \
  --batch-size 16
```

Synthetic demo data:

```bash
python run_inference.py \
  --synthetic-count 10 \
  --n-compartments 3 \
  --model-name attention_unet_3c \
  --grid_size 64
```

Supported input matrix formats are `.npy`, `.npz`, `.csv`, and `.txt`.

Available inference model names include:

```text
2d_ilt
3d_ilt
attention_unet
attention_unet_g16
plain_unet
plain_unet_g16
pinn
pinn_g16
deep_unfolding
deep_unfolding_g16
deeponet
fno
attention_unet_3c
attention_unet_3c_g16
plain_unet_3c
plain_unet_3c_g16
pinn_3c
pinn_3c_g16
deep_unfolding_3c
deep_unfolding_3c_g16
```

Grid support differs by checkpoint. The `*_g16` models use 16x16 grids;
most non-`g16` U-Net, deep-unfolding, and FNO models support 16x16 or 64x64;
DeepONet and the base PINN checkpoints are 64x64-oriented.

`3d_ilt` is listed by the CLI registry but currently raises a
`NotImplementedError`; use a trained 3C model for 3-compartment inference.

## Python API

Use the unified inference wrapper when building notebooks or interfaces:

```python
from improved_2d_dexsy import DEXSYInferencePipeline

pipeline = DEXSYInferencePipeline(
    model_name="attention_unet",
    device="auto",
    grid_size=64,
)

result = pipeline.predict_from_signal(
    signal,
    true_spectrum=ground_truth,   # optional
    include_figure=True,
    source_name="sample_001",
)

print(result.dei)
print(result.summary_metrics)
result.figure.show()
```

The returned object includes the reconstructed spectrum, DEI value, optional
ground-truth comparison metrics, metadata, and a Matplotlib figure when figure
generation is enabled.

## Dataset Generation

Dataset configs live in `dexsy_datasets/configs/`.

Create a dataset:

```bash
python -m dexsy_datasets.cli create-dataset \
  --config dexsy_datasets/configs/gaussian_2c_16x16.yaml \
  --output datasets
```

List available datasets:

```bash
python -m dexsy_datasets.cli list-datasets --base-path datasets
```

Verify an existing dataset:

```bash
python -m dexsy_datasets.cli load-dataset DATASET_ID --base-path datasets --verify
```

Extend a dataset with more training samples:

```bash
python -m dexsy_datasets.cli extend-dataset DATASET_ID \
  --base-path datasets \
  --add-train 1000
```

The dataset manager writes immutable dataset folders with config, metadata,
split information, arrays, and checksums so that model comparisons can reuse
the same train/validation/test samples.

## Training

The easiest way to manage training runs is:

```bash
python app_training.py
```

That app can create datasets, launch model training, compare trained runs on a
fixed test set, and record experiment history under `outputs/research_runs/`.

Training scripts can also be launched directly. Examples:

```bash
python models_2d/attention_unet/train.py --grid_size 64 --epochs 60
python models_2d/plain_unet/train.py --grid_size 16 --epochs 60
python models_2d/pinn/train.py --n_d 64 --n_b 64 --epochs 60
python models_2d/deep_unfolding/train.py --n_d 64 --n_b 64 --epochs 60
python models_2d/neural_operators/train.py --model_type fno --n_d 64 --n_b 64
python models_3d/attention_unet/train.py --grid_size 64 --epochs 60
python models_nd/attention_unet/train_unified.py --n_min 2 --n_max 7
python -m models_nonGaussian.cnn.train --n-b 16 --epochs 80
```

Most training scripts accept `--dataset_id` and `--datasets_dir` to train from a
pre-generated immutable dataset instead of generating data inside the script.

## Gradio Interfaces

There are three local interfaces:

- `python app.py`: lightweight 2D upload-and-reconstruct demo
- `python app_3step.py`: guided workflow for data input, model selection, and
  result inspection
- `python app_training.py`: full research workflow for dataset generation,
  training, fair comparison, and experiment history

Gradio prints the local URL in the terminal after launch.

## Outputs And Checkpoints

Generated outputs are intentionally separated from source code:

- inference outputs: `outputs/inference/`
- research run history: `outputs/research_runs/`
- generated datasets: `datasets/`
- training logs: `training_logs/` or model-specific output folders
- bundled/default checkpoints: `checkpoints_2d/`, `checkpoints_3d/`,
  `checkpoints_nd/`, `checkpoints_nonGaussian/`
- custom checkpoints saved by the training app: `checkpoints_other/`

Large generated outputs are ignored by Git through `.gitignore`. If new
checkpoint files become too large for normal Git hosting, store them with Git
LFS or external artifact storage.

## Tests

Run the full test suite:

```bash
python -m unittest discover -s tests
```

Run a focused test file:

```bash
python -m unittest tests.test_forward_model_3c_nongaussian
```

Some smoke tests train small models and can take longer than pure unit tests.

## Troubleshooting

- Always run scripts from the repository root so local imports such as
  `dexsy_core`, `dexsy_datasets`, and `improved_2d_dexsy` resolve correctly.
- If Matplotlib warns that the user cache is not writable, set a local cache
  before running plotting-heavy scripts:

  ```bash
  mkdir -p .mplcache
  export MPLCONFIGDIR="$PWD/.mplcache"
  ```

- If a checkpoint cannot be resolved, pass it explicitly:

  ```bash
  python run_inference.py \
    --input path/to/signal.npy \
    --model-name attention_unet \
    --checkpoint-path checkpoints_2d/attention_unet/attention_unet_best_model.pt
  ```

- Use `python run_inference.py --help` and
  `python -m dexsy_datasets.cli --help` for the exact options available in the
  current checkout.
