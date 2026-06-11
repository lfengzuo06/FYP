# DEXSY Interface User Guide

This guide explains how to use the two main Gradio interfaces in this
codebase:

- `app_3step.py`: interactive reconstruction and inference interface
- `app_training.py`: reproducible dataset, training, comparison, and history
  interface

Use `app_training.py` when you want to create datasets, train models, or compare
experiments. Use `app_3step.py` when you want to inspect one DEXSY signal,
choose a model/checkpoint, run inference, and export the result.

## Before You Start

Run all commands from the repository root:

```bash
cd FYP
python -m pip install -r requirements.txt
```

If Matplotlib warns that its cache directory is not writable, set a local cache:

```bash
mkdir -p .mplcache
export MPLCONFIGDIR="$PWD/.mplcache"
```

Stop a running interface with `Ctrl+C` in the terminal.

## Interface 1: Three-Step Reconstruction

Launch:

```bash
python app_3step.py
```

Gradio will print a local URL in the terminal. This interface has three tabs:

1. `Step 1: Data Input`
2. `Step 2: Model Selection & Inference`
3. `Step 3: Results`

### Intended Use

Use this interface to run inference on a single sample. The sample can be:

- generated from explicit physical parameters
- generated randomly
- uploaded from disk
- loaded from an immutable historical dataset

The interface is best for visual inspection, debugging model behavior, checking
one trained checkpoint, and exporting a single result bundle.

### Step 1: Data Input

First choose the grid size:

- `16`: faster and useful for quick checks or g16 checkpoints
- `64`: higher-resolution spectra and the default for many checkpoints

Then choose one input method.

#### Parametric Generation

Use this when you want a controlled synthetic sample.

1. Select `2` or `3` compartments.
2. Adjust diffusion coefficients, volume fractions, exchange rates, mixing time,
   and noise.
3. Click `Generate from Parameters`.
4. Inspect the signal preview and parameter JSON.
5. Click `Confirm Input & Continue to Step 2`.

For 2-compartment samples, the main controls are `D1`, `D2`, volume fraction,
exchange rate, mixing time, and noise sigma.

For 3-compartment samples, the controls expose three diffusion pools, volume
fractions, pairwise exchange rates, mixing time, and noise sigma.

#### Random Generation

Use this when you want a valid synthetic sample without hand-picking physical
parameters.

1. Select `2` or `3` compartments.
2. Click `Generate Random Sample`.
3. Check the preview and sampled parameters.
4. Click `Confirm Input & Continue to Step 2`.

#### Upload Signal Image

Despite the tab name, this loads matrix files rather than image files.

Supported formats:

- `.npy`
- `.npz`
- `.csv`
- `.txt`

Workflow:

1. Upload one signal matrix.
2. Click `Load Signal`.
3. Check the preview and signal metadata.
4. Click `Confirm Input & Continue to Step 2`.

Uploaded signals do not include ground truth, so the results tab will show
prediction metrics such as predicted DEI but not ground-truth reconstruction
error unless a ground truth is otherwise available.

#### Historical Dataset Sample

Use this when you want reproducible input from datasets created in
`app_training.py` or `dexsy_datasets`.

1. Set `Datasets Base Path`; the default is `datasets`.
2. Click `Refresh Datasets`.
3. Select a `Dataset ID`.
4. Choose `train`, `val`, or `test`.
5. Enter a sample index.
6. Click `Load Dataset Sample`.
7. Check the preview and metadata.
8. Click `Confirm Input & Continue to Step 2`.

This is the preferred input mode for comparing trained runs on known samples.

### Step 2: Model Selection & Inference

Choose a model, checkpoint, and device.

Key fields:

- `Model`: inverse model family, such as `attention_unet`,
  `plain_unet`, `pinn`, `deep_unfolding`, `fno`, or a 3C variant.
- `Checkpoint`: checkpoint file to load. The dropdown is populated from
  `checkpoints_2d/` and `checkpoints_3d/`, and custom paths can be entered.
- `Device`: `auto`, `cpu`, or `cuda`.

If you trained a model in the research interface:

1. Keep the `Run History File` as `outputs/research_runs/history.jsonl`, or set
   it to the correct history file.
2. Click `Refresh Runs`.
3. Select a `Run ID`.
4. Click `Use This Run Checkpoint`.
5. Click `Run Inference`.

Make sure the model and checkpoint match the input grid size and task type.
For example, a `*_g16` checkpoint expects 16x16 data, while many non-`g16`
checkpoints expect or support 64x64 data.

### Step 3: Results

After inference, this tab shows:

- comparison visualization
- predicted DEI
- reconstruction metrics when ground truth is available
- full metrics JSON
- downloadable result bundle

Click `Download Result Bundle (.zip)` to export the prediction arrays, metrics,
metadata, and figures for the current sample.

### Common Reconstruction Interface Issues

- If `Run Inference` fails immediately, check that Step 1 input was confirmed.
- If a checkpoint cannot load, verify that the checkpoint matches the selected
  model family.
- If the output shape is wrong, check whether the input grid size is 16 or 64.
- If no ground-truth metrics appear, the selected input source probably did not
  provide a ground-truth spectrum.
- `3d_ilt` is registered as a model choice in some code paths but is not yet
  implemented for inference.

## Interface 2: Reproducible Research

Launch:

```bash
python app_training.py
```

This interface launches on port `7861` by default and enables sharing. For a
local browser, use the URL printed by Gradio, typically:

```text
http://127.0.0.1:7861
```

The interface has four tabs:

1. `Dataset Lab`
2. `Training Runs`
3. `Fair Compare`
4. `Experiment History`

### Intended Use

Use this interface to manage the full experimental workflow:

- create immutable datasets
- preview and verify dataset contents
- train models with explicit seeds
- compare trained models on one fixed test dataset
- inspect persistent run history

### Global Dataset Path

At the top of the interface, `Datasets Base Path` controls where datasets are
loaded from and saved to. The default is:

```text
datasets
```

Click `Refresh Datasets` whenever you create, delete, extend, or manually edit
datasets outside the UI.

### Dataset Lab

Use this tab to create, inspect, verify, and extend datasets.

#### Dataset Registry

The registry table lists datasets under the selected base path. Typical columns
include dataset ID, task type, model type, grid size, number of compartments,
split sizes, seed, and creation time.

Useful actions:

- `Verify Dataset`: checks dataset integrity and metadata consistency.
- `Delete Dataset`: removes the selected dataset. Use this carefully.
- `Export Config YAML`: exports the selected dataset config.
- `Preview Sample`: displays one signal/spectrum sample from `train`, `val`, or
  `test`.

#### Create Dataset

Choose:

- `Model Type`: `gaussian_2c`, `gaussian_3c`, `gaussian_nc`, or
  `nongaussian_3c`
- `Grid Size (n_b)`: `16` or `64`
- `Seed`: random seed used for reproducible generation
- `n_comp`: only used for `gaussian_nc`
- `Dataset Purpose`: `train_val_test` or `compare_test_only`
- `Sampling Strategy`: usually `log_uniform`
- split sizes: `n_train`, `n_val`, `n_test`

The `Parameter Ranges` textbox accepts YAML or JSON overrides. Leave the default
values when you want the standard configuration for the selected model type.

Click `Create Dataset`. When creation finishes, the registry and dropdowns are
updated.

Use `compare_test_only` when you want a fixed benchmark dataset for evaluating
existing checkpoints. In that mode, training and validation counts are forced to
zero.

#### Extend Dataset

Use this when a training dataset needs more training samples while keeping the
same validation/test logic.

1. Select `Base Dataset`.
2. Set `Add Train Samples`.
3. Click `Extend Dataset`.

The result is saved as a new dataset rather than mutating the original dataset.

#### Append Manual Sample

Use this for one-off controlled examples.

1. Select the base dataset in `Base Dataset`.
2. Choose a target split.
3. Edit the manual sample YAML/JSON.
4. Click `Append Manual Sample`.

This is useful for adding edge cases, but it should be used sparingly because it
can make a dataset less statistically uniform than a generated batch.

### Training Runs

Use this tab to train one model on one selected dataset.

Workflow:

1. Select a `Dataset ID`.
2. Select a compatible `Model`.
3. Optionally enter a run name.
4. Set epochs, batch size, learning rate, weight decay, and early stopping.
5. Set the logged data seed, initialization seed, and dataloader seed.
6. Open `Advanced Model Args` only if you need model-specific controls.
7. Click `Run Training`.

Outputs shown in the tab:

- training status
- training curve plot
- test metrics
- run manifest

The training run is also recorded in:

```text
outputs/research_runs/history.jsonl
```

Model artifacts are written to the relevant checkpoint/output folders. The
history manifest is what lets `app_3step.py` later load a trained checkpoint by
run ID.

### Fair Compare

Use this tab to compare already-trained models on one fixed test dataset.

Workflow:

1. Select a benchmark `Dataset ID`.
2. Set `Eval Batch Size`.
3. Choose model runs from `Models`.
4. Click `Run Fair Compare`.

The interface reports:

- model availability
- summary table with mean and standard deviation
- per-run results
- comparison plot

For fair comparison, use the same test dataset for every model. Avoid comparing
models evaluated on different generated test sets unless the difference is the
point of the experiment.

### Experiment History

Use this tab to inspect previous training runs.

Actions:

- `Refresh History`: reloads `outputs/research_runs/history.jsonl`
- select a `Run ID`: displays the full run detail JSON

The run history is useful for:

- finding checkpoint paths
- confirming training seeds
- checking dataset IDs used for training
- recording model hyperparameters
- selecting checkpoints in `app_3step.py`

### Recommended Research Workflow

For a clean experiment:

1. Open `app_training.py`.
2. Create or refresh a dataset in `Dataset Lab`.
3. Verify and preview the dataset.
4. Train one or more compatible models in `Training Runs`.
5. Check run manifests in `Experiment History`.
6. Compare trained runs on a fixed test dataset in `Fair Compare`.
7. Open `app_3step.py`.
8. Load a historical dataset sample.
9. Load a checkpoint from run history.
10. Run inference and export a result bundle for figures or reporting.

### Common Research Interface Issues

- If model choices are empty in `Training Runs`, select or refresh a dataset
  first.
- If no comparison candidates appear, there may be no compatible trained
  checkpoints for the selected dataset type and grid size.
- If training is slow, reduce `n_train`, use `16` grid size, reduce epochs, or
  use a smaller model for a smoke test.
- If CUDA runs out of memory, reduce batch size or switch `Device` to CPU in
  the reconstruction interface.
- If dataset verification fails, recreate the dataset from config or inspect the
  dataset folder for missing metadata/array files.

## Choosing Between The Two Interfaces

Use `app_training.py` for experiment management:

- dataset creation
- reproducible model training
- run history
- fair model comparison

Use `app_3step.py` for single-sample inference:

- controlled synthetic sample generation
- uploaded signal reconstruction
- historical dataset sample inspection
- checkpoint-by-checkpoint visual debugging
- result bundle export

In practice, most experiments start in `app_training.py` and finish with a
small number of qualitative examples in `app_3step.py`.
