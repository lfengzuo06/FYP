# FYP 2D DEXSY Bundle

This is a slimmed-down GitHub/Colab-ready version of the project that keeps only the parts needed for the 2D workflow:

- `improved_2d_dexsy/forward_model_2d.py`
- `improved_2d_dexsy/model_2d.py`
- `improved_2d_dexsy/preprocessing_2d.py`
- `improved_2d_dexsy/train_2d.py`
- `improved_2d_dexsy/inference_2d.py`
- `improved_2d_dexsy/generate_forward_figures.py`
- `checkpoints/attention_unet_best_model_20260411_155746.pt`
- `notebooks/colab_demo.ipynb`

The bundled checkpoint is the latest trained 2D inverse model and can be loaded directly for inference in Colab.

## Repo Layout

```text
.
├── checkpoints/
│   ├── attention_unet_best_model_20260411_155746.pt
│   └── training_log.csv
├── app.py
├── improved_2d_dexsy/
│   ├── __init__.py
│   ├── config.py
│   ├── forward_model_2d.py
│   ├── generate_forward_figures.py
│   ├── inference_2d.py
│   ├── io_2d.py
│   ├── model_2d.py
│   ├── preprocessing_2d.py
│   └── train_2d.py
├── notebooks/
│   └── colab_demo.ipynb
├── run_inference.py
├── tests/
│   ├── test_checkpoint_path.py
│   ├── test_inference_smoke.py
│   └── test_shape_validation.py
└── requirements.txt
```

## Stable Inference API

The project now exposes one main prediction entrypoint for interface work:

```python
from improved_2d_dexsy import predict_from_signal

result = predict_from_signal(signal_64x64)
print(result.dei)
print(result.summary_metrics)
result.figure.show()
```

`predict_from_signal(signal)` returns:

- `result.reconstructed_spectrum`
- `result.dei`
- `result.summary_metrics`
- `result.figure`

This uses the latest bundled U-Net checkpoint (`20260411_155746`) by default,
while still keeping the `model_name=` and `checkpoint_path=` hooks needed for
future model swapping.

For batch usage, you can also keep a pipeline instance alive and reuse it:

```python
from improved_2d_dexsy import DEXSYInferencePipeline

pipeline = DEXSYInferencePipeline()
results = pipeline.predict_batch_from_signals(batch_of_signals)
```

## Colab Quick Start

Open a new Colab notebook and run:

```python
!git clone https://github.com/<your-username>/FYP.git
%cd /content/FYP
!pip install -r requirements.txt
```

Then open and run:

- `notebooks/colab_demo.ipynb`

That notebook will:

1. find the repo root automatically
2. load the bundled checkpoint
3. generate a synthetic 2D forward-model sample
4. run inverse-model prediction
5. show the saved training curve

You can also run plain Python inference in Colab:

```python
from improved_2d_dexsy import predict_from_signal

result = predict_from_signal(signal_64x64)
result.summary_metrics
```

## Command-Line Inference

You can now run inference without opening a notebook.

Single input:

```bash
python run_inference.py --input path/to/signal.npy
```

Batch directory:

```bash
python run_inference.py --input-dir path/to/signals --pattern "*.npy"
```

Synthetic batch demo:

```bash
python run_inference.py --synthetic-count 10
```

All outputs are saved under `outputs/inference/` by default.

## Minimal Interface

Launch the Gradio app locally:

```bash
python app.py
```

The interface supports:

- uploading one `64x64` signal matrix
- choosing a model family and checkpoint
- reconstructing the spectrum
- viewing the figure and DEI
- downloading a zipped result bundle

## Local Training

From the repo root:

```bash
python improved_2d_dexsy/train_2d.py
```

Training outputs will be written to `training_output_2d/`.

## Tests

Run the inference-focused tests with:

```bash
python -m unittest discover -s tests
```

## Forward Figures

To regenerate paper-style forward-model examples:

```bash
python improved_2d_dexsy/generate_forward_figures.py
```

Figures will be saved under `outputs/forward_model_figures/`.

## Notes

- The checkpoint is about 69 MB, so it still fits within GitHub's normal 100 MB per-file limit.
- If you later add more large checkpoints, switch to Git LFS before pushing them.
- This bundle intentionally excludes the older `DEXSI_Python` baseline code and the large local output folders so the repo stays easier to upload and run.
