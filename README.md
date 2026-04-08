# FYP 2D DEXSY Bundle

This is a slimmed-down GitHub/Colab-ready version of the project that keeps only the parts needed for the 2D workflow:

- `improved_2d_dexsy/forward_model_2d.py`
- `improved_2d_dexsy/model_2d.py`
- `improved_2d_dexsy/train_2d.py`
- `improved_2d_dexsy/generate_forward_figures.py`
- `checkpoints/attention_unet_best_model.pt`
- `notebooks/colab_demo.ipynb`

The bundled checkpoint is the latest trained 2D inverse model and can be loaded directly for inference in Colab.

## Repo Layout

```text
.
├── checkpoints/
│   ├── attention_unet_best_model.pt
│   └── training_log.csv
├── improved_2d_dexsy/
│   ├── __init__.py
│   ├── forward_model_2d.py
│   ├── generate_forward_figures.py
│   ├── inference_2d.py
│   ├── model_2d.py
│   └── train_2d.py
├── notebooks/
│   └── colab_demo.ipynb
└── requirements.txt
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

## Local Training

From the repo root:

```bash
python improved_2d_dexsy/train_2d.py
```

Training outputs will be written to `training_output_2d/`.

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
