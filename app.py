#!/usr/bin/env python3
"""Minimal Gradio interface for the 2D DEXSY inference pipeline."""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from improved_2d_dexsy import (  # noqa: E402
    CHECKPOINTS_DIR,
    DEXSYInferencePipeline,
    available_models,
    create_output_archive,
    create_run_output_dir,
    list_available_checkpoints,
    load_matrix,
    save_prediction_result,
    to_serializable,
)


CHECKPOINT_CHOICES = [path.name for path in list_available_checkpoints()]


@lru_cache(maxsize=8)
def get_pipeline(model_name: str, checkpoint_name: str, device_name: str):
    checkpoint_path = CHECKPOINTS_DIR / checkpoint_name if checkpoint_name else None
    device = None if device_name == "auto" else device_name
    return DEXSYInferencePipeline(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def run_inference_app(uploaded_file, model_name: str, checkpoint_name: str, device_name: str):
    if uploaded_file is None:
        raise gr.Error("Please upload a 64x64 signal matrix first.")

    file_path = Path(uploaded_file)
    signal = load_matrix(file_path)
    pipeline = get_pipeline(model_name, checkpoint_name, device_name)

    result = pipeline.predict_from_signal(
        signal,
        include_figure=True,
        source_name=file_path.stem,
    )

    output_dir = create_run_output_dir(prefix="app")
    save_prediction_result(
        result,
        output_dir,
        stem=file_path.stem,
        save_figure=True,
    )
    archive_path = create_output_archive(output_dir, output_dir / f"{file_path.stem}_bundle")

    return (
        result.figure,
        float(result.dei),
        json.dumps(to_serializable(result.summary_metrics), indent=2),
        str(archive_path),
    )


def build_app():
    with gr.Blocks(title="2D DEXSY Reconstruction Demo") as demo:
        gr.Markdown(
            """
            # 2D DEXSY Reconstruction

            Upload one `64x64` DEXSY signal matrix (`.npy`, `.npz`, `.csv`, `.txt`),
            choose a checkpoint, and reconstruct the 2D spectrum with the current U-Net pipeline.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                signal_file = gr.File(
                    label="Upload Signal Matrix",
                    file_count="single",
                    file_types=[".npy", ".npz", ".csv", ".txt"],
                )
                model_name = gr.Dropdown(
                    choices=available_models(),
                    value="attention_unet",
                    label="Model Family",
                )
                checkpoint_name = gr.Dropdown(
                    choices=CHECKPOINT_CHOICES,
                    value=CHECKPOINT_CHOICES[0] if CHECKPOINT_CHOICES else None,
                    label="Checkpoint",
                )
                device_name = gr.Dropdown(
                    choices=["auto", "cpu", "cuda"],
                    value="auto",
                    label="Device",
                )
                run_button = gr.Button("Reconstruct", variant="primary")

            with gr.Column(scale=2):
                output_plot = gr.Plot(label="Reconstruction Figure")
                dei_value = gr.Number(label="Predicted DEI")
                summary_json = gr.Code(label="Summary Metrics", language="json")
                download_bundle = gr.File(label="Download Result Bundle (.zip)")

        run_button.click(
            fn=run_inference_app,
            inputs=[signal_file, model_name, checkpoint_name, device_name],
            outputs=[output_plot, dei_value, summary_json, download_bundle],
        )

    return demo


if __name__ == "__main__":
    build_app().launch()
