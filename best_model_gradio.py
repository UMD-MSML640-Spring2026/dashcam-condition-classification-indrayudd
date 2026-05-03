from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

from robustness_utils import (
    COMBINED_CLASSES,
    DATASET_ROOT,
    MODEL_DIR,
    load_checkpoint_model,
    load_test_manifest,
    perturbation_config_from_controls,
    predict_single_image,
    select_best_checkpoint,
    test_gallery_items,
)


BEST_CHECKPOINT = select_best_checkpoint(MODEL_DIR)
MODEL, CHECKPOINT_PAYLOAD = load_checkpoint_model(BEST_CHECKPOINT.path)
TEST_FRAME = load_test_manifest(DATASET_ROOT)
GALLERY_ITEMS = test_gallery_items(TEST_FRAME)


def _selected_row(index: int) -> pd.Series:
    if not 0 <= index < len(TEST_FRAME):
        raise IndexError(f"Gallery index out of range: {index}")
    return TEST_FRAME.iloc[index]


def _prediction_markdown(result: dict, row: pd.Series, checkpoint_path: Path) -> str:
    road_probs = "\n".join(f"- `{label}`: {score:.3f}" for label, score in result["road_probs"].items())
    visibility_probs = "\n".join(
        f"- `{label}`: {score:.3f}" for label, score in result["visibility_probs"].items()
    )
    return f"""
### Prediction

**Road:** `{result["road_prediction"]}`  
**Visibility:** `{result["visibility_prediction"]}`  
**Combined:** `{result["combined_prediction"]}`

### Probabilities

**Road**
{road_probs}

**Visibility**
{visibility_probs}

### Reference

**Test index:** `{int(row.name)}`  
**Image file:** `{row.get("img", Path(row["output_path"]).name)}`  
**Output path:** `{row["output_path"]}`  
**Ground truth:** `{row["road"]} / {row["visibility"]}`  
**Checkpoint:** `{checkpoint_path.name}`  
**Corpus:** `{CHECKPOINT_PAYLOAD.get("corpus_name", "unknown")}`
"""


def _render(index: int, noise_percent: float, blur_percent: float, lighting_percent: float, occlusion_percent: float):
    row = _selected_row(index)
    image = Image.open(row["path"]).convert("RGB")
    config = perturbation_config_from_controls(
        noise_percent=noise_percent,
        blur_percent=blur_percent,
        lighting_percent=lighting_percent,
        occlusion_percent=occlusion_percent,
    )
    result = predict_single_image(MODEL, image, perturbation=config)
    label = f"Image {index:02d} | true={row['road']} / {row['visibility']}"
    return result["perturbed_image"], _prediction_markdown(result, row, BEST_CHECKPOINT.path), label, index


def _on_gallery_select(evt: gr.SelectData, noise_percent: float, blur_percent: float, lighting_percent: float, occlusion_percent: float):
    return _render(evt.index, noise_percent, blur_percent, lighting_percent, occlusion_percent)


def _on_slider_change(index: int, noise_percent: float, blur_percent: float, lighting_percent: float, occlusion_percent: float):
    return _render(index, noise_percent, blur_percent, lighting_percent, occlusion_percent)


with gr.Blocks(title="RWVC Best Model Inspector") as demo:
    gr.Markdown(
        f"""
## RWVC Best Model Inspector

**Dataset root:** `{DATASET_ROOT}`  
**Checkpoint dir:** `{MODEL_DIR}`  
**Selected best checkpoint:** `{BEST_CHECKPOINT.path.name}`  
**Selection metric:** highest `valid_exact_match_acc`, then `valid_joint_score`, then lower `valid_loss`
"""
    )

    current_index = gr.State(0)

    gallery = gr.Gallery(
        value=GALLERY_ITEMS,
        label="Test Set Gallery",
        columns=5,
        rows=2,
        object_fit="contain",
        height="auto",
        allow_preview=False,
    )

    with gr.Row():
        with gr.Column(scale=3):
            selected_label = gr.Markdown()
            image_output = gr.Image(label="Perturbed image", type="pil")
            noise_slider = gr.Slider(0, 100, value=0, step=1, label="Synthesized noise (%)")
            blur_slider = gr.Slider(0, 100, value=0, step=1, label="Blur (%)")
            lighting_slider = gr.Slider(-100, 100, value=0, step=1, label="Lighting change (%)")
            occlusion_slider = gr.Slider(0, 100, value=0, step=1, label="Occlusion (%)")
        with gr.Column(scale=2):
            prediction_output = gr.Markdown(label="Prediction")

    gallery.select(
        _on_gallery_select,
        inputs=[noise_slider, blur_slider, lighting_slider, occlusion_slider],
        outputs=[image_output, prediction_output, selected_label, current_index],
    )

    for control in [noise_slider, blur_slider, lighting_slider, occlusion_slider]:
        control.change(
            _on_slider_change,
            inputs=[current_index, noise_slider, blur_slider, lighting_slider, occlusion_slider],
            outputs=[image_output, prediction_output, selected_label, current_index],
        )

    demo.load(
        _on_slider_change,
        inputs=[current_index, noise_slider, blur_slider, lighting_slider, occlusion_slider],
        outputs=[image_output, prediction_output, selected_label, current_index],
    )


if __name__ == "__main__":
    print(f"Using checkpoint: {BEST_CHECKPOINT.path}")
    print(f"Checkpoint corpus: {BEST_CHECKPOINT.corpus_name}")
    print(f"Best valid exact-match acc: {BEST_CHECKPOINT.valid_exact_match_acc:.4f}")
    print(f"Best valid joint score: {BEST_CHECKPOINT.valid_joint_score:.4f}")
    print(f"Best valid loss: {BEST_CHECKPOINT.valid_loss:.4f}")
    demo.launch(share=True)
