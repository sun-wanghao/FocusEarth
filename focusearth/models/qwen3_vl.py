"""Pinned Qwen3-VL integration for one-image MME answering.

The adapter keeps the Transformers model's visual encoder and generation path
intact. The local hook changes only post-merger visual feature groups, image
placeholder count, and the matching sparse spatial positions.
"""

from __future__ import annotations

from typing import Any

from ..visual_representation.canvas_geometry import fetch_smart_resize
from .qwen3_vl_integration import (
    QwenVisualCompression,
    rewrite_processor_batch_one_image,
)

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TRANSFORMERS_VERSION = "4.57.6"
GENERATION = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 64,
    "use_cache": True,
}


def load_qwen3_vl(checkpoint: str, *, local_files_only: bool = True):
    """Load the BF16/SDPA model and left-padding processor."""
    import transformers
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"requires transformers=={TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )
    processor = AutoProcessor.from_pretrained(
        checkpoint, padding_side="left", local_files_only=local_files_only
    )
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=local_files_only,
    ).eval()
    return model, processor


def _prepare_image(image):
    """Apply the factor-28 image-fetch resize before the pinned Qwen processor.

    RGB images use factor-28 geometry and PIL bicubic interpolation. The
    Transformers processor subsequently applies factor-32 smart resizing.
    """
    from PIL import Image

    rgb = image.convert("RGB")
    target = fetch_smart_resize(rgb.size)
    if rgb.size != target:
        resized = rgb.resize(target, Image.Resampling.BICUBIC)
        rgb.close()
        return resized
    return rgb


def build_processor_batch(
    processor, image, prompt: str
) -> tuple[dict[str, Any], list[int]]:
    """Construct the user message and one-image processor tensors."""
    if not prompt:
        raise ValueError("prompt must not be empty")
    prepared = _prepare_image(image)
    try:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": prepared},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        rendered = processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        batch = processor(
            text=[rendered],
            images=[prepared],
            videos=None,
            padding=True,
            return_tensors="pt",
        )
    finally:
        prepared.close()
    grid = batch.get("image_grid_thw")
    if grid is None or tuple(grid.shape) != (1, 3):
        raise RuntimeError("Qwen processor did not return one image_grid_thw row")
    return batch, [int(value) for value in grid[0].tolist()]


def generate_answer(
    model, processor, batch: dict[str, Any], plan: dict
) -> tuple[str, dict[str, Any]]:
    """Run greedy generation after validating placeholder/feature counts."""
    import torch

    token_id = int(processor.tokenizer.convert_tokens_to_ids(processor.image_token))
    rewritten = rewrite_processor_batch_one_image(
        {**batch, "region_protected_compression_image_token_id": token_id}, plan
    )
    model_inputs = {
        key: value
        for key, value in rewritten.items()
        if not key.startswith("region_protected_compression_")
    }
    device = next(model.parameters()).device
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }
    with QwenVisualCompression(model, plan), torch.inference_mode():
        generated = model.generate(**model_inputs, **GENERATION)
    input_length = int(model_inputs["input_ids"].shape[1])
    generated_only = generated[:, input_length:]
    raw = processor.batch_decode(
        generated_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return raw, {
        "rendered_input_length": input_length,
        "generated_sequence_length": int(generated.shape[1]),
        "generation": dict(GENERATION),
        "image_grid_thw": batch["image_grid_thw"].tolist(),
    }
