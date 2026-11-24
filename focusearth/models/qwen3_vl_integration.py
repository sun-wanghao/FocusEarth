"""Qwen3-VL visual-stream compression and sparse spatial position integration."""

from __future__ import annotations

import types
from contextlib import AbstractContextManager
from typing import Any


def build_sparse_visual_position_triplets(
    plan: dict, text_prefix_length: int
) -> list[list[int]]:
    rows, cols = plan["grid"]
    output = []
    for index in plan["representative_indices"]:
        row, col = divmod(index, cols)
        output.append(
            [text_prefix_length, text_prefix_length + row, text_prefix_length + col]
        )
    return output


def _compress_one_stream(stream, plan: dict):
    import torch

    if stream.ndim != 2 or stream.shape[0] != plan["original_tokens"]:
        raise RuntimeError(f"stream shape mismatch {tuple(stream.shape)}")
    groups = plan["output_groups"]
    flat_members = [index for group in groups for index in group["members"]]
    lengths = [len(group["members"]) for group in groups]
    if sorted(flat_members) != list(range(plan["original_tokens"])):
        raise RuntimeError("runtime mapping is not an exact partition")
    gather = torch.tensor(flat_members, dtype=torch.long, device=stream.device)
    segment_lengths = torch.tensor(lengths, dtype=torch.long, device=stream.device)
    grouped = stream.index_select(0, gather).to(torch.float32)
    # Accumulate each row-major group in FP32, then restore the stream dtype.
    # Protected singleton features are copied directly from the input stream.
    compressed = torch.segment_reduce(grouped, "mean", lengths=segment_lengths).to(
        stream.dtype
    )
    protected_output_positions = []
    protected_source_positions = []
    for output_index, group in enumerate(groups):
        if group["kind"] == "protected_identity":
            if group["members"] != [group["representative"]]:
                raise RuntimeError("protected group not identity singleton")
            protected_output_positions.append(output_index)
            protected_source_positions.append(group["representative"])
    if protected_output_positions:
        output_selector = torch.tensor(
            protected_output_positions, dtype=torch.long, device=stream.device
        )
        source_selector = torch.tensor(
            protected_source_positions, dtype=torch.long, device=stream.device
        )
        compressed.index_copy_(
            0, output_selector, stream.index_select(0, source_selector)
        )
    if compressed.shape[0] != plan["retained_tokens"]:
        raise RuntimeError("compressed stream cardinality mismatch")
    return compressed


def compress_final_and_deepstack(final_stream, deepstack_streams, plan: dict):
    if not plan.get("valid"):
        raise RuntimeError(f"invalid mapping plan: {plan.get('failure')}")
    compressed_final = _compress_one_stream(final_stream, plan)
    compressed_deep = []
    for layer, stream in enumerate(deepstack_streams):
        if stream.shape != final_stream.shape:
            raise RuntimeError(
                f"deepstack layer {layer} shape differs from final stream"
            )
        compressed_deep.append(_compress_one_stream(stream, plan))
    return compressed_final, compressed_deep


def rewrite_processor_batch_one_image(
    batch: dict[str, Any], plan: dict
) -> dict[str, Any]:
    """Shorten only the image-placeholder run; pixel_values/grid remain full."""
    import torch

    if batch["input_ids"].shape[0] != 1:
        raise RuntimeError("Region-Protected Token Compression requires batch size one")
    input_ids = batch["input_ids"]
    image_token_id = int(batch["region_protected_compression_image_token_id"])
    active = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten().tolist()
    if len(active) != plan["original_tokens"] or not active:
        raise RuntimeError("original image-placeholder cardinality mismatch")
    if active != list(range(active[0], active[0] + len(active))):
        raise RuntimeError("image placeholders are not one contiguous run")
    keep = list(range(input_ids.shape[1]))
    remove = set(active[plan["retained_tokens"] :])
    keep = [index for index in keep if index not in remove]
    rewritten = dict(batch)
    selector = torch.tensor(keep, dtype=torch.long, device=input_ids.device)
    rewritten["input_ids"] = input_ids.index_select(1, selector)
    if batch.get("attention_mask") is not None:
        rewritten["attention_mask"] = batch["attention_mask"].index_select(1, selector)
    rewritten["region_protected_compression_original_input_length"] = int(
        input_ids.shape[1]
    )
    rewritten["region_protected_compression_rewritten_input_length"] = int(
        rewritten["input_ids"].shape[1]
    )
    rewritten["region_protected_compression_plan_sha256"] = plan["mapping_sha256"]
    return rewritten


class QwenVisualCompression(AbstractContextManager):
    """Apply visual compression at Qwen3-VL insertion points for a single image.

    Call rewrite_processor_batch_one_image before model.generate, then enter this
    context. The full rectangular pixel tensor is still encoded unchanged. Only
    post-merger final/deepstack streams and downstream placeholders/positions are
    compressed.
    """

    def __init__(self, model, plan: dict):
        if not plan.get("valid"):
            raise RuntimeError("cannot enable invalid mapping plan")
        self.model = model
        self.core = model.model
        self.plan = plan
        self.original_get_image_features = self.core.get_image_features
        self.original_get_rope_index = self.core.get_rope_index
        self.original_rope_deltas = self.core.rope_deltas

    def __enter__(self):
        plan = self.plan
        original_rope = self.original_get_rope_index

        def patched_get_image_features(core, pixel_values, image_grid_thw=None):
            if image_grid_thw is None or tuple(image_grid_thw.shape) != (1, 3):
                raise RuntimeError(
                    "Region-Protected Token Compression supports exactly one image"
                )
            full, deep = core.visual(
                pixel_values.type(core.visual.dtype), grid_thw=image_grid_thw
            )
            compressed, compressed_deep = compress_final_and_deepstack(full, deep, plan)
            return [compressed], compressed_deep

        def patched_get_rope_index(
            core,
            input_ids=None,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
        ):
            import torch

            if (
                video_grid_thw is not None
                or input_ids is None
                or image_grid_thw is None
            ):
                return original_rope(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            if input_ids.shape[0] != 1 or tuple(image_grid_thw.shape) != (1, 3):
                raise RuntimeError(
                    "Region-Protected Token Compression positional encoding supports one image per batch"
                )
            mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_ids)
            )
            active_positions = (mask[0] == 1).nonzero(as_tuple=False).flatten()
            active_ids = input_ids[0].index_select(0, active_positions)
            image_token_id = core.config.image_token_id
            image_positions = (
                (active_ids == image_token_id)
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            if len(image_positions) != plan["retained_tokens"] or not image_positions:
                raise RuntimeError("compressed placeholder count does not match plan")
            if image_positions != list(
                range(image_positions[0], image_positions[0] + len(image_positions))
            ):
                raise RuntimeError("compressed placeholders are not contiguous")
            prefix = image_positions[0]
            suffix = len(active_ids) - prefix - len(image_positions)
            triplets: list[list[int]] = [[i, i, i] for i in range(prefix)]
            triplets.extend(build_sparse_visual_position_triplets(plan, prefix))
            next_position = max(max(values) for values in triplets) + 1
            triplets.extend([[next_position + i] * 3 for i in range(suffix)])
            if len(triplets) != len(active_ids):
                raise RuntimeError("sparse position length mismatch")
            positions = torch.ones(
                (3, 1, input_ids.shape[1]),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            active_tensor = torch.tensor(
                triplets, dtype=input_ids.dtype, device=input_ids.device
            ).T
            positions[:, 0, active_positions] = active_tensor
            delta = torch.tensor(
                [[int(active_tensor.max().item()) + 1 - input_ids.shape[1]]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            return positions, delta

        self.core.get_image_features = types.MethodType(
            patched_get_image_features, self.core
        )
        self.core.get_rope_index = types.MethodType(patched_get_rope_index, self.core)
        self.core.rope_deltas = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.core.get_image_features = self.original_get_image_features
        self.core.get_rope_index = self.original_get_rope_index
        self.core.rope_deltas = self.original_rope_deltas
        return False
