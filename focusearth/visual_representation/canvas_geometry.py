"""Reference and spatially aligned canvas geometry.

Canvas dimensions depend on source aspect ratio and selected-region geometry.
Sizes use factor-aligned integer rounding before the 0.875 canvas scale."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable


PATCH_SIZE = 16
MERGE_SIZE = 2
PROCESSOR_FACTOR = PATCH_SIZE * MERGE_SIZE
FETCH_FACTOR = 28
PROCESSOR_MIN_PIXELS = 65536
PROCESSOR_MAX_PIXELS = 16777216
FETCH_MIN_PIXELS = 4 * FETCH_FACTOR**2
FETCH_MAX_PIXELS = 16384 * FETCH_FACTOR**2
BOARD_TILE_PX = 448
BOARD_PADDING_PX = 8
BOARD_LABEL_PX = 24
CANONICAL_GLOBAL_LONG_SIDE = 2048
ALIGNED_CANVAS_SCALE = 0.875


def _round_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def _floor_factor(value: float, factor: int) -> int:
    return math.floor(value / factor) * factor


def _ceil_factor(value: float, factor: int) -> int:
    return math.ceil(value / factor) * factor


def smart_resize(size: tuple[int, int] | list[int]) -> tuple[int, int]:
    """Pinned Qwen processor smart_resize; returns (width, height)."""
    width, height = map(int, size)
    h_bar = max(PROCESSOR_FACTOR, _round_factor(height, PROCESSOR_FACTOR))
    w_bar = max(PROCESSOR_FACTOR, _round_factor(width, PROCESSOR_FACTOR))
    if h_bar * w_bar > PROCESSOR_MAX_PIXELS:
        beta = math.sqrt((height * width) / PROCESSOR_MAX_PIXELS)
        h_bar = _floor_factor(height / beta, PROCESSOR_FACTOR)
        w_bar = _floor_factor(width / beta, PROCESSOR_FACTOR)
    elif h_bar * w_bar < PROCESSOR_MIN_PIXELS:
        beta = math.sqrt(PROCESSOR_MIN_PIXELS / (height * width))
        h_bar = _ceil_factor(height * beta, PROCESSOR_FACTOR)
        w_bar = _ceil_factor(width * beta, PROCESSOR_FACTOR)
    return int(w_bar), int(h_bar)


def fetch_smart_resize(size: tuple[int, int] | list[int]) -> tuple[int, int]:
    """Pinned qwen_vl_utils factor-28 fetch resize; returns (width, height)."""
    width, height = map(int, size)
    h_bar = max(FETCH_FACTOR, _round_factor(height, FETCH_FACTOR))
    w_bar = max(FETCH_FACTOR, _round_factor(width, FETCH_FACTOR))
    if h_bar * w_bar > FETCH_MAX_PIXELS:
        beta = math.sqrt((height * width) / FETCH_MAX_PIXELS)
        h_bar = _floor_factor(height / beta, FETCH_FACTOR)
        w_bar = _floor_factor(width / beta, FETCH_FACTOR)
    elif h_bar * w_bar < FETCH_MIN_PIXELS:
        beta = math.sqrt(FETCH_MIN_PIXELS / (height * width))
        h_bar = _ceil_factor(height * beta, FETCH_FACTOR)
        w_bar = _ceil_factor(width * beta, FETCH_FACTOR)
    return int(w_bar), int(h_bar)


@lru_cache(maxsize=None)
def _processor_geometry_cached(width: int, height: int) -> dict:
    fetched_w, fetched_h = fetch_smart_resize((width, height))
    processor_w, processor_h = smart_resize((fetched_w, fetched_h))
    grid = [1, processor_h // PATCH_SIZE, processor_w // PATCH_SIZE]
    return {
        "fetch_size": [fetched_w, fetched_h],
        "processor_size": [processor_w, processor_h],
        "image_grid_thw": grid,
        "visual_tokens": (grid[1] // MERGE_SIZE) * (grid[2] // MERGE_SIZE),
    }


def processor_geometry(size: tuple[int, int] | list[int]) -> dict:
    width, height = map(int, size)
    return _processor_geometry_cached(width, height)


def resize_long_side_size(
    source_size: tuple[int, int] | list[int], target: int = CANONICAL_GLOBAL_LONG_SIDE
) -> tuple[int, int]:
    width, height = map(int, source_size)
    if max(width, height) <= target:
        return width, height
    scale = target / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def board_size(selected_regions: Iterable[dict]) -> tuple[int, int]:
    regions = list(selected_regions)
    rows = sorted({int(item["row"]) for item in regions})
    cols = sorted({int(item["col"]) for item in regions})
    if not rows or not cols:
        raise ValueError("selected regions must not be empty")
    width = len(cols) * BOARD_TILE_PX + (len(cols) + 1) * BOARD_PADDING_PX
    height = (
        len(rows) * (BOARD_TILE_PX + BOARD_LABEL_PX)
        + (len(rows) + 1) * BOARD_PADDING_PX
    )
    return width, height


def reference_layout_geometry(
    source_size: tuple[int, int] | list[int], selected_regions: Iterable[dict]
) -> dict:
    """Compute the global-view and spatial-board reference token capacity."""
    global_size = resize_long_side_size(source_size)
    board_wh = board_size(selected_regions)
    global_geometry = processor_geometry(global_size)
    board_geometry = processor_geometry(board_wh)
    return {
        "global_size": list(global_size),
        "global_geometry": global_geometry,
        "board_size": list(board_wh),
        "board_geometry": board_geometry,
        "total_visual_tokens": global_geometry["visual_tokens"]
        + board_geometry["visual_tokens"],
    }


def token_balanced_canvas(
    source_size: tuple[int, int] | list[int], target_tokens: int
) -> tuple[int, int]:
    """Find a token-balanced canvas with deterministic aspect and token-error ties."""
    source_width, source_height = map(int, source_size)
    ratio = source_width / source_height
    candidates = []
    for cells_h in range(1, 513):
        center = target_tokens / cells_h
        for cells_w in range(max(1, int(round(center)) - 3), int(round(center)) + 4):
            tokens = cells_h * cells_w
            if tokens > 16384:
                continue
            token_error = abs(tokens - target_tokens) / max(1, target_tokens)
            aspect_error = abs(math.log((cells_w / cells_h) / ratio))
            scale_x = (cells_w * PROCESSOR_FACTOR) / source_width
            scale_y = (cells_h * PROCESSOR_FACTOR) / source_height
            anisotropy = abs(scale_x - scale_y) / max(scale_x, scale_y)
            candidates.append(
                (
                    2.0 * token_error + aspect_error,
                    anisotropy,
                    token_error,
                    cells_h,
                    cells_w,
                )
            )
    if not candidates:
        raise RuntimeError("no token-balanced canvas candidate")
    _, _, _, cells_h, cells_w = min(candidates)
    return cells_w * PROCESSOR_FACTOR, cells_h * PROCESSOR_FACTOR


def mapped_box(
    box: list[int] | tuple[int, int, int, int],
    source_size: tuple[int, int] | list[int],
    target_size: tuple[int, int] | list[int],
) -> list[int]:
    source_width, source_height = map(int, source_size)
    target_width, target_height = map(int, target_size)
    x1, y1, x2, y2 = map(int, box)
    dx1 = max(0, min(target_width - 1, round(x1 * target_width / source_width)))
    dy1 = max(0, min(target_height - 1, round(y1 * target_height / source_height)))
    dx2 = max(dx1 + 1, min(target_width, round(x2 * target_width / source_width)))
    dy2 = max(dy1 + 1, min(target_height, round(y2 * target_height / source_height)))
    return [int(dx1), int(dy1), int(dx2), int(dy2)]


def clarity_target_canvas(
    source_size: tuple[int, int] | list[int],
    support_bboxes: list[list[int] | tuple[int, int, int, int]],
    board_tile_processor_size: tuple[float, float] | list[float],
    prior_canvas_size: tuple[int, int] | list[int],
) -> tuple[int, int]:
    """Choose an aspect-consistent canvas from selected-region scale constraints."""
    source_width, source_height = map(int, source_size)
    source_ratio = source_width / source_height
    target_w, target_h = map(float, board_tile_processor_size)
    candidates = []
    prior_processor_w, prior_processor_h = processor_geometry(prior_canvas_size)[
        "processor_size"
    ]
    for cells_h in range(1, 513):
        center = source_ratio * cells_h
        for cells_w in range(max(1, int(round(center)) - 4), int(round(center)) + 5):
            target_size = (cells_w * PROCESSOR_FACTOR, cells_h * PROCESSOR_FACTOR)
            geometry = processor_geometry(target_size)
            if geometry["visual_tokens"] > 16384:
                continue
            processor_w, processor_h = geometry["processor_size"]
            scale_x = target_size[0] / source_width
            scale_y = target_size[1] / source_height
            anisotropy = abs(scale_x - scale_y) / max(scale_x, scale_y)
            if anisotropy > 0.02:
                continue
            prior_gain = min(
                processor_w / prior_processor_w, processor_h / prior_processor_h
            )
            processor_scale_x = processor_w / source_width
            processor_scale_y = processor_h / source_height
            processor_anisotropy = abs(processor_scale_x - processor_scale_y) / max(
                processor_scale_x, processor_scale_y
            )
            if processor_anisotropy > 0.02:
                continue
            processor_placement_anisotropies = []
            canvas_placement_anisotropies = []
            mapped_processor = [
                mapped_box(box, source_size, (processor_w, processor_h))
                for box in support_bboxes
            ]
            mapped_canvas = [
                mapped_box(box, source_size, target_size) for box in support_bboxes
            ]
            for source_box, processor_box, canvas_box in zip(
                support_bboxes, mapped_processor, mapped_canvas
            ):
                source_box_width = int(source_box[2]) - int(source_box[0])
                source_box_height = int(source_box[3]) - int(source_box[1])
                pscale_x = (processor_box[2] - processor_box[0]) / source_box_width
                pscale_y = (processor_box[3] - processor_box[1]) / source_box_height
                cscale_x = (canvas_box[2] - canvas_box[0]) / source_box_width
                cscale_y = (canvas_box[3] - canvas_box[1]) / source_box_height
                processor_placement_anisotropies.append(
                    abs(pscale_x - pscale_y) / max(pscale_x, pscale_y)
                )
                canvas_placement_anisotropies.append(
                    abs(cscale_x - cscale_y) / max(cscale_x, cscale_y)
                )
            if (
                max(processor_placement_anisotropies) > 0.02
                or max(canvas_placement_anisotropies) > 0.02
            ):
                continue
            width_ratios = [(box[2] - box[0]) / target_w for box in mapped_processor]
            height_ratios = [(box[3] - box[1]) / target_h for box in mapped_processor]
            minimum_observed = min(width_ratios + height_ratios)
            aspect_error = abs(math.log((cells_w / cells_h) / source_ratio))
            common_processor_scale = min(processor_scale_x, processor_scale_y)
            candidates.append(
                {
                    "pil_size": [target_size[0], target_size[1]],
                    "processor_size": [processor_w, processor_h],
                    "visual_tokens": geometry["visual_tokens"],
                    "source_scale_anisotropy": anisotropy,
                    "processor_scale_anisotropy": processor_anisotropy,
                    "aspect_error": aspect_error,
                    "minimum_region_axis_ratio": minimum_observed,
                    "minimum_prior_linear_gain": prior_gain,
                    "common_processor_scale": common_processor_scale,
                    "maximum_processor_placement_anisotropy": max(
                        processor_placement_anisotropies
                    ),
                    "maximum_canvas_placement_anisotropy": max(
                        canvas_placement_anisotropies
                    ),
                    "pil_pixels": target_size[0] * target_size[1],
                }
            )
    if not candidates:
        raise RuntimeError("no aspect-consistent canvas within processor ceiling")
    strict = [
        item
        for item in candidates
        if item["minimum_region_axis_ratio"] + 1e-12 >= 1.0
        and item["minimum_prior_linear_gain"] + 1e-12 >= 1.20
    ]
    if strict:
        chosen = min(
            strict,
            key=lambda item: (
                item["visual_tokens"],
                item["source_scale_anisotropy"],
                item["aspect_error"],
                item["pil_pixels"],
            ),
        )
    else:
        maximum_common_scale = max(
            item["common_processor_scale"] for item in candidates
        )
        maximum = [
            item
            for item in candidates
            if abs(item["common_processor_scale"] - maximum_common_scale) <= 1e-12
        ]
        chosen = min(
            maximum,
            key=lambda item: (
                item["pil_pixels"],
                item["visual_tokens"],
                item["source_scale_anisotropy"],
                item["aspect_error"],
            ),
        )
    return tuple(chosen["pil_size"])


def spatially_aligned_canvas_geometry(
    source_size: tuple[int, int] | list[int], selected_regions: list[dict]
) -> dict:
    """Return the spatially aligned canvas and deterministic intermediate geometry."""
    ordered_regions = sorted(
        selected_regions,
        key=lambda item: (int(item["row"]), int(item["col"]), str(item.get("id", ""))),
    )
    reference_layout = reference_layout_geometry(source_size, ordered_regions)
    token_canvas = token_balanced_canvas(
        source_size, reference_layout["total_visual_tokens"]
    )
    board_proc = reference_layout["board_geometry"]["processor_size"]
    tile = [
        BOARD_TILE_PX * board_proc[0] / reference_layout["board_size"][0],
        BOARD_TILE_PX * board_proc[1] / reference_layout["board_size"][1],
    ]
    reference = clarity_target_canvas(
        source_size, [item["bbox"] for item in ordered_regions], tile, token_canvas
    )
    canvas_size = [
        max(
            FETCH_FACTOR,
            round(reference[0] * ALIGNED_CANVAS_SCALE / FETCH_FACTOR) * FETCH_FACTOR,
        ),
        max(
            FETCH_FACTOR,
            round(reference[1] * ALIGNED_CANVAS_SCALE / FETCH_FACTOR) * FETCH_FACTOR,
        ),
    ]
    reference_boxes = [
        mapped_box(item["bbox"], source_size, reference) for item in ordered_regions
    ]
    sx, sy = canvas_size[0] / reference[0], canvas_size[1] / reference[1]
    scaled_boxes = [
        [round(box[0] * sx), round(box[1] * sy), round(box[2] * sx), round(box[3] * sy)]
        for box in reference_boxes
    ]
    return {
        "reference_layout": reference_layout,
        "token_balanced_canvas": list(token_canvas),
        "board_tile_processor_size": tile,
        "reference_canvas": list(reference),
        "spatially_aligned_canvas": canvas_size,
        "canvas_processor_geometry": processor_geometry(canvas_size),
        "reference_region_boxes": reference_boxes,
        "spatially_aligned_region_boxes": scaled_boxes,
    }
