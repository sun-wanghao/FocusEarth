"""Spatially Aligned Image Construction from one bilinear full-scene resize."""

from __future__ import annotations

from typing import Any

from .feature_mapping import OUTLINE_INNER_PX, OUTLINE_OUTER_PX


SPATIAL_REPRESENTATION_ID = (
    "FULL_SOURCE_DIRECT_BILINEAR_SAME_CANVAS_OUTSIDE_ONLY_OUTLINES"
)


def _filled_rectangle(draw, box: tuple[int, int, int, int], value: int) -> None:
    x1, y1, x2, y2 = box
    if x2 > x1 and y2 > y1:
        # PIL mode "1" masks require 255 for a true bit; fill=1 is a distinct
        # low-valued pixel and breaks logical inversion semantics.
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255 if value else 0)


def construct_spatially_aligned_image(
    source, target_size: tuple[int, int], placements: list[dict[str, Any]]
):
    """Resize the full scene and draw outer outlines without replacing region pixels."""
    from PIL import Image, ImageChops, ImageDraw

    target_size = tuple(map(int, target_size))
    sharp = source.resize(target_size, Image.Resampling.BILINEAR).convert("RGB")
    width, height = target_size
    union = Image.new("1", target_size, 0)
    white = Image.new("1", target_size, 0)
    black = Image.new("1", target_size, 0)
    union_draw = ImageDraw.Draw(union)
    white_draw = ImageDraw.Draw(white)
    black_draw = ImageDraw.Draw(black)
    for placement in placements:
        x1, y1, x2, y2 = map(int, placement["canvas_box"])
        _filled_rectangle(union_draw, (x1, y1, x2, y2), 1)
        for draw, outer, inner in (
            (white_draw, OUTLINE_OUTER_PX, OUTLINE_INNER_PX),
            (black_draw, OUTLINE_INNER_PX, 0),
        ):
            ex = (
                max(0, x1 - outer),
                max(0, y1 - outer),
                min(width, x2 + outer),
                min(height, y2 + outer),
            )
            inner_box = (
                max(0, x1 - inner),
                max(0, y1 - inner),
                min(width, x2 + inner),
                min(height, y2 + inner),
            )
            _filled_rectangle(draw, ex, 1)
            _filled_rectangle(draw, inner_box, 0)
    black = ImageChops.logical_and(black, ImageChops.invert(white))
    outside_union = ImageChops.invert(union)
    white = ImageChops.logical_and(white, outside_union)
    black = ImageChops.logical_and(black, outside_union)
    # Masks are complete at this point, so painting the resized RGB image in
    # place avoids a second full-canvas RGB allocation without changing pixels.
    output = sharp
    output.paste((235, 235, 235), mask=white)
    output.paste((32, 32, 32), mask=black)
    return output, {
        "construction_id": SPATIAL_REPRESENTATION_ID,
        "source_resize": "direct PIL bilinear full-scene resize to the spatially aligned canvas",
        "canvas_size": list(target_size),
        "placements": placements,
        "outline_outer_px": OUTLINE_OUTER_PX,
        "outline_inner_px": OUTLINE_INNER_PX,
        "outline_contract": "sequential masks; black minus white; both masks exclude the selected-region union",
        "union_pixel_replacement": False,
        "background": "direct full-source raster",
    }
