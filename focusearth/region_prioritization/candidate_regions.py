"""Row-major source candidates and five source-coordinate views."""

from __future__ import annotations

import math

VIEW_ORDER = ("FULL", "TL_2OVER3", "TR_2OVER3", "BL_2OVER3", "BR_2OVER3")


def candidate_boxes(width: int, height: int) -> list[list[int]]:
    """Construct a row-major 6 × 6 grid with 5% expansion on each side."""
    if width <= 0 or height <= 0:
        raise ValueError("source dimensions must be positive")
    tile_w, tile_h = width / 6, height / 6
    overlap_w, overlap_h = tile_w * 0.1, tile_h * 0.1
    return [
        [
            max(0, int(round(c * tile_w - overlap_w / 2))),
            max(0, int(round(r * tile_h - overlap_h / 2))),
            min(width, int(round((c + 1) * tile_w + overlap_w / 2))),
            min(height, int(round((r + 1) * tile_h + overlap_h / 2))),
        ]
        for r in range(6)
        for c in range(6)
    ]


def view_box(box: list[int], view: str) -> list[int]:
    """Return FULL or a corner-anchored, ceil(2/3)-size source view."""
    x0, y0, x1, y1 = map(int, box)
    if view == "FULL":
        return [x0, y0, x1, y1]
    if view not in VIEW_ORDER:
        raise ValueError(f"unknown view: {view}")
    width, height = x1 - x0, y1 - y0
    cw, ch = max(1, math.ceil(2 * width / 3)), max(1, math.ceil(2 * height / 3))
    left = x0 if view in {"TL_2OVER3", "BL_2OVER3"} else x1 - cw
    top = y0 if view in {"TL_2OVER3", "TR_2OVER3"} else y1 - ch
    return [left, top, left + cw, top + ch]
