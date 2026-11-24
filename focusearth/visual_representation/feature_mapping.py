"""Map half-open canvas rectangles onto intersecting post-merger feature cells."""

from __future__ import annotations

from typing import Any


PATCH_SIZE = 16
MERGE_SIZE = 2
MERGED_CELL_PIXELS = PATCH_SIZE * MERGE_SIZE
OUTLINE_OUTER_PX = 8
OUTLINE_INNER_PX = 4


def ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


def canvas_rect_to_merged_cells(
    rect: tuple[int, int, int, int] | list[int],
    canvas_size: tuple[int, int] | list[int],
    merged_grid: tuple[int, int] | list[int],
) -> tuple[int, int, int, int]:
    """Map a half-open canvas rectangle to intersecting half-open merged-cell bounds.

    Geometry is deterministic integer arithmetic. It protects cells whose 32x32
    resized-pixel footprint geometrically intersects the scaled rectangle.
    """
    x1, y1, x2, y2 = map(int, rect)
    width, height = map(int, canvas_size)
    rows, cols = map(int, merged_grid)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError((rect, canvas_size))
    column_start = max(0, min(cols, (x1 * cols) // width))
    row_start = max(0, min(rows, (y1 * rows) // height))
    column_end = max(column_start + 1, min(cols, ceil_div(x2 * cols, width)))
    row_end = max(row_start + 1, min(rows, ceil_div(y2 * rows, height)))
    return row_start, column_start, row_end, column_end


def protected_cell_indices(
    placements: list[dict[str, Any]],
    canvas_size: tuple[int, int] | list[int],
    image_grid_thw: tuple[int, int, int] | list[int],
    outer_px: int = OUTLINE_OUTER_PX,
) -> tuple[list[int], dict[str, Any]]:
    """Return a conservative exact superset of union/ring-intersecting cells.

    For each selected box, union plus both outside outline rings lies inside the
    box's 8-pixel outer envelope. Protecting every merged cell intersecting that
    envelope guarantees no union or rendered outline cell is merged. It may
    retain extra cells where another box's sequential outline mask erases an
    overlapping halo; these additional cells remain protected.
    """
    t, patch_rows, patch_cols = map(int, image_grid_thw)
    if t != 1 or patch_rows % MERGE_SIZE or patch_cols % MERGE_SIZE:
        raise ValueError(f"unsupported image grid {image_grid_thw}")
    rows, cols = patch_rows // MERGE_SIZE, patch_cols // MERGE_SIZE
    width, height = map(int, canvas_size)
    protected: set[int] = set()
    mapped_envelopes = []
    for placement in placements:
        x1, y1, x2, y2 = map(int, placement["canvas_box"])
        envelope = [
            max(0, x1 - outer_px),
            max(0, y1 - outer_px),
            min(width, x2 + outer_px),
            min(height, y2 + outer_px),
        ]
        row_start, column_start, row_end, column_end = canvas_rect_to_merged_cells(
            envelope, (width, height), (rows, cols)
        )
        for row in range(row_start, row_end):
            base = row * cols
            protected.update(range(base + column_start, base + column_end))
        mapped_envelopes.append(
            {
                "support_id": placement["support_id"],
                "canvas_envelope": envelope,
                "merged_cell_bounds": [row_start, column_start, row_end, column_end],
            }
        )
    return sorted(protected), {
        "protection_rule": "UNION_PLUS_BOTH_OUTLINES_CONSERVATIVE_OUTER_ENVELOPE",
        "outer_px": outer_px,
        "merged_grid": [rows, cols],
        "mapped_envelopes": mapped_envelopes,
        "safe_overprotection_note": "May retain cells in an outline halo erased by another box; never omits a union or rendered-ring intersection.",
    }
