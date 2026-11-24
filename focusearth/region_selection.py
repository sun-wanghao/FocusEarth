"""Multi-Query Region Selection with Context-first proposals and stable ranking."""

from __future__ import annotations

import numpy as np


def rank(field: np.ndarray) -> list[int]:
    values = np.asarray(field)
    if values.shape != (36,) or not np.isfinite(values).all():
        raise ValueError("one finite 36-candidate score field is required")
    return sorted(range(36), key=lambda i: (-float(values[i]), i))


def select_regions(context: np.ndarray, entities: list[np.ndarray]) -> dict:
    """Propose Context then each Entity, deduplicate, and complete Context Top-6."""
    context_rank = rank(context)
    anchors = [("Context", context_rank[0])]
    for ordinal, field in enumerate(entities, 1):
        anchors.append((f"E{ordinal}", rank(field)[0]))
    insertion = []
    for _, candidate in anchors:
        if candidate not in insertion and len(insertion) < 6:
            insertion.append(candidate)
    for candidate in context_rank:
        if len(insertion) == 6:
            break
        if candidate not in insertion:
            insertion.append(candidate)
    return {
        "insertion_order": insertion,
        "selected_ids": sorted(insertion),
        "role_anchors": anchors,
    }
