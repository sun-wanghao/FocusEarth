"""Geometry-dependent protection reserve and retained-context token budget."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from ..region_prioritization.candidate_regions import candidate_boxes

REGION_COUNT = 6
# Retained-context budget coefficient applied to the grid capacity after the protection reserve.
RHO = 0.605
OUTLINE_OUTER_PX = 8


def _map_box(
    box: list[int],
    source_size: tuple[int, int],
    base_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> list[int]:
    sw, sh = source_size
    bw, bh = base_size
    cw, ch = canvas_size
    x1, y1, x2, y2 = box
    base = [
        round(x1 * bw / sw),
        round(y1 * bh / sh),
        round(x2 * bw / sw),
        round(y2 * bh / sh),
    ]
    return [
        round(base[0] * cw / bw),
        round(base[1] * ch / bh),
        round(base[2] * cw / bw),
        round(base[3] * ch / bh),
    ]


@lru_cache(maxsize=None)
def _standard_costs_cached(
    source: tuple[int, int],
    base: tuple[int, int],
    canvas: tuple[int, int],
    grid: tuple[int, int, int],
) -> tuple[int, ...]:
    """Geometry-only candidate boxes; protection mapping is applied by caller."""
    canvas_w, canvas_h = canvas
    result = []
    for i, source_box in enumerate(candidate_boxes(*source)):
        box = _map_box(source_box, source, base, canvas)
        envelope = [
            box[0] - OUTLINE_OUTER_PX,
            box[1] - OUTLINE_OUTER_PX,
            box[2] + OUTLINE_OUTER_PX,
            box[3] + OUTLINE_OUTER_PX,
        ]
        if (
            envelope[0] >= 0
            and envelope[1] >= 0
            and envelope[2] <= canvas_w
            and envelope[3] <= canvas_h
        ):
            result.append(i)
    return tuple(result)


@lru_cache(maxsize=None)
def _standard_cost_values_cached(
    source: tuple[int, int],
    base: tuple[int, int],
    canvas: tuple[int, int],
    grid: tuple[int, int, int],
) -> tuple[int, ...]:
    from .feature_mapping import protected_cell_indices

    values = []
    for i in _standard_costs_cached(source, base, canvas, grid):
        box = _map_box(candidate_boxes(*source)[i], source, base, canvas)
        values.append(
            len(
                protected_cell_indices(
                    [{"support_id": f"STD_R{i+1}", "canvas_box": box}], canvas, grid
                )[0]
            )
        )
    return tuple(values)


def compute_budget(
    *,
    source_size: tuple[int, int] | list[int],
    base_canvas_size: tuple[int, int] | list[int],
    canvas_size: tuple[int, int] | list[int],
    image_grid_thw: tuple[int, int, int] | list[int],
    protected_indices: list[int],
    outside_component_count: int,
) -> dict[str, Any]:
    """Compute the protection reserve and retained-token target for one question.

    ``Pstd`` is the maximum protected-cell cost among canonical
    patches whose 8px outer envelope is represented on all four canvas sides.
    The max is a deterministic conservative convention for grid quantization.
    """
    grid = list(map(int, image_grid_thw))
    if grid[0] != 1 or grid[1] % 2 or grid[2] % 2:
        raise ValueError(f"unsupported image grid: {image_grid_thw}")
    rows, cols = grid[1] // 2, grid[2] // 2
    grid_token_count = rows * cols
    canvas_w, canvas_h = map(int, canvas_size)
    source = tuple(map(int, source_size))
    base = tuple(map(int, base_canvas_size))
    canvas = (canvas_w, canvas_h)
    standard_costs: list[int] = []
    standard_costs.extend(
        _standard_cost_values_cached(source, base, canvas, tuple(grid))
    )
    if not standard_costs:
        raise ValueError("no fully represented standard patch for current geometry")
    standard_region_cost = max(standard_costs)
    protection_reserve = REGION_COUNT * standard_region_cost
    if protection_reserve > grid_token_count:
        return {
            "valid": False,
            "failure": "STANDARD_RESERVE_EXCEEDS_AVAILABLE_GRID",
            "grid_token_count": grid_token_count,
            "standard_region_cost": standard_region_cost,
            "region_count": REGION_COUNT,
            "protection_reserve": protection_reserve,
            "rho": RHO,
        }
    minimum_context_groups = int(outside_component_count)
    base_context_budget = max(
        minimum_context_groups,
        int(math.ceil(RHO * (grid_token_count - protection_reserve))),
    )
    retained_token_budget = min(
        grid_token_count, protection_reserve + base_context_budget
    )
    protected_token_count = len(set(map(int, protected_indices)))
    reserve_rebate = protection_reserve - protected_token_count
    context_group_budget = base_context_budget + reserve_rebate
    valid = (
        protected_token_count <= protection_reserve
        and context_group_budget >= minimum_context_groups
        and retained_token_budget <= grid_token_count
        and protected_token_count + context_group_budget == retained_token_budget
        and context_group_budget == base_context_budget + reserve_rebate
    )
    return {
        "valid": bool(valid),
        "failure": None if valid else "BUDGET_IDENTITY_OR_SAFETY_FAILURE",
        "grid_token_count": grid_token_count,
        "standard_region_cost": standard_region_cost,
        "standard_patch_costs": sorted(set(standard_costs)),
        "fully_represented_standard_patch_count": len(standard_costs),
        "region_count": REGION_COUNT,
        "protection_reserve": protection_reserve,
        "rho": RHO,
        "minimum_context_groups": minimum_context_groups,
        "base_context_budget": base_context_budget,
        "retained_token_budget": retained_token_budget,
        "protected_token_count": protected_token_count,
        "reserve_rebate": reserve_rebate,
        "context_group_budget": context_group_budget,
        "checks": {
            "protected_within_reserve": protected_token_count <= protection_reserve,
            "context_covers_components": context_group_budget >= minimum_context_groups,
            "budget_within_grid": retained_token_budget <= grid_token_count,
            "protected_plus_context_equals_budget": protected_token_count
            + context_group_budget
            == retained_token_budget,
            "context_equals_base_plus_rebate": context_group_budget
            == base_context_budget + reserve_rebate,
        },
    }
