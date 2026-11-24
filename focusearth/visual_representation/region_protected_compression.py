"""Connected visual-token groups with protected identity cells and spatial representatives."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from ..io_utils import canonical_json_sha256


NEIGHBORS = ((-1, 0), (0, -1), (0, 1), (1, 0))
MAPPING_RULE_ID = "OUTSIDE_COMPONENT_MULTISOURCE_BFS_MEAN_MEDOID"


def _coords(index: int, cols: int) -> tuple[int, int]:
    return divmod(index, cols)


def _morton_key(index: int, cols: int) -> tuple[int, int]:
    row, col = _coords(index, cols)
    value = 0
    for bit in range(max(row.bit_length(), col.bit_length(), 1)):
        value |= ((col >> bit) & 1) << (2 * bit)
        value |= ((row >> bit) & 1) << (2 * bit + 1)
    return value, index


def unprotected_components(
    rows: int, cols: int, protected: set[int]
) -> list[list[int]]:
    unseen = set(range(rows * cols)) - protected
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            row, col = _coords(current, cols)
            for dr, dc in NEIGHBORS:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbor = nr * cols + nc
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda values: values[0])
    return components


def _allocate_group_counts(components: list[list[int]], groups: int) -> list[int]:
    if not components:
        if groups:
            raise ValueError("groups requested without outside cells")
        return []
    if groups < len(components):
        raise ValueError(
            "budget cannot retain one representative per outside component"
        )
    if groups > sum(map(len, components)):
        raise ValueError("more groups than outside cells")
    total = sum(map(len, components))
    ideal = [groups * len(component) / total for component in components]
    counts = [
        min(len(component), max(1, int(value)))
        for component, value in zip(components, ideal)
    ]
    while sum(counts) > groups:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        if not candidates:
            raise ValueError("cannot reduce component allocations")
        chosen = min(candidates, key=lambda i: (ideal[i] - counts[i], i))
        counts[chosen] -= 1
    while sum(counts) < groups:
        candidates = [
            i
            for i, (component, count) in enumerate(zip(components, counts))
            if count < len(component)
        ]
        if not candidates:
            raise ValueError("cannot increase component allocations")
        chosen = max(candidates, key=lambda i: (ideal[i] - counts[i], -i))
        counts[chosen] += 1
    return counts


def _quantile_seeds(component: list[int], count: int, cols: int) -> list[int]:
    ordered = sorted(component, key=lambda index: _morton_key(index, cols))
    positions = [
        min(len(ordered) - 1, ((2 * rank + 1) * len(ordered)) // (2 * count))
        for rank in range(count)
    ]
    seeds = [ordered[position] for position in positions]
    if len(set(seeds)) != count:
        raise RuntimeError("quantile seed collision")
    return sorted(seeds)


def _multisource_groups(
    component: list[int], group_count: int, rows: int, cols: int
) -> list[list[int]]:
    allowed = set(component)
    seeds = _quantile_seeds(component, group_count, cols)
    labels = {seed: rank for rank, seed in enumerate(seeds)}
    queue = deque(seeds)
    while queue:
        current = queue.popleft()
        label = labels[current]
        row, col = _coords(current, cols)
        for dr, dc in NEIGHBORS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor = nr * cols + nc
                if neighbor in allowed and neighbor not in labels:
                    labels[neighbor] = label
                    queue.append(neighbor)
    if len(labels) != len(component):
        raise RuntimeError("component labeling incomplete")
    groups = [[] for _ in seeds]
    for index in sorted(component):
        groups[labels[index]].append(index)
    if any(not group for group in groups):
        raise RuntimeError("empty outside group")
    return groups


def _representative_medoid(group: list[int], cols: int) -> int:
    sum_row = sum(index // cols for index in group)
    sum_col = sum(index % cols for index in group)
    count = len(group)
    return min(
        group,
        key=lambda index: (
            (index // cols * count - sum_row) ** 2
            + (index % cols * count - sum_col) ** 2,
            index,
        ),
    )


def _is_connected(group: Iterable[int], rows: int, cols: int) -> bool:
    values = set(group)
    if not values:
        return False
    start = min(values)
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        row, col = _coords(current, cols)
        for dr, dc in NEIGHBORS:
            nr, nc = row + dr, col + dc
            neighbor = nr * cols + nc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and neighbor in values
                and neighbor not in seen
            ):
                seen.add(neighbor)
                queue.append(neighbor)
    return seen == values


def build_mapping_plan(
    rows: int,
    cols: int,
    protected_indices: list[int],
    target_budget: int,
) -> dict:
    original = rows * cols
    protected = set(protected_indices)
    if len(protected) != len(protected_indices) or any(
        index < 0 or index >= original for index in protected
    ):
        raise ValueError("invalid protected indices")
    outside = original - len(protected)
    requested_target = max(len(protected), int(target_budget))
    retained_target = min(original, requested_target)
    outside_group_target = retained_target - len(protected)
    components = unprotected_components(rows, cols, protected)
    valid = not outside or outside_group_target >= len(components)
    failure = None if valid else "OUTSIDE_COMPONENT_COUNT_EXCEEDS_AVAILABLE_BUDGET"
    outside_groups: list[list[int]] = []
    if valid and outside:
        allocation = _allocate_group_counts(components, outside_group_target)
        for component, count in zip(components, allocation):
            outside_groups.extend(_multisource_groups(component, count, rows, cols))
    protected_groups = [
        {"kind": "protected_identity", "members": [index], "representative": index}
        for index in protected_indices
    ]
    merged_groups = [
        {
            "kind": "outside_arithmetic_mean",
            "members": group,
            "representative": _representative_medoid(group, cols),
        }
        for group in outside_groups
    ]
    output_groups = sorted(
        protected_groups + merged_groups,
        key=lambda row: (row["representative"], row["kind"]),
    )
    representatives = [row["representative"] for row in output_groups]
    partition = [index for row in output_groups for index in row["members"]]
    connectivity_ok = all(
        _is_connected(row["members"], rows, cols) for row in merged_groups
    )
    exact_partition = sorted(partition) == list(range(original)) and len(
        partition
    ) == len(set(partition))
    protected_identity = all(
        row["members"] == [row["representative"]] for row in protected_groups
    )
    plan_core = {
        "mapping_rule_id": MAPPING_RULE_ID,
        "grid": [rows, cols],
        "original_tokens": original,
        "protected_tokens": len(protected),
        "outside_tokens": outside,
        "target_budget": int(target_budget),
        "requested_target_budget": requested_target,
        "retained_tokens": len(output_groups),
        "target_above_original": requested_target > original,
        "protected_exceeds_target_budget": len(protected) > int(target_budget),
        "outside_component_count": len(components),
        "outside_group_count": len(merged_groups),
        "outside_group_sizes": [len(row["members"]) for row in merged_groups],
        "representative_indices": representatives,
        "output_groups": output_groups,
        "valid": valid and connectivity_ok and exact_partition and protected_identity,
        "failure": failure,
        "checks": {
            "outside_groups_four_neighbor_connected": connectivity_ok,
            "exact_full_grid_partition": exact_partition,
            "protected_cells_identity_singletons": protected_identity,
            "retained_equals_min_original_target": (
                len(output_groups) == retained_target if valid else False
            ),
            "deterministic_output_order": representatives == sorted(representatives),
        },
        "merge": {
            "protected": "feature vector retained byte-for-byte/tensor-value identity",
            "outside": "members sorted by original row-major index; arithmetic mean accumulated in float32 then cast to stream dtype",
            "representative_position": "member nearest arithmetic coordinate centroid by squared distance; row-major tie break",
            "serialization": "all output groups sorted by representative original row-major index",
        },
    }
    plan_core["mapping_sha256"] = canonical_json_sha256(plan_core)
    return plan_core
