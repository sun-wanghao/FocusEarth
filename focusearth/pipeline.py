"""Complete one-question MME inference pipeline using caller-owned checkpoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from .evaluation.prompt import build_mme_prompt
from .models.qwen3_vl import build_processor_batch, generate_answer
from .region_prioritization.entropy_matching import entropy_matched_fusion
from .region_prioritization.candidate_regions import candidate_boxes
from .region_selection import select_regions
from .visual_representation.token_budget import compute_budget
from .visual_representation.canvas_geometry import spatially_aligned_canvas_geometry
from .visual_representation.spatially_aligned_image import (
    construct_spatially_aligned_image,
)
from .visual_representation.feature_mapping import protected_cell_indices
from .visual_representation.region_protected_compression import (
    unprotected_components,
    build_mapping_plan,
)


def prioritize_sample(sample, scorer) -> dict[str, Any]:
    """Build query roles, fresh scores, six-region selection, and canvas plan."""
    from PIL import Image
    from .query_processing import construct_mme_roles

    role_result = construct_mme_roles(
        sample.question_id, sample.category, sample.question
    )
    roles = role_result["roles"]
    with Image.open(sample.image_path) as opened:
        image = opened.convert("RGB")
    try:
        source_size = image.size
        raw_scores = scorer.score(image, [role["text"] for role in roles])
        if raw_scores.shape != (len(roles), 4, 5, 36):
            raise RuntimeError(
                f"unexpected fresh score tensor shape: {raw_scores.shape}"
            )
        fused = entropy_matched_fusion(raw_scores)
        selection = select_regions(
            fused["fused"][0], [row for row in fused["fused"][1:]]
        )
        boxes = candidate_boxes(*source_size)
        regions = [
            {
                "id": f"R{index + 1}",
                "row": index // 6,
                "col": index % 6,
                "bbox": boxes[index],
                "patch_index": index,
            }
            for index in selection["selected_ids"]
        ]
        geometry = spatially_aligned_canvas_geometry(source_size, regions)
    finally:
        image.close()
    return {
        "roles": roles,
        "context_trace": role_result["context_trace"],
        "selection": selection,
        "regions": regions,
        "geometry": geometry,
        "source_size": list(source_size),
        "role_rankings": [
            np.argsort(-field, kind="stable").tolist() for field in fused["fused"]
        ],
        "role_temperature": fused["temperatures"].tolist(),
        "target_entropy": fused["target_entropy"].tolist(),
    }


def answer_prioritized_sample(
    sample, prepared: dict[str, Any], model, processor
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct the protected visual input and generate one MME answer."""
    from PIL import Image

    source_size = tuple(prepared["source_size"])
    geometry, regions = prepared["geometry"], prepared["regions"]
    canvas_size = tuple(geometry["spatially_aligned_canvas"])
    placements = [
        {
            "support_id": region["id"],
            "source_box": region["bbox"],
            "canvas_box": canvas_box,
            "row": region["row"],
            "col": region["col"],
        }
        for region, canvas_box in zip(
            regions, geometry["spatially_aligned_region_boxes"]
        )
    ]
    with Image.open(sample.image_path) as opened:
        image = opened.convert("RGB")
    try:
        canvas, construction = construct_spatially_aligned_image(
            image, canvas_size, placements
        )
    finally:
        image.close()
    try:
        prompt = build_mme_prompt(sample.question, sample.answer_choices)
        batch, grid = build_processor_batch(processor, canvas, prompt)
    finally:
        canvas.close()
    protected, protection_audit = protected_cell_indices(placements, canvas_size, grid)
    rows, cols = grid[1] // 2, grid[2] // 2
    components = len(unprotected_components(rows, cols, set(protected)))
    budget = compute_budget(
        source_size=source_size,
        base_canvas_size=geometry["reference_canvas"],
        canvas_size=canvas_size,
        image_grid_thw=grid,
        protected_indices=protected,
        outside_component_count=components,
    )
    if not budget["valid"]:
        raise RuntimeError(
            f"protected-token budget is infeasible: {budget.get('failure')}"
        )
    plan = build_mapping_plan(rows, cols, protected, budget["retained_token_budget"])
    if not plan["valid"]:
        raise RuntimeError(f"feature mapping is infeasible: {plan.get('failure')}")
    raw_response, generation_trace = generate_answer(model, processor, batch, plan)
    prediction = {
        "question_id": sample.question_id,
        "category": sample.category,
        "question": sample.question,
        "ground_truth": sample.ground_truth,
        "raw_response": raw_response,
        "status": "OK",
    }
    trace = {
        "question_id": sample.question_id,
        "query_roles": prepared["roles"],
        "context_trace": prepared["context_trace"],
        "role_temperature": prepared["role_temperature"],
        "target_entropy": prepared["target_entropy"],
        "role_rankings": prepared["role_rankings"],
        "selection": prepared["selection"],
        "regions": regions,
        "canvas_geometry": geometry,
        "construction": construction,
        "processor_grid_thw": grid,
        "protected_token_count": len(protected),
        "feature_protection": protection_audit,
        "token_budget": budget,
        "mapping_sha256": plan["mapping_sha256"],
        "retained_tokens": plan["retained_tokens"],
        "generation": generation_trace,
    }
    return prediction, trace
