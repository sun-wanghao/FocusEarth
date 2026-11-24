"""Deterministic checks for region prioritization and visual representation."""

import unittest

import numpy as np
from PIL import Image

from focusearth.region_prioritization.candidate_regions import candidate_boxes, view_box
from focusearth.region_prioritization.entropy_matching import entropy_matched_fusion
from focusearth.region_selection import select_regions
from focusearth.visual_representation.canvas_geometry import (
    mapped_box,
    processor_geometry,
)
from focusearth.visual_representation.spatially_aligned_image import (
    construct_spatially_aligned_image,
)
from focusearth.visual_representation.token_budget import RHO, compute_budget
from focusearth.visual_representation.feature_mapping import (
    canvas_rect_to_merged_cells,
    protected_cell_indices,
)
from focusearth.visual_representation.region_protected_compression import (
    build_mapping_plan,
)


class ScoringTests(unittest.TestCase):
    def test_candidate_geometry_views_and_bounds(self):
        boxes = candidate_boxes(600, 300)
        self.assertEqual(len(boxes), 36)
        self.assertEqual(boxes[0], [0, 0, 105, 52])
        self.assertEqual(boxes[-1], [495, 248, 600, 300])
        x0, y0, x1, y1 = boxes[14]
        self.assertEqual(view_box(boxes[14], "FULL"), boxes[14])
        corner = view_box(boxes[14], "BR_2OVER3")
        self.assertEqual(corner[2] - corner[0], 74)
        self.assertEqual(corner[3] - corner[1], 36)
        self.assertEqual(corner[2:], [x1, y1])
        self.assertGreaterEqual(corner[0], x0)
        self.assertGreaterEqual(corner[1], y0)

    def test_entropy_fusion_is_normalized_deterministic_and_finite(self):
        raw = np.arange(4 * 5 * 36, dtype=np.float32).reshape(1, 4, 5, 36) / 100
        one = entropy_matched_fusion(raw)
        two = entropy_matched_fusion(raw)
        np.testing.assert_array_equal(one["fused"], two["fused"])
        self.assertAlmostEqual(float(one["fused"].sum()), 1.0, places=12)
        self.assertTrue(
            np.all((one["temperatures"] >= 0.25) & (one["temperatures"] <= 4))
        )

    def test_selection_anchor_order_dedup_completion_and_serialization(self):
        context = np.arange(36, dtype=float)
        entities = [context[::-1], np.roll(context, -2)]
        got = select_regions(context, entities)
        self.assertEqual(got["role_anchors"], [("Context", 35), ("E1", 0), ("E2", 33)])
        self.assertEqual(got["insertion_order"], [35, 0, 33, 34, 32, 31])
        self.assertEqual(got["selected_ids"], [0, 31, 32, 33, 34, 35])


class GeometryAndCompressionTests(unittest.TestCase):
    def test_processor_grid_and_coordinate_rounding(self):
        geometry = processor_geometry((1024, 768))
        self.assertEqual(geometry["processor_size"], [1024, 768])
        self.assertEqual(geometry["image_grid_thw"], [1, 48, 64])
        self.assertEqual(mapped_box([1, 2, 9, 10], (10, 10), (28, 28)), [3, 6, 25, 28])

    def test_half_open_feature_cell_boundary_and_canvas_edges(self):
        self.assertEqual(
            canvas_rect_to_merged_cells([32, 32, 33, 33], (64, 64), (2, 2)),
            (1, 1, 2, 2),
        )
        self.assertEqual(
            canvas_rect_to_merged_cells([0, 0, 32, 32], (64, 64), (2, 2)), (0, 0, 1, 1)
        )
        self.assertEqual(
            canvas_rect_to_merged_cells([63, 63, 64, 64], (64, 64), (2, 2)),
            (1, 1, 2, 2),
        )

    def test_protected_cells_and_mapping_partition_are_deterministic(self):
        canvas = (256, 256)
        grid = (1, 16, 16)
        placements = [{"support_id": "R1", "canvas_box": [32, 32, 96, 96]}]
        protected, _ = protected_cell_indices(placements, canvas, grid)
        self.assertTrue(protected)
        first = build_mapping_plan(8, 8, protected, 32)
        second = build_mapping_plan(8, 8, protected, 32)
        self.assertTrue(first["valid"], first.get("failure"))
        self.assertEqual(first["mapping_sha256"], second["mapping_sha256"])
        self.assertEqual(first["retained_tokens"], 32)
        protected_groups = [
            g for g in first["output_groups"] if g["kind"] == "protected_identity"
        ]
        self.assertEqual(len(protected_groups), len(protected))

    def test_token_budget_identities_and_feasibility(self):
        canvas = (608, 608)
        grid = (1, 38, 38)
        placements = [{"support_id": "R1", "canvas_box": [208, 208, 400, 400]}]
        protected, _ = protected_cell_indices(placements, canvas, grid)
        components = len(
            __import__(
                "focusearth.visual_representation.region_protected_compression",
                fromlist=["_outside_components"],
            ).unprotected_components(19, 19, set(protected))
        )
        result = compute_budget(
            source_size=(608, 608),
            base_canvas_size=canvas,
            canvas_size=canvas,
            image_grid_thw=grid,
            protected_indices=protected,
            outside_component_count=components,
        )
        self.assertEqual(RHO, 0.605)
        self.assertTrue(result["valid"], result["failure"])
        self.assertEqual(
            result["protection_reserve"],
            result["region_count"] * result["standard_region_cost"],
        )
        self.assertEqual(
            result["retained_token_budget"],
            result["protected_token_count"] + result["context_group_budget"],
        )
        self.assertEqual(
            result["base_context_budget"],
            max(
                result["minimum_context_groups"],
                __import__("math").ceil(
                    RHO * (result["grid_token_count"] - result["protection_reserve"])
                ),
            ),
        )

    def test_spatially_aligned_image_preserves_scene_and_draws_outline(self):
        source = Image.new("RGB", (8, 8), "white")
        image, details = construct_spatially_aligned_image(
            source, (16, 16), [{"support_id": "R1", "canvas_box": [4, 4, 12, 12]}]
        )
        self.assertEqual(image.size, (16, 16))
        self.assertTrue(details)
        self.assertNotEqual(image.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
