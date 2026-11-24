"""Focused deterministic checks for MME query construction."""

import unittest

from focusearth import query_processing as query
from focusearth._query_processing.context_refinement import source_token_guard
from focusearth.query_processing import reformulate_context, validate_roles


class ContextConstructionTests(unittest.TestCase):
    def test_context_description_and_route(self):
        result = reformulate_context(
            "perception/remote_sensing/color/0001",
            "color",
            "What color is the roof of the square building in the lower right area of the picture?",
        )
        self.assertEqual(result["route"], "REFORMULATED_QUERY")
        self.assertEqual(
            result["context"], "roof of the square building in the lower right area"
        )
        self.assertTrue(result["source_token_guard"])
        self.assertTrue(result["semantic_token_guard"])

    def test_context_rules_do_not_depend_on_record_identity(self):
        cases = [
            (
                "perception/remote_sensing/color/1683",
                "color",
                "What color is the roof?2",
                "roof?2",
            ),
            (
                "perception/remote_sensing/count/2573",
                "count",
                "How many vehicles are visible?2",
                "vehicles visible?2",
            ),
            (
                "perception/remote_sensing/count/3066",
                "count",
                "On the lower right side of the picture, how many tributaries does the river have?",
                "tributaries does the river have, on the lower right side",
            ),
            (
                "perception/remote_sensing/position/3387",
                "position",
                "Where does the gray ground take the color stripes in the picture?",
                "gray ground take the color stripes",
            ),
        ]
        for question_id, category, question, expected in cases:
            with self.subTest(question_id=question_id):
                result = reformulate_context(question_id, category, question)
                self.assertEqual(result["route"], "REFORMULATED_QUERY")
                self.assertEqual(result["status"], "CLEAN")
                self.assertEqual(result["context"], expected)
                self.assertEqual(result["source_insertions"], [])
                self.assertTrue(result["source_token_guard"])
                self.assertTrue(result["semantic_token_guard"])
                self.assertEqual(
                    result, reformulate_context("unseen-record", category, question)
                )

    def test_record_identity_does_not_override_question_text(self):
        question = "What color is the roof?"
        expected = reformulate_context("unseen-record", "color", question)
        for question_id in (
            "perception/remote_sensing/color/1683",
            "perception/remote_sensing/count/2573",
            "perception/remote_sensing/count/3066",
            "perception/remote_sensing/position/3387",
        ):
            with self.subTest(question_id=question_id):
                self.assertEqual(
                    reformulate_context(question_id, "color", question), expected
                )

    def test_count_transformation_preserves_source_tokens(self):
        for target, reference, location in (
            ("tributaries", "river", "lower right"),
            ("windows", "building", "upper left"),
        ):
            with self.subTest(target=target):
                question = (
                    f"On the {location} side of the picture, how many "
                    f"{target} does the {reference} have?"
                )
                result = reformulate_context("new-count", "count", question)
                self.assertEqual(
                    result["context"],
                    f"{target} does the {reference} have, on the {location} side",
                )
                self.assertEqual(result["source_insertions"], [])

    def test_source_guard_rejects_insertions_and_extra_occurrences(self):
        question = "How many tributaries does the river have?"
        self.assertEqual(
            source_token_guard(question, "tributaries does the river have"),
            (True, []),
        )
        self.assertEqual(
            source_token_guard(question, "tributaries of the river"),
            (False, ["of"]),
        )
        self.assertEqual(
            source_token_guard(question, "tributaries tributaries"),
            (False, ["tributaries"]),
        )

    def test_failed_unhandled_question_uses_original_surface(self):
        for question in ("Describe the scene.", "Identify the objects."):
            with self.subTest(question=question):
                result = reformulate_context("unseen-record", "position", question)
                self.assertEqual(result["route"], "ORIGINAL_QUESTION_FALLBACK")
                self.assertEqual(result["status"], "FAILED_UNHANDLED")
                self.assertEqual(result["context"], question)

    def test_role_order_and_duplicate_entity_text(self):
        roles = [
            {"kind": "Context", "ordinal": None, "text": "roof of building"},
            {"kind": "Entity", "ordinal": 1, "text": "roof"},
            {"kind": "Entity", "ordinal": 2, "text": "roof"},
        ]
        self.assertIs(validate_roles(roles), roles)

    def test_role_order_rejects_entity_before_context(self):
        with self.assertRaises(ValueError):
            validate_roles([{"kind": "Entity", "ordinal": 1, "text": "roof"}])

    def test_entity_occurrence_duplicates_and_placeholder_filter(self):
        class StubTarget:
            @staticmethod
            def extract(question):
                return {
                    "entity_instances": [
                        {"clean_entity_phrase": "roof"},
                        {"clean_entity_phrase": ""},
                        {"clean_entity_phrase": "-"},
                        {"clean_entity_phrase": "roof"},
                    ]
                }

        original = query._load_component
        try:
            query._load_component = lambda *args, **kwargs: StubTarget
            self.assertEqual(
                query.extract_entities("Which roof is visible?"), ["roof", "roof"]
            )
        finally:
            query._load_component = original


if __name__ == "__main__":
    unittest.main()
