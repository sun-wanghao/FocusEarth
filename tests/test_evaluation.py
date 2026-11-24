"""Synthetic MME fixtures for official annotation, prompt, and parser contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from focusearth.evaluation import (
    build_mme_prompt,
    compute_metrics,
    load_mme_dataset,
    parse_mme_answer,
    score_response,
)


class AnswerExtractionTests(unittest.TestCase):
    def test_letter_parentheses_punctuation(self):
        for text in ("A", " a ", "(A)", "A.", "(a).", "A))"):
            self.assertEqual(parse_mme_answer(text).answer, "A", text)
        for text in ("Answer: A", "I select A", "AB", "A because visible", "(A):"):
            self.assertFalse(parse_mme_answer(text).valid, text)

    def test_tag_json_line_precedence(self):
        parsed = parse_mme_answer('prefix {"response":"B"} <answer>C</answer>')
        self.assertEqual((parsed.answer, parsed.extraction_method), ("C", "answer_tag"))
        parsed = parse_mme_answer('{"response":"B","answer":"C"}')
        self.assertEqual((parsed.answer, parsed.extraction_method), ("B", "json"))
        self.assertEqual(parse_mme_answer("\n C\nD").answer, "C")
        self.assertEqual(parse_mme_answer('D\n{"answer":"B"}').answer, "B")
        self.assertFalse(parse_mme_answer("<answer>Evidence</answer>\nA").valid)

    def test_structured_candidates_and_fallback_contract(self):
        cases = (
            ("<ANSWER>\n (e). \n</ANSWER>", "(e).", "answer_tag", "E"),
            ("<answer></answer>\nA", "", "answer_tag", None),
            ("<answer>B</answer><answer>C</answer>", "B", "answer_tag", "B"),
            ('<answer>unfinished\n{"answer":"D"}', "D", "json", "D"),
            ('{"final_answer":"C"}', "C", "json", "C"),
            ('{"prediction":"D"}', "D", "json", "D"),
            ('{"answer":"(E)","prediction":"B"}', "(E)", "json", "E"),
            ('{"response":null,"answer":"B"}', "None", "json", None),
            ('{"answer":["A"]}', "['A']", "json", None),
            ('{"answer":"C","response":"D"}', "D", "json", "D"),
            ('17\n{"answer":"A"}', "A", "json", "A"),
            ('"answer"\n{"answer":"A"}', "A", "json", "A"),
            ('A\n{bad}\n{"answer":"B"}', "A", "first_line", "A"),
            ('C\n{"other":"A"}\n{"answer":"B"}', "C", "first_line", "C"),
            ('B\n{"answer":{"answer":"C"}}', "B", "first_line", "B"),
            ("\n\t\n", "", "empty", None),
        )
        for raw, candidate, method, answer in cases:
            with self.subTest(raw=raw):
                parsed = parse_mme_answer(raw)
                self.assertEqual(parsed.extracted_candidate, candidate)
                self.assertEqual(parsed.extraction_method, method)
                self.assertEqual(parsed.answer, answer)
                self.assertEqual(parsed.valid, answer is not None)

    def test_legacy_punctuation_grammar(self):
        for option in "ABCDE":
            for prefix in ("", "("):
                for suffix in ("", ")", ".", ").", "))"):
                    raw = prefix + option.lower() + suffix
                    with self.subTest(raw=raw):
                        self.assertEqual(parse_mme_answer(raw).answer, option)
        for raw in ("A..", "A.)", "((A))", "A,", "F", "A B"):
            with self.subTest(raw=raw):
                self.assertFalse(parse_mme_answer(raw).valid)

    def test_final_answer_takes_precedence_over_legacy_formats(self):
        parsed = parse_mme_answer('<answer>A</answer>\n{"answer":"B"}\nFinal answer: C')
        self.assertEqual(
            (parsed.answer, parsed.extraction_method), ("C", "final_answer_line")
        )
        for final in ("Final answer:", "Final answer D", "Final answer: D or E"):
            with self.subTest(final=final):
                parsed = parse_mme_answer(f"<answer>A</answer>\n{final}")
                self.assertEqual(parsed.extraction_method, "final_answer_line")
                self.assertFalse(parsed.valid)

    def test_malformed_json_limits_use_line_fallback(self):
        cases = (
            'A\n{"answer":' + "9" * 5000 + "}",
            "B\n" + "{" * 2000 + "}",
        )
        for raw in cases:
            with self.subTest(first_line=raw[0]):
                parsed = parse_mme_answer(raw)
                self.assertEqual(
                    (parsed.answer, parsed.extraction_method), (raw[0], "first_line")
                )

    def test_malformed_and_evidence_format_remain_invalid(self):
        for text in ('{"answer":B}', "", "\n", "Evidence: a red roof\nFinal answer:"):
            result = score_response(text, "A")
            self.assertFalse(result["valid"], text)
            self.assertFalse(result["correct"], text)
        with self.assertRaises(TypeError):
            parse_mme_answer(None)

    def test_evidence_final_answer_accepts_each_option(self):
        for option in "ABCDE":
            with self.subTest(option=option):
                text = f"Evidence: Several objects marked A-E are visible.\nFinal answer: {option}"
                parsed = parse_mme_answer(text)
                self.assertEqual(
                    (parsed.answer, parsed.extraction_method, parsed.valid),
                    (option, "final_answer_line", True),
                )

    def test_evidence_final_answer_whitespace_and_punctuation(self):
        cases = {
            "Evidence: A road and B buildings.\nFinal answer: B": "B",
            "Evidence: C vehicles are beside D buildings.\n  Final answer :   (c).  ": "C",
            "Evidence: E is a letter in the scene.\r\nFINAL ANSWER:\tE)": "E",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_mme_answer(text).answer, expected)

    def test_final_answer_missing_malformed_or_truncated_fails_closed(self):
        invalid = (
            "Evidence: A and B are visible, C is near them.",
            "Evidence: A-E appear in a sign.\nFinal answer:",
            "Evidence: red car.\nFinal answer: B or C",
            "Evidence: red car.\nFinal answer B",
            "Evidence: red car.\nFinal answer: option B",
            "Evidence: red car.\nFinal answer: B\nFinal answer: C",
            "Evidence: red car.\nFinal answer: (B",
            "Evidence: red car.\nFinal answer: B\ntruncated",
        )
        for text in invalid:
            with self.subTest(text=text):
                parsed = parse_mme_answer(text)
                self.assertFalse(parsed.valid)
                self.assertIsNone(parsed.answer)

    def test_metric_denominator_invalid_and_failed(self):
        rows = [
            {
                "question_id": "a",
                "category": "color",
                "raw_response": "A",
                "ground_truth": "A",
            },
            {
                "question_id": "b",
                "category": "color",
                "raw_response": "invalid",
                "ground_truth": "B",
            },
            {
                "question_id": "c",
                "category": "count",
                "raw_response": "B",
                "ground_truth": "B",
                "status": "ERROR",
            },
            {
                "question_id": "d",
                "category": "position",
                "raw_response": "C",
                "ground_truth": "D",
            },
        ]
        result = compute_metrics(rows, expected_rows=4)
        self.assertEqual(
            (result["correct"], result["total"], result["invalid"], result["failed"]),
            (1, 4, 1, 1),
        )
        self.assertEqual(result["categories"]["color"]["accuracy"], 0.5)
        with self.assertRaises(ValueError):
            compute_metrics(rows, expected_rows=3738)
        with self.assertRaises(ValueError):
            compute_metrics(rows + [rows[0]])


class DatasetAndPromptTests(unittest.TestCase):
    def fixture(self):
        return {
            "Subtask": "Remote Sensing",
            "Question_id": "synthetic/color/1",
            "Image": "remote_sensing/example.png",
            "Text": "Which color?",
            "Answer choices": ["(A) Red", "(B) Blue", "(C) Gray", "(D) Green", "(E) "],
            "Ground truth": "A",
            "Category": "color",
        }

    def test_dataset_preserves_original_values_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "remote_sensing/example.png"
            image.parent.mkdir()
            image.write_bytes(b"synthetic placeholder; loader does not decode")
            path = root / "annotations.json"
            path.write_text(json.dumps([{"Subtask": "Other"}, self.fixture()]))
            rows = load_mme_dataset(root, path, strict_population=False)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].answer_choices[-1], "(E) ")
            self.assertEqual(rows[0].annotation_index, 1)
            with self.assertRaises(ValueError):
                load_mme_dataset(root, path)
            row = self.fixture()
            row["Image"] = "../private.png"
            path.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):
                load_mme_dataset(
                    root, path, strict_population=False, verify_images=False
                )

    def test_exact_prompt_spacing_and_original_question_choices(self):
        row = self.fixture()
        text = build_mme_prompt(row["Text"], row["Answer choices"])
        self.assertIn(
            "Question: Which color? The choices are listed below:\n(A) Red\n(B) Blue",
            text,
        )
        self.assertIn(
            "Use the following response format:\n\nEvidence: [A concise, image-grounded observation-based explanation]",
            text,
        )
        self.assertTrue(text.endswith("Final answer: [A single MME option letter]"))
        self.assertIn(
            "Image 1 is a single full-scene image obtained by bilinearly resizing the source image.",
            text,
        )


if __name__ == "__main__":
    unittest.main()
