"""MME option extraction and pooled question-level accuracy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
import json
import re
from typing import Any

from .mme_dataset import CATEGORY_COUNTS

_FINAL_LINE = re.compile(
    r"^[ \t]*Final[ \t]+answer\b([^\r\n]*)", re.IGNORECASE | re.MULTILINE
)
_FINAL_VALUE = re.compile(r"[ \t]*:[ \t]*(.*)")
_FINAL_OPTION = re.compile(r"^[ \t]*(?:([A-E])|\(([A-E])\))[.)]?[ \t]*$", re.IGNORECASE)
PARSER_PROTOCOL = "mme_explicit_final_answer_then_structured_formats"


@dataclass(frozen=True)
class ParsedAnswer:
    extracted_candidate: str
    extraction_method: str
    answer: str | None
    valid: bool


def normalize_option(candidate: str | None) -> str | None:
    """Validate one A–E letter with the supported legacy option punctuation."""
    if candidate is None:
        return None
    token = candidate.strip().removeprefix("(")
    if not token or token[0].upper() not in "ABCDE":
        return None
    return token[0].upper() if token[1:] in ("", ")", ".", ").", "))") else None


def _legacy_candidate(text: str) -> tuple[str, str]:
    """Select an answer-tag value, JSON field, or first nonempty line."""
    content_start = None
    for boundary in re.finditer(r"</?answer>", text, re.IGNORECASE):
        closing = boundary.group()[1] == "/"
        if content_start is None and not closing:
            content_start = boundary.end()
        elif content_start is not None and closing:
            return text[content_start : boundary.start()].strip(), "answer_tag"

    start = text.find("{")
    end = text.find("}", start + 1) if start >= 0 else -1
    if end >= 0:
        try:
            fields = json.loads(text[start : end + 1])
        except (ValueError, RecursionError):
            fields = {}
        if isinstance(fields, dict):
            for name in ("response", "answer", "final_answer", "prediction"):
                if name in fields:
                    return str(fields[name]).strip(), "json"

    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()), None
    )
    return (first_line, "first_line") if first_line is not None else ("", "empty")


def parse_mme_answer(raw_response: str) -> ParsedAnswer:
    """Extract an explicit Final answer line before answer-tag, JSON, or first-line formats.

    A missing value, malformed label, or multiple final-answer lines fails
    closed, without falling back to a letter in Evidence or another format.
    Without this label, retain answer-tag, JSON, and first-line precedence.
    """
    if not isinstance(raw_response, str):
        raise TypeError("raw_response must be the original decoded string")
    text = raw_response.strip()
    final_matches = list(_FINAL_LINE.finditer(text))
    if final_matches:
        # A tagged answer must be the sole final nonempty line. This prevents a
        # truncated continuation or a second label from being silently ignored.
        nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        is_last = (
            final_matches[-1].group(0).strip() == nonempty_lines[-1]
            if nonempty_lines
            else False
        )
        value = (
            _FINAL_VALUE.fullmatch(final_matches[0].group(1))
            if len(final_matches) == 1 and is_last
            else None
        )
        candidate = value.group(1).strip() if value else ""
        option = _FINAL_OPTION.fullmatch(candidate) if value else None
        answer = (option.group(1) or option.group(2)).upper() if option else None
        return ParsedAnswer(candidate, "final_answer_line", answer, answer is not None)
    candidate, method = _legacy_candidate(text)
    answer = normalize_option(candidate)
    return ParsedAnswer(candidate, method, answer, answer is not None)


def score_response(raw_response: str, ground_truth: str) -> dict[str, Any]:
    """Parse and score one response without removing invalid predictions."""
    truth = normalize_option(str(ground_truth))
    if truth is None:
        raise ValueError(f"invalid MME ground truth: {ground_truth!r}")
    parsed = parse_mme_answer(raw_response)
    return {
        **asdict(parsed),
        "ground_truth": truth,
        "correct": parsed.answer is not None and parsed.answer == truth,
        "error": None if parsed.valid else "invalid_option_format",
    }


def compute_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Report pooled and category accuracy, counting invalid/failed rows incorrect.

    Each input must contain ``question_id``, ``category``, ``ground_truth``, and
    the original ``raw_response``. Evaluation is recomputed from raw outputs;
    caller-provided correctness fields cannot change the denominator or score.
    ``status=ERROR`` marks an execution failure even if its diagnostic text parses.
    """
    totals = {
        key: {"correct": 0, "total": 0, "invalid": 0, "failed": 0}
        for key in CATEGORY_COUNTS
    }
    seen: set[str] = set()
    for row in records:
        qid = row["question_id"]
        if not isinstance(qid, str) or qid in seen:
            raise ValueError(f"invalid or duplicate metric question ID: {qid!r}")
        category = row["category"]
        if category not in totals:
            raise ValueError(f"unsupported MME category: {category!r}")
        seen.add(qid)
        parsed = score_response(row["raw_response"], row["ground_truth"])
        failed = row.get("status") in {"ERROR", "FAILED", "FAILURE"}
        bucket = totals[category]
        bucket["total"] += 1
        bucket["correct"] += int(parsed["correct"] and not failed)
        bucket["invalid"] += int(not parsed["valid"])
        bucket["failed"] += int(failed)
    total = sum(row["total"] for row in totals.values())
    if not total or (expected_rows is not None and total != expected_rows):
        raise ValueError(
            f"metric population mismatch: expected {expected_rows}, got {total}"
        )
    result = {
        name: sum(row[name] for row in totals.values())
        for name in ("correct", "total", "invalid", "failed")
    }
    result["accuracy"] = result["correct"] / total
    result["accuracy_percent"] = 100 * result["accuracy"]
    result["categories"] = {
        category: {
            **value,
            "accuracy": value["correct"] / value["total"] if value["total"] else None,
            "accuracy_percent": (
                100 * value["correct"] / value["total"] if value["total"] else None
            ),
        }
        for category, value in totals.items()
    }
    return result
