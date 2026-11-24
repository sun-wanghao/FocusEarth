"""Context description normalization, compaction, and semantic guards."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

NEGATIVE_RE = re.compile(
    r"\b(?:not|no|without|cannot|can't|doesn't|does not|not seen|not visible|absent|missing)\b",
    re.I,
)
UNNAMED_START_RE = re.compile(
    r"^(?:next to|near|beside|behind|between|above|below|under|around|inside|outside|on|at|in|not)\b",
    re.I,
)

CONTENT_TOKEN_RE = re.compile("[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

COMPACTION_REMOVABLE_TOKENS = {
    "the",
    "this",
    "picture",
    "image",
    "frame",
    "map",
    "pictue",
    "that",
    "which",
    "is",
    "are",
    "located",
    "can",
    "be",
    "seen",
    "there",
    "of",
    "in",
    "on",
    "at",
}

COMPACTION_INSERTABLE_TOKENS = {"in", "on", "at"}


def span(
    document: str, start: int, end: int, document_name: str = "ORIGINAL_QUESTION"
) -> dict[str, Any]:
    return {
        "document": document_name,
        "start": start,
        "end": end,
        "text": document[start:end],
    }


def find_all_spans(
    document: str, phrase: str, document_name: str = "ORIGINAL_QUESTION"
) -> list[dict[str, Any]]:
    if not phrase:
        return []
    return [
        span(document, match.start(), match.end(), document_name)
        for match in re.finditer(re.escape(phrase), document, re.I)
    ]


def normalize_deletion_operation(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": "SOURCE_DESCRIPTION",
        "operation": operation["operation"],
        "rule_id": operation["rule_id"],
        "reason": operation["reason"],
        "source_spans": [
            {
                "document": "ORIGINAL_QUESTION",
                "start": operation["source_start"],
                "end": operation["source_end"],
                "text": operation["text"],
            }
        ],
        "output_text": "",
        "semantic_support": "Source-preserving answer-variable or grammatical deletion rule.",
        "function_words_only": False,
        "meaning_change_risk": "NONE",
    }


def reformulate_description(
    context_rules: Any, category: str, question: str
) -> dict[str, Any]:
    """Construct a source-preserving description using general question rules."""
    base = context_rules.remove_question_operators(category, question)
    return {
        "source_description": base["source_description"],
        "question_form": base["question_form"],
        "matched_rule": base["matched_rule"],
        "operations": [normalize_deletion_operation(op) for op in base["operations"]],
        "ambiguous": base["ambiguous"],
        "failed": base["failed"],
        "failure_reason": base["failure_reason"],
        "equivalence_note": "Question-operator removal preserves the source description without additional edits.",
    }


def record_compaction_deletion(
    source_description: str, matched_text: str, rule: str, reason: str
) -> dict[str, Any]:
    matches = find_all_spans(source_description, matched_text, "CONTEXT_DESCRIPTION")
    return {
        "stage": "DESCRIPTION_COMPACTION",
        "operation": "DELETE",
        "rule_id": rule,
        "reason": reason,
        "source_spans": matches[:1],
        "output_text": "",
        "semantic_support": "Retrieval-only grammatical or global-canvas boilerplate reduction; no referential content is removed.",
        "function_words_only": all(
            (
                t.lower() in COMPACTION_REMOVABLE_TOKENS
                for t in CONTENT_TOKEN_RE.findall(matched_text)
            )
        ),
        "meaning_change_risk": "NONE",
    }


def replace_with_source_spans(
    text: str,
    source_description: str,
    pattern: re.Pattern[str],
    replacement: str,
    rule: str,
    reason: str,
    operations: list[dict[str, Any]],
) -> str:
    while True:
        match = pattern.search(text)
        if not match:
            return text
        matched = match.group(0)
        operations.append(
            record_compaction_deletion(
                source_description, matched.strip(), rule, reason
            )
        )
        text = text[: match.start()] + replacement + text[match.end() :]


def compact_description(source_description: str) -> dict[str, Any]:
    """Compact a target-centered description without adding referential content."""
    if not source_description:
        return {
            "compact_description": "",
            "operations": [],
            "style": "NO_OUTPUT",
            "equivalence_note": "An empty source description yields no compact description.",
        }
    operations: list[dict[str, Any]] = []
    text = source_description
    preposed = re.match(
        "^(?P<prep>In|On|At)\\s+(?P<scope>[^,]+),\\s*(?P<body>.+)$", text, re.I
    )
    if preposed and (
        not re.match("^(?:from|counting)\\b", preposed.group("body"), re.I)
    ):
        prep, scope, body = (
            preposed.group("prep").lower(),
            preposed.group("scope"),
            preposed.group("body"),
        )
        if re.fullmatch("(?:the|this) (?:picture|image|map|frame)", scope, re.I):
            reordered_text = body
        else:
            reordered_text = f"{body}, {prep} {scope}"
        operations.append(
            {
                "stage": "DESCRIPTION_COMPACTION",
                "operation": "MOVE",
                "rule_id": "COMPACT_TARGET_CENTER_PREPOSED_SCOPE",
                "reason": "TARGET_CENTERED_REORDER",
                "source_spans": [
                    span(
                        source_description,
                        preposed.start("body"),
                        preposed.end("body"),
                        "CONTEXT_DESCRIPTION",
                    ),
                    span(
                        source_description,
                        preposed.start("prep"),
                        preposed.end("scope"),
                        "CONTEXT_DESCRIPTION",
                    ),
                ],
                "output_text": reordered_text,
                "semantic_support": "Comma-delimited preposed locative scope is moved after the unchanged target description; the same preposition and scope are retained unless the scope is only the global picture canvas.",
                "function_words_only": False,
                "meaning_change_risk": "LOW",
            }
        )
        text = reordered_text
    reductions = [
        (
            re.compile(
                "\\b(?:that|which)\\s+(?:is|are)\\s+(?=(?!not\\b|no\\b|without\\b)[A-Za-z])",
                re.I,
            ),
            "",
            "COMPACT_REDUCE_POSITIVE_RELATIVE_CLAUSE",
            "OPTIONAL_GRAMMATICAL_SCAFFOLD",
        ),
        (
            re.compile(
                "\\blocated\\s+(?=(?:in|on|at|near|beside|next|under|above|below|between|around|along|inside|within|to)\\b)",
                re.I,
            ),
            "",
            "COMPACT_DROP_REDUNDANT_LOCATED",
            "OPTIONAL_GRAMMATICAL_SCAFFOLD",
        ),
        (
            re.compile("\\bcan be seen\\b", re.I),
            "",
            "COMPACT_DROP_POSITIVE_VISIBILITY_SCAFFOLD",
            "OPTIONAL_GRAMMATICAL_SCAFFOLD",
        ),
        (
            re.compile("\\bthere\\s+(?=(?:in|on|at)\\b)", re.I),
            "",
            "COMPACT_DROP_RESIDUAL_EXPLETIVE",
            "OPTIONAL_GRAMMATICAL_SCAFFOLD",
        ),
    ]
    for pattern, replacement, rule, reason in reductions:
        text = replace_with_source_spans(
            text, source_description, pattern, replacement, rule, reason, operations
        )
    media_patterns = [
        re.compile(
            "\\s+(?:in|on)\\s+(?:the|this)\\s+(?:picture|image|map|frame|pictue)\\b(?!\\s+(?:center|centre|edge|side|corner|top|bottom|middle)\\b)",
            re.I,
        ),
        re.compile(
            "\\s+of\\s+(?:the|this)\\s+(?:picture|image|map|frame|pictue)\\b(?!\\s+(?:center|centre|edge|side|corner|top|bottom|middle)\\b)",
            re.I,
        ),
    ]
    for pattern in media_patterns:
        text = replace_with_source_spans(
            text,
            source_description,
            pattern,
            "",
            "COMPACT_DROP_GLOBAL_CANVAS_BOILERPLATE",
            "GLOBAL_CANVAS_BOILERPLATE",
            operations,
        )
    text = re.sub("\\s{2,}", " ", text)
    text = re.sub("\\s+,", ",", text)
    text = re.sub(",\\s*,+", ",", text)
    text = text.strip(" ,")
    return {
        "compact_description": text,
        "operations": operations,
        "style": (
            "TARGET_CENTERED_COMPACT_DESCRIPTION"
            if text != source_description
            else "IDENTITY_SEMANTIC_GUARD"
        ),
        "equivalence_note": "B starts only from A; it optionally target-centres an explicit comma-delimited locative scope and removes positive grammatical/global-canvas boilerplate while retaining the visual content-token multiset.",
    }


def tokens(text: str) -> list[str]:
    return [token.lower() for token in CONTENT_TOKEN_RE.findall(text)]


def semantic_token_guard(
    source_description: str, compact_description: str
) -> tuple[bool, dict[str, list[str]]]:
    source_counts, compact_counts = (
        Counter(tokens(source_description)),
        Counter(tokens(compact_description)),
    )
    removed = list((source_counts - compact_counts).elements())
    inserted = list((compact_counts - source_counts).elements())
    ok = all((token in COMPACTION_REMOVABLE_TOKENS for token in removed)) and all(
        (token in COMPACTION_INSERTABLE_TOKENS for token in inserted)
    )
    return (ok, {"removed": sorted(removed), "inserted": sorted(inserted)})


def source_token_guard(
    question: str, source_description: str
) -> tuple[bool, list[str]]:
    """Require every description token to occur in the source question."""
    inserted = list(
        (Counter(tokens(source_description)) - Counter(tokens(question))).elements()
    )
    return (not inserted, sorted(inserted))


def question_status(
    source_result: dict[str, Any],
    negative: bool,
    unnamed: bool,
) -> tuple[str, bool, str]:
    if source_result["failed"]:
        return ("FAILED_UNHANDLED", True, source_result["failure_reason"])
    if negative:
        return (
            "NEGATIVE_OR_ABSENCE_QUERY",
            False,
            "Explicit negation/absence is preserved; neither output is treated as a positive visual assertion.",
        )
    if unnamed:
        return (
            "UNNAMED_TARGET",
            True,
            "No target noun is explicit; the relation/reference fragment is preserved without inserting object/something.",
        )
    if source_result["ambiguous"]:
        return ("AMBIGUOUS", True, source_result["failure_reason"])
    return ("CLEAN", False, "")
