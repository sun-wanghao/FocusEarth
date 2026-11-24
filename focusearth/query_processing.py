"""Deterministic, answer-blind MME query construction.

The MME role generator uses a spaCy entity extractor and a
source-offset-preserving Context transformation. Context is emitted before
Entity descriptions, which retain their occurrence order and duplicates.
"""

from __future__ import annotations

from importlib import import_module
import re
from types import ModuleType
from typing import Any


def _load_component(name: str) -> ModuleType:
    """Import an NLP component lazily; spaCy loads only for entity extraction."""
    return import_module(f"._query_processing.{name}", __package__)


def validate_roles(roles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check Context-first scorer role ordering and nonempty text."""
    if not isinstance(roles, list) or not roles or roles[0].get("kind") != "Context":
        raise ValueError("first role must be Context")
    for index, role in enumerate(roles):
        expected = "Context" if index == 0 else "Entity"
        if role.get("kind") != expected:
            raise ValueError(
                "Entity roles must follow Context in source occurrence order"
            )
        if not isinstance(role.get("text"), str) or not role["text"].strip():
            raise ValueError("each role needs nonempty text")
        if index > 0 and role.get("ordinal") != index:
            raise ValueError("Entity ordinals must be consecutive and one-based")
    return roles


def extract_entities(question: str) -> list[str]:
    """Return target-aware entity phrases for one MME question.

    Requires spaCy and the ``en_core_web_sm`` model. This function performs
    no scoring, image access, or answer-dependent processing.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty text")
    target = _load_component("entity_target")
    result = target.extract(question)
    values = [
        str(item.get("clean_entity_phrase", "")).strip()
        for item in result.get("entity_instances", [])
    ]
    # This is the MME scorer's role filter: empty fields and the placeholder
    # '-' are excluded, while duplicates and occurrence order are retained.
    return [value for value in values if value and value != "-"]


_DANGLING = re.compile(
    r"\b(?:of|in|on|at|with|near|behind|between|above|below|under|to|from)\s*$", re.I
)
_COLOR_START = re.compile(r"^(?:What colors are|What color are|What color is)\b", re.I)


def reformulate_context(
    question_id: str, category: str, question: str
) -> dict[str, Any]:
    """Construct the MME Context description and its routing trace."""
    if not all(
        isinstance(value, str) and value.strip()
        for value in (question_id, category, question)
    ):
        raise ValueError("question_id, category, and question are required")
    context_rules = _load_component("context_rules")
    context_refinement = _load_component("context_refinement")

    # The category is canonical MME metadata. The two transformations use
    # source-offset edits and guards; they do not inspect options, labels, or
    # image content.
    operator_result = context_rules.remove_question_operators(
        category.lower(), question
    )
    source_result = context_refinement.reformulate_description(
        context_rules, category.lower(), question
    )
    compaction_result = context_refinement.compact_description(
        source_result["source_description"]
    )
    source_description, compact_description = (
        source_result["source_description"],
        compaction_result["compact_description"],
    )
    negative = bool(context_refinement.NEGATIVE_RE.search(question))
    unnamed = bool(
        source_description
        and _COLOR_START.match(question)
        and context_refinement.UNNAMED_START_RE.search(source_description)
    )
    status, uncertainty, reason = context_refinement.question_status(
        source_result, negative, unnamed
    )
    if (
        source_description
        and _DANGLING.search(source_description)
        and status == "CLEAN"
    ):
        status, uncertainty = "AMBIGUOUS", True
        reason = (
            "The source-derived description ends in a relation or predicate "
            "without an explicit complement."
        )
    source_valid, inserted = (
        context_refinement.source_token_guard(question, source_description)
        if source_description
        else (True, [])
    )
    compaction_valid, delta = (
        context_refinement.semantic_token_guard(source_description, compact_description)
        if source_description
        else (True, {"removed": [], "inserted": []})
    )
    if not source_valid or not compaction_valid:
        raise RuntimeError(f"semantic token guard failed for {question_id}")
    if compact_description:
        route, context = "REFORMULATED_QUERY", compact_description
    elif status == "FAILED_UNHANDLED" and not source_description:
        route, context = "ORIGINAL_QUESTION_FALLBACK", question
    else:
        raise RuntimeError(f"empty nonfailed Context for {question_id}: {status}")
    return {
        "context": context,
        "route": route,
        "status": status,
        "source_description": source_description,
        "compact_description": compact_description,
        "uncertainty": uncertainty,
        "uncertainty_reason": reason,
        "source_token_guard": source_valid,
        "semantic_token_guard": compaction_valid,
        "source_insertions": inserted,
        "compaction_token_delta": delta,
        "question_operator_description": operator_result.get("source_description", ""),
    }


def construct_mme_roles(
    question_id: str, category: str, question: str
) -> dict[str, Any]:
    """Return Context-first MME scorer roles and query-transformation details."""
    context = reformulate_context(question_id, category, question)
    entities = extract_entities(question)
    roles: list[dict[str, Any]] = [
        {"kind": "Context", "ordinal": None, "text": context["context"]}
    ]
    roles.extend(
        {"kind": "Entity", "ordinal": i, "text": value}
        for i, value in enumerate(entities, 1)
    )
    validate_roles(roles)
    return {
        "question_id": question_id,
        "question": question,
        "roles": roles,
        "context_trace": context,
    }
