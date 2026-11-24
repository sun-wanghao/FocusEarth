"""Explicit target recovery with source-ordered nominal descriptions."""

from __future__ import annotations

import copy
import re


from . import entity_core
from . import entity_candidate_refinement

QUESTION_WORDS = {"what", "which", "where", "who", "how"}

QUESTION_FRAME = {
    "what",
    "which",
    "where",
    "who",
    "how",
    "color",
    "colors",
    "shape",
    "state",
    "status",
    "activity",
    "condition",
    "function",
    "use",
    "form",
    "amount",
    "number",
    "many",
    "type",
    "kind",
    "way",
    "position",
}

REL_PREPS = {
    "near",
    "beside",
    "next",
    "around",
    "above",
    "below",
    "under",
    "over",
    "at",
    "on",
    "in",
    "inside",
    "within",
    "to",
    "from",
    "behind",
    "before",
    "after",
    "along",
    "across",
    "between",
    "of",
    "with",
}

GENERIC_HEADS = {"item", "thing", "object", "objects", "something", "anything"}

YES_NO_AUX = {
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "has",
    "have",
    "had",
}


def _token_by_char(doc, start):
    return next((t for t in doc if t.idx == start), None)


def _question_target_proposal(question: str) -> tuple[dict | None, str]:
    proposals = entity_candidate_refinement.extract(question).get(
        "entity_instances", []
    )
    if not proposals:
        return (None, "no_structural_proposal")
    doc = entity_core.nlp(question)
    first = next((t for t in doc if entity_core.norm(t) in QUESTION_WORDS), None)
    if first is None:
        first = next((t for t in doc if entity_core.norm(t) in YES_NO_AUX), None)
    if first is None:
        return (None, "no_question_form")
    candidates = []
    for proposal in proposals:
        token = _token_by_char(doc, proposal.get("head_start"))
        if token is None or token.i <= first.i:
            continue
        words = [
            entity_core.norm(x) for x in proposal.get("clean_entity_phrase", "").split()
        ]
        if words and words[0] in QUESTION_FRAME:
            continue
        candidates.append((proposal, token))
    if not candidates:
        return (None, "no_explicit_nominal_after_question_frame")
    candidates.sort(
        key=lambda pair: (pair[0]["source_start"], pair[0]["source_end"], pair[1].i)
    )
    proposal, token = candidates[0]
    prep = token.head if token.dep_ == "pobj" else None
    parent = (
        prep.head if prep is not None and entity_core.norm(prep) in REL_PREPS else None
    )
    asked_complement = (
        prep is not None
        and entity_core.norm(prep) == "of"
        and (parent is not None)
        and (entity_core.norm(parent) in entity_core.ASKED)
    )
    has_restrictive_child = any(
        (child.head == token and child.dep_ in {"acl", "relcl"} for child in doc)
    )
    if has_restrictive_child or (
        token.dep_ == "pobj"
        and (not asked_complement)
        and (parent is not None)
        and (parent.pos_ in {"VERB", "AUX"})
    ):
        return (None, "ambiguous_or_relation_only")
    if entity_core.norm(token) in GENERIC_HEADS:
        span = proposal["literal_source_span"]
        match = list(
            re.finditer(
                "\\b([A-Za-z][A-Za-z-]*)\\s+shaped\\s+" + re.escape(token.text) + "\\b",
                span,
                re.I,
            )
        )
        if match:
            end = (
                proposal["source_start"] + match[-1].start(0) + len(match[-1].group(1))
            )
            start = proposal["source_start"]
            replacement = copy.deepcopy(proposal)
            replacement["source_end"] = end
            replacement["literal_source_span"] = question[start:end]
            replacement["clean_entity_phrase"] = replacement["literal_source_span"]
            replacement["head"] = match[-1].group(1)
            hs = question.find(match[-1].group(1), start, end)
            replacement["head_start"] = hs
            replacement["head_end"] = hs + len(match[-1].group(1))
            replacement["local_modifiers"] = [
                x
                for x in replacement.get("local_modifiers", [])
                if entity_core.norm(x) != "shaped"
            ]
            proposal = replacement
    return (proposal, "earliest_explicit_nominal_after_question_frame")


def _repair_target(question, output):
    proposal, reason = _question_target_proposal(question)
    if proposal is None:
        return ([], reason)
    existing = output["entity_instances"]
    hs, he = (proposal.get("head_start"), proposal.get("head_end"))
    if any(
        (
            m.get("head_start") == hs
            and m.get("head_end") == he
            or m.get("source_start", 10**9) <= hs < m.get("source_end", -1)
            for m in existing
        )
    ):
        return ([], "target_already_covered")
    if any(
        (
            proposal["source_start"] >= m["source_start"]
            and proposal["source_end"] <= m["source_end"]
            for m in existing
        )
    ):
        return ([], "target_span_already_covered")
    add = copy.deepcopy(proposal)
    add["origin"] = "target_aware_entity_extraction_recovery"
    add["mention_id"] = None
    existing.append(add)
    existing.sort(
        key=lambda m: (
            m["source_start"],
            m["source_end"],
            m.get("head_start") if m.get("head_start") is not None else -1,
        )
    )
    for i, mention in enumerate(existing, 1):
        mention["mention_id"] = f"e{i}"
    output["state"] = "OK"
    output["no_entity"] = False
    return ([add["clean_entity_phrase"]], reason)


def extract(question: str) -> dict:
    """Extract nominal entities and recover an explicit question target."""
    output = copy.deepcopy(entity_core.extract(question))
    _repair_target(question, output)
    return output
