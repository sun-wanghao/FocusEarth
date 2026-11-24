"""Nominal proposal boundaries, coordinated selectors, and structural filtering."""

from __future__ import annotations

import re

from . import entity_candidates

nlp = entity_candidates.nlp


ALWAYS_EXCLUDE = {
    "relation",
    "relations",
    "urban",
    "rural",
    "planning",
    "answer",
    "question",
    "option",
    "choice",
}

COORDINATED_SELECTOR = re.compile(
    "\\b(?:left|right|top|bottom)[ -]?most\\s+and\\s+(?:the\\s+)?(?:largest|smallest|longest|tallest|closest|nearest|farthest|furthest|first|second|third|larger|smaller)(?:\\s+[A-Za-z][A-Za-z'-]*){0,3}\\s*$",
    re.I,
)


def _token_at(doc, start: int | None):
    if start is None:
        return None
    return next((token for token in doc if token.idx == start), None)


def _trim_post_head(question: str, mention: dict) -> None:
    if mention.get("head_end") is None or mention["source_end"] <= mention["head_end"]:
        return
    if mention["literal_source_span"].lower().startswith("logo of "):
        return
    mention["source_end"] = mention["head_end"]
    mention["literal_source_span"] = question[
        mention["source_start"] : mention["source_end"]
    ]
    mention["clean_entity_phrase"] = mention["literal_source_span"]
    mention["local_modifiers"] = entity_candidates.entity_core._surface_modifiers(
        mention["clean_entity_phrase"], mention["head"]
    )


def _expand_coordinated_selector(question: str, mention: dict) -> None:
    if mention.get("head_end") is None:
        return
    prefix = question[: mention["head_end"]]
    matches = list(COORDINATED_SELECTOR.finditer(prefix))
    if not matches:
        return
    match = matches[-1]
    mention["source_start"] = match.start()
    mention["source_end"] = mention["head_end"]
    mention["literal_source_span"] = question[
        mention["source_start"] : mention["source_end"]
    ]
    mention["clean_entity_phrase"] = mention["literal_source_span"]
    mention["local_modifiers"] = entity_candidates.entity_core._surface_modifiers(
        mention["clean_entity_phrase"], mention["head"]
    )
    mention["selector_like"] = True


def extract(question: str) -> dict:
    output = entity_candidates.extract(question)
    prefix = entity_candidates._question_prefix(question)
    doc = nlp(prefix)
    mentions = []
    for mention in output["entity_instances"]:
        surface = entity_candidates.norm(mention["head"])
        token = _token_at(doc, mention.get("head_start"))
        if surface in ALWAYS_EXCLUDE:
            continue
        if mention.get("origin") == "open_nominal_structure" and token is not None:
            if entity_candidates.entity_core.SELECTOR_RE.fullmatch(surface):
                continue
            if surface in {"front", "rear", "section"}:
                has_of = any(
                    (
                        child.head == token
                        and child.dep_ == "prep"
                        and (entity_candidates.norm(child) == "of")
                        for child in doc
                    )
                )
                if not has_of or token.dep_ == "pobj":
                    continue
        cleaned = dict(mention)
        _trim_post_head(prefix, cleaned)
        _expand_coordinated_selector(prefix, cleaned)
        mentions.append(cleaned)
    pruned = []
    head_starts = {mention.get("head_start") for mention in mentions}
    for mention in mentions:
        token = _token_at(doc, mention.get("head_start"))
        if (
            mention.get("origin") == "open_nominal_structure"
            and token is not None
            and (token.dep_ == "compound")
        ):
            ancestor = token.head
            while ancestor.dep_ == "compound" and ancestor.head != ancestor:
                ancestor = ancestor.head
            if ancestor.idx in head_starts:
                continue
        pruned.append(mention)
    pruned.sort(
        key=lambda mention: (
            mention["source_start"],
            mention["source_end"],
            mention["head_start"] if mention["head_start"] is not None else -1,
        )
    )
    entity_candidates._renumber(pruned)
    output.update(
        {
            "state": (
                "OK"
                if pruned
                else "SCENE_ONLY" if output["state"] == "SCENE_ONLY" else "NO_ENTITY"
            ),
            "entity_instances": pruned,
            "no_entity": not bool(pruned),
        }
    )
    return output
