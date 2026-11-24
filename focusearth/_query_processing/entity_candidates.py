"""Structural nominal-phrase proposals for explicit target recovery."""

from __future__ import annotations

import re

from . import entity_core

nlp = entity_core.nlp


OPTION_TAIL = re.compile("\\s+(?:Options?\\s*:|\\([A-E]\\)\\s+|[A-E][.)]\\s+)", re.I)

CONTRACT_EXCLUSIONS = (
    set(entity_core.NONVISUAL)
    | set(entity_core.ASKED)
    | set(entity_core.REL)
    | {
        "answer",
        "answers",
        "choice",
        "choices",
        "question",
        "questions",
        "word",
        "words",
        "phrase",
        "phrases",
        "option",
        "options",
        "label",
        "labels",
        "direction",
        "directions",
        "distance",
        "relation",
        "relations",
        "purpose",
        "configuration",
        "category",
        "class",
        "size",
        "layout",
        "arrangement",
        "distribution",
        "location",
        "environment",
        "surrounding",
        "surroundings",
        "landscape",
        "pattern",
        "patterns",
        "region",
        "regions",
        "planning",
        "majority",
        "mode",
        "count",
        "weather",
        "traffic",
        "roadside",
        "body",
        "end",
        "respect",
        "northwest",
        "northeast",
        "southwest",
        "southeast",
        "north",
        "south",
        "east",
        "west",
        "one",
        "ones",
        "something",
        "anything",
        "nothing",
        "it",
        "them",
        "this",
        "that",
        "red",
        "green",
        "blue",
        "yellow",
        "black",
        "white",
        "gray",
        "grey",
        "brown",
        "orange",
        "purple",
        "pink",
        "cyan",
    }
)

LEADING_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those"}

INTERROGATIVE_PREFIX = {"what", "which", "how"}

NOMINAL_DEPS = set(entity_core.NOMINAL_DEPS) | {"ROOT"}

LOCAL_LEFT_DEPS = {
    "amod",
    "compound",
    "nummod",
    "poss",
    "appos",
    "npadvmod",
    "advmod",
    "det",
    "punct",
    "cc",
    "conj",
}


def norm(value) -> str:
    return entity_core.norm(value)


def _question_prefix(question: str) -> str:
    match = OPTION_TAIL.search(question)
    return question[: match.start()] if match else question


def _candidate_pos(token, doc) -> bool:
    if token.pos_ in {"NOUN", "PROPN"}:
        return True
    if token.pos_ not in {"ADJ", "VERB"} or token.dep_ not in NOMINAL_DEPS:
        return False
    children = [child for child in doc if child.head == token]
    return (
        any((child.dep_ == "det" for child in children))
        or any(
            (
                child.dep_ == "compound" and child.pos_ in {"NOUN", "PROPN"}
                for child in children
            )
        )
        or any((child.dep_ == "acl" for child in children))
    )


def admissible_head(token, doc) -> bool:
    lemma = norm(token.lemma_) or norm(token.text)
    if (
        not lemma
        or lemma in CONTRACT_EXCLUSIONS
        or token.dep_ not in NOMINAL_DEPS
        or (not _candidate_pos(token, doc))
    ):
        return False
    if token.pos_ == "PROPN":
        return False
    if len(norm(token.text)) == 1 and token.is_alpha:
        return False
    if (
        norm(token.head.lemma_) in {"shape", "shaped"}
        and token.head.dep_ == "amod"
        and (token.dep_ in {"npadvmod", "compound", "amod", "nsubj"})
    ):
        return False
    return True


def _chunk_for(doc, head):
    for chunk in doc.noun_chunks:
        if chunk.root.i == head.i:
            return chunk
    return None


def _recursive_left_tokens(doc, head):
    keep = {head}
    changed = True
    while changed:
        changed = False
        for token in doc:
            if token in keep or token.i > head.i:
                continue
            if token.head in keep and token.dep_ in LOCAL_LEFT_DEPS:
                keep.add(token)
                changed = True
    return sorted(keep, key=lambda token: token.i)


def _trim_span_tokens(tokens):
    tokens = list(tokens)
    while tokens and norm(tokens[0]) in LEADING_DETERMINERS:
        tokens.pop(0)
    if len(tokens) >= 2 and norm(tokens[0]) == "how" and (norm(tokens[1]) == "many"):
        tokens = tokens[2:]
    elif tokens and norm(tokens[0]) in INTERROGATIVE_PREFIX:
        tokens = tokens[1:]
    while tokens and (tokens[0].is_punct and tokens[0].text != "-"):
        tokens.pop(0)
    return tokens


def build_candidate(question: str, doc, head) -> dict | None:
    chunk = _chunk_for(doc, head)
    tokens = list(chunk) if chunk is not None else _recursive_left_tokens(doc, head)
    for child in doc:
        if (
            child.head == head
            and child.dep_ == "amod"
            and (norm(child.lemma_) in {"shape", "shaped"})
        ):
            tokens = sorted(set(tokens) | set(child.subtree), key=lambda token: token.i)
    tokens = _trim_span_tokens(tokens)
    if not tokens or head not in tokens:
        return None
    start, end = (tokens[0].idx, tokens[-1].idx + len(tokens[-1].text))
    literal = question[start:end]
    if not literal.strip() or any(
        (
            token.i > head.i and token.dep_ in {"prep", "acl", "relcl"}
            for token in tokens
        )
    ):
        return None
    modifiers = []
    for token in tokens:
        if token.i == head.i or token.is_punct or norm(token) in LEADING_DETERMINERS:
            continue
        modifiers.append(token.text)
    return {
        "mention_id": None,
        "literal_source_span": literal,
        "source_start": start,
        "source_end": end,
        "clean_entity_phrase": literal,
        "head": head.text,
        "head_start": head.idx,
        "head_end": head.idx + len(head.text),
        "local_modifiers": modifiers,
        "inherited_head": False,
        "antecedent_mention_id": None,
        "antecedent_provenance": None,
        "selector_like": bool(entity_core.SELECTOR_RE.search(literal)),
        "uncertainty": False,
        "uncertainty_reason": None,
        "origin": "open_nominal_structure",
    }


def _renumber(instances: list[dict]) -> None:
    for index, mention in enumerate(instances, 1):
        mention["mention_id"] = f"e{index}"
    mentions_by_head_span = {
        (
            mention.get("head_start"),
            mention.get("head_end"),
            mention.get("head"),
        ): mention
        for mention in instances
        if not mention.get("inherited_head")
    }
    for mention in instances:
        if not mention.get("inherited_head"):
            continue
        provenance = mention.get("antecedent_provenance") or {}
        antecedent = mentions_by_head_span.get(
            (
                provenance.get("head_start"),
                provenance.get("head_end"),
                provenance.get("literal_head"),
            )
        )
        if antecedent:
            mention["antecedent_mention_id"] = antecedent["mention_id"]
            provenance["mention_id"] = antecedent["mention_id"]


def extract(question: str) -> dict:
    prefix = _question_prefix(question)
    base = entity_core.extract(prefix)
    doc = nlp(prefix)
    instances = [dict(mention) for mention in base["entity_instances"]]
    existing_heads = {
        (mention.get("head_start"), mention.get("head_end"))
        for mention in instances
        if mention.get("head_start") is not None
    }
    existing_spans = {
        (mention["source_start"], mention["source_end"], mention["head_start"])
        for mention in instances
    }
    additions = []
    for token in doc:
        if (
            not admissible_head(token, doc)
            or (token.idx, token.idx + len(token.text)) in existing_heads
        ):
            continue
        candidate = build_candidate(prefix, doc, token)
        if candidate is None:
            continue
        key = (
            candidate["source_start"],
            candidate["source_end"],
            candidate["head_start"],
        )
        if key in existing_spans:
            continue
        if any(
            (
                candidate["source_start"] >= mention["source_start"]
                and candidate["source_end"] <= mention["source_end"]
                for mention in instances + additions
            )
        ):
            continue
        additions.append(candidate)
        existing_spans.add(key)
    instances.extend(additions)
    instances.sort(
        key=lambda mention: (
            mention["source_start"],
            mention["source_end"],
            mention["head_start"] if mention["head_start"] is not None else -1,
        )
    )
    _renumber(instances)
    base.update(
        {
            "state": (
                "OK"
                if instances
                else "SCENE_ONLY" if base["state"] == "SCENE_ONLY" else "NO_ENTITY"
            ),
            "entity_instances": instances,
            "no_entity": not bool(instances),
        }
    )
    return base
