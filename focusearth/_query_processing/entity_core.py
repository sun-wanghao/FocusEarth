"""Nominal-head extraction with identity modifiers and local anaphor resolution."""

import re

import spacy

NLP_MODEL = "en_core_web_sm"

nlp = spacy.load(NLP_MODEL)

VISUAL_HEADS = {
    "bridge",
    "ship",
    "ships",
    "car",
    "cars",
    "plane",
    "planes",
    "aircraft",
    "airplane",
    "airbus",
    "boeing",
    "boat",
    "boats",
    "building",
    "buildings",
    "structure",
    "structures",
    "object",
    "objects",
    "feature",
    "features",
    "court",
    "courts",
    "apron",
    "terminal",
    "terminals",
    "tower",
    "towers",
    "substation",
    "intersection",
    "intersections",
    "road",
    "roads",
    "runway",
    "runways",
    "parking",
    "lot",
    "lots",
    "field",
    "fields",
    "genset",
    "tank",
    "tanks",
    "truck",
    "trucks",
    "vehicle",
    "vehicles",
    "van",
    "river",
    "rivers",
    "house",
    "houses",
    "roof",
    "roofs",
    "rooftop",
    "rooftops",
    "complex",
    "complexes",
    "flag",
    "flags",
    "roundabout",
    "roundabouts",
    "slide",
    "slides",
    "facility",
    "facilities",
    "statue",
    "deck",
    "tail",
    "track",
    "logo",
    "fountain",
    "playground",
    "entrance",
    "billboard",
    "billboards",
    "shore",
    "island",
    "water",
    "surface",
    "yard",
    "rectangle",
    "rectangles",
    "lattice",
    "crossing",
    "soil",
    "ground",
    "wall",
    "walls",
    "space",
    "dock",
    "docks",
    "lawn",
    "lawns",
    "park",
    "parks",
    "construction",
    "site",
    "sites",
    "front",
    "rear",
    "section",
    "sections",
    "roof",
    "roofs",
    "entrance",
    "entrances",
    "part",
    "parts",
    "gen-set",
    "gensets",
}

NONVISUAL = {
    "what",
    "which",
    "amount",
    "number",
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
    "transportation",
    "image",
    "picture",
    "scene",
    "setting",
    "area",
    "areas",
    "part",
    "parts",
    "side",
    "sides",
    "corner",
    "type",
    "kind",
    "way",
    "thing",
    "things",
    "time",
    "row",
    "position",
    "edge",
    "top",
    "bottom",
    "left",
    "right",
    "middle",
    "center",
    "most",
}

ASKED = {
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
    "transportation",
    "amount",
    "number",
    "many",
}

REL = {
    "near",
    "next",
    "adjacent",
    "alongside",
    "beside",
    "besides",
    "around",
    "surrounding",
    "between",
    "above",
    "below",
    "under",
    "over",
    "north",
    "south",
    "east",
    "west",
    "left",
    "right",
    "top",
    "bottom",
    "parallel",
    "located",
    "passing",
    "through",
    "along",
    "inside",
    "within",
    "at",
    "on",
    "to",
}

SCAFFOLD = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "one",
    "two",
    "three",
    "four",
    "five",
    "only",
    "most",
    "many",
    "different",
    "visible",
    "predominant",
    "dominant",
    "similar",
}

HARD_FRAME_HEADS = {
    "what",
    "which",
    "amount",
    "number",
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
    "image",
    "picture",
    "scene",
    "setting",
    "type",
    "kind",
    "way",
    "time",
    "row",
    "position",
    "edge",
    "top",
    "bottom",
    "left",
    "right",
    "middle",
    "center",
    "most",
}

CLASSIFIER_FRAME_HEADS = {"type", "kind", "form", "class", "category"}

MATERIAL_PREDICATES = {
    "find",
    "locate",
    "place",
    "park",
    "use",
    "travel",
    "cross",
    "occupy",
    "contain",
    "hold",
    "show",
    "see",
    "stand",
    "sit",
    "appear",
    "exist",
    "remain",
}

SPATIAL_PREPOSITIONS = {
    "in",
    "on",
    "at",
    "inside",
    "within",
    "near",
    "beside",
    "next",
    "around",
    "under",
    "over",
    "above",
    "below",
    "across",
    "through",
    "along",
    "between",
}

NOMINAL_DEPS = {"dobj", "obj", "pobj", "nsubj", "nsubjpass", "conj", "attr", "appos"}

ANAPHOR_DETERMINERS = {"the", "another", "other"}

SELECTOR_RE = re.compile(
    "(?:left|right|top|bottom)[ -]?most|largest|smallest|longest|tallest|closest|nearest|farthest|furthest|first|second|third|larger|smaller",
    re.I,
)


def norm(value):
    text = value.text if hasattr(value, "text") else str(value)
    return re.sub("^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", text).lower()


def scene_context(question):
    return bool(re.search("\\b(?:urban|rural)\\b", question, re.I))


def scene_only(question):
    return bool(
        re.search(
            "\\bscene\\b.*\\b(?:urban|rural)\\b|\\b(?:urban|rural)\\b.*\\bscene\\b",
            question,
            re.I,
        )
    ) and (
        not bool(
            re.search(
                "\\b(?:bridge|building|boat|car|plane|airplane|tank|genset|intersection|runway|ship|road|house|truck|terminal|court|dock|lawn|site)\\b",
                question,
                re.I,
            )
        )
    )


def base_visual(token):
    return norm(token.lemma_) in VISUAL_HEADS or norm(token.text) in VISUAL_HEADS


def _has_spatial_complement(predicate, doc):
    for token in doc:
        if (
            token.head == predicate
            and token.dep_ in {"prep", "agent"}
            and (norm(token) in SPATIAL_PREPOSITIONS)
        ):
            return True
    return False


def _material_predicate_for(token, doc):
    """Return a local material/spatial predicate licensing entity use."""
    anchors = [token]
    if (
        token.dep_ == "pobj"
        and norm(token.head) == "of"
        and (norm(token.head.head.lemma_) in CLASSIFIER_FRAME_HEADS)
    ):
        anchors.append(token.head.head)
    for anchor in anchors:
        for child in doc:
            if (
                child.head == anchor
                and child.dep_ in {"acl", "relcl"}
                and (norm(child.lemma_) in MATERIAL_PREDICATES)
                and _has_spatial_complement(child, doc)
            ):
                return child
        predicate = anchor.head
        if anchor.dep_ in {
            "nsubj",
            "nsubjpass",
            "dobj",
            "obj",
            "attr",
        } and predicate.pos_ in {"VERB", "AUX"}:
            if norm(
                predicate.lemma_
            ) in MATERIAL_PREDICATES and _has_spatial_complement(predicate, doc):
                return predicate
    return None


def contextual_entity_use(token, doc):
    """Promote an otherwise filtered noun only under strong local evidence.

    Two structural licenses are intentionally narrow: a visual compound whose
    lexical head is the filtered token (``parking area``), or a locally spatial
    material predicate applying to the nominal/classifier phrase.  Hard frame
    nouns remain excluded even if a parser attaches them to such predicates.
    """
    word = norm(token.lemma_)
    if token.pos_ != "NOUN" or token.dep_ not in NOMINAL_DEPS:
        return False
    if word in HARD_FRAME_HEADS or word in REL:
        return False
    compound_children = [
        child for child in doc if child.head == token and child.dep_ == "compound"
    ]
    if any((base_visual(child) for child in compound_children)):
        return True
    return _material_predicate_for(token, doc) is not None


def is_candidate_head(token, doc):
    return base_visual(token) or contextual_entity_use(token, doc)


def source_span(question, tokens):
    ordered = sorted(tokens, key=lambda token: token.i)
    start = ordered[0].idx
    end = ordered[-1].idx + len(ordered[-1].text)
    return (start, end, question[start:end])


def descriptor_kind(token):
    if SELECTOR_RE.fullmatch(norm(token)):
        return "SELECTOR_LIKE_DESCRIPTOR"
    return "LOCAL_IDENTITY_MODIFIER"


def local_tokens(doc, head):
    keep = {head}
    changed = True
    while changed:
        changed = False
        for token in doc:
            if token in keep or token.i > head.i:
                continue
            if token.head in keep and (
                token.dep_ in {"amod", "compound", "nummod", "poss", "appos"}
                or token.pos_ in {"ADJ", "PROPN"}
            ):
                if (
                    (
                        norm(token) not in REL
                        or (norm(token) == "parallel" and token.dep_ == "amod")
                    )
                    and norm(token) not in ASKED
                    and (norm(token) not in SCAFFOLD)
                ):
                    keep.add(token)
                    changed = True
    for modifier in list(keep):
        if (
            modifier.i >= head.i
            or modifier.dep_ != "amod"
            or norm(modifier.lemma_) not in {"shape", "shaped"}
        ):
            continue
        for token in doc:
            if (
                token.head == modifier
                and token.i < modifier.i
                and (modifier.i - token.i <= 2)
                and (token.dep_ in {"npadvmod", "compound", "amod"})
            ):
                if (
                    token.is_alpha
                    and norm(token) not in ASKED
                    and (norm(token) not in REL)
                    and (norm(token) not in SCAFFOLD)
                ):
                    keep.add(token)
    for token in doc:
        if (
            token.i < head.i
            and token.i == head.i - 1
            and base_visual(token)
            and (base_visual(head) or contextual_entity_use(head, doc))
        ):
            keep.add(token)
    if (
        head.i >= 2
        and doc[head.i - 1].text == "-"
        and (doc[head.i - 2].is_alpha or doc[head.i - 2].like_num)
    ):
        keep.add(doc[head.i - 1])
        keep.add(doc[head.i - 2])
    if (
        head.i >= 3
        and doc[head.i - 2].text == "-"
        and (doc[head.i - 3].is_alpha or doc[head.i - 3].like_num)
    ):
        keep.add(doc[head.i - 2])
        keep.add(doc[head.i - 3])
    return sorted(keep, key=lambda token: token.i)


def _surface_modifiers(text, head):
    if text.lower().startswith("logo of "):
        return [text[8:]]
    head_at = text.lower().rfind(head.lower())
    prefix = text[:head_at] if head_at >= 0 else ""
    tokens = re.findall("[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", prefix)
    stop = {"the", "a", "an", "and", "of", "to", "with", "in"}
    return [token for token in tokens if token.lower() not in stop]


def make_candidate(question, doc, head, origin):
    word = norm(head.lemma_)
    contextual = contextual_entity_use(head, doc)
    if not base_visual(head) and (not contextual):
        return None
    if (word in NONVISUAL or word in ASKED or word in REL) and (not contextual):
        return None
    if word in HARD_FRAME_HEADS or word in REL:
        return None
    if head.pos_ == "VERB" and head.dep_ not in NOMINAL_DEPS:
        return None
    if (
        head.i + 2 < len(doc)
        and doc[head.i + 1].text == "-"
        and (norm(doc[head.i + 2]) in {"like", "shaped", "roofed"})
    ):
        return None
    if (
        head.head.i > head.i
        and norm(head.head.lemma_) in {"shape", "shaped"}
        and (head.head.dep_ == "amod")
        and (head.dep_ in {"npadvmod", "compound", "amod"})
    ):
        return None
    if word in {"front", "rear", "section"}:
        has_of = any(
            (
                child.head == head and child.dep_ == "prep" and (norm(child) == "of")
                for child in doc
            )
        )
        if not has_of or head.dep_ == "pobj":
            return None
    tokens = local_tokens(doc, head)
    start, end, text = source_span(question, tokens)
    for token in doc:
        if token.head == head and token.dep_ == "prep" and (norm(token) == "of"):
            children = [
                child for child in doc if child.head == token and child.pos_ == "PROPN"
            ]
            if children and (not any((base_visual(child) for child in children))):
                start = min(start, token.idx)
                end = max(end, children[-1].idx + len(children[-1].text))
                if end < len(question) and question[end : end + 2] in {"'s", "’s"}:
                    end += 2
                text = question[start:end]
    prefix = question[: head.idx]
    coordinated = list(
        re.finditer(
            "(?:left|right|top|bottom)[ -]?most\\s+and\\s+(?:the\\s+)?(?:largest|smallest|longest|tallest|closest|nearest|farthest|first|second|third|larger|smaller)\\s*$",
            prefix,
            re.I,
        )
    )
    if coordinated:
        start = coordinated[-1].start()
        text = question[start:end]
    selector = list(
        re.finditer(
            "(?:left|right|top|bottom)\\s*-\\s*most(?:\\s+[A-Za-z-]+){0,3}\\s*$",
            prefix,
            re.I,
        )
    )
    if selector and (not coordinated):
        start = selector[-1].start()
        text = question[start:end]
    descriptors = []
    for token in tokens:
        if token.i == head.i or token.is_punct:
            continue
        if token.dep_ in {
            "amod",
            "compound",
            "nummod",
            "poss",
            "appos",
            "npadvmod",
        } or token.pos_ in {"ADJ", "PROPN"}:
            descriptors.append(
                {
                    "text": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "kind": descriptor_kind(token),
                }
            )
    return {
        "text": text,
        "literal_source_span": text,
        "start": start,
        "end": end,
        "source_start": start,
        "source_end": end,
        "clean_entity_phrase": text,
        "head": head.text,
        "head_start": head.idx,
        "head_end": head.idx + len(head.text),
        "head_i": head.i,
        "head_lemma": norm(head.lemma_),
        "descriptors": descriptors,
        "local_modifiers": _surface_modifiers(text, head.text),
        "source_chunk": text,
        "root_dep": head.dep_,
        "origin": origin,
        "inherited_head": False,
        "antecedent_mention_id": None,
        "antecedent_provenance": None,
        "selector_like": bool(SELECTOR_RE.search(text)),
        "uncertainty": False,
        "uncertainty_reason": None,
    }


def _anaphor_literal_span(question, doc, token):
    members = [member for member in token.subtree]
    start = min([token.idx] + [member.idx for member in members])
    determiners = [
        child
        for child in doc
        if child.head == token
        and child.dep_ in {"det", "amod"}
        and (norm(child) in ANAPHOR_DETERMINERS)
    ]
    if determiners:
        start = min(start, min((child.idx for child in determiners)))
    end = max(
        [token.idx + len(token.text)]
        + [member.idx + len(member.text) for member in members]
    )
    return (start, end, question[start:end])


def recover_local_anaphors(question, doc, explicit):
    recovered = []
    for token in doc:
        if (
            norm(token) != "one"
            or norm(token.lemma_) != "one"
            or token.pos_ not in {"NOUN", "PRON", "NUM"}
            or (token.dep_ not in NOMINAL_DEPS)
        ):
            continue
        determiners = {
            norm(child)
            for child in doc
            if child.head == token and child.dep_ in {"det", "amod"}
        }
        if not determiners & ANAPHOR_DETERMINERS:
            continue
        prior = [
            mention
            for mention in explicit
            if mention["head_i"] < token.i and token.i - mention["head_i"] <= 16
        ]
        if not prior:
            continue
        classes = {mention["head_lemma"] for mention in prior}
        if len(classes) != 1:
            continue
        antecedent = max(prior, key=lambda mention: mention["head_i"])
        start, end, literal = _anaphor_literal_span(question, doc, token)
        recovered.append(
            {
                "text": antecedent["head"],
                "literal_source_span": literal,
                "start": start,
                "end": end,
                "source_start": start,
                "source_end": end,
                "clean_entity_phrase": antecedent["head"],
                "head": antecedent["head"],
                "head_start": None,
                "head_end": None,
                "head_i": token.i,
                "head_lemma": antecedent["head_lemma"],
                "descriptors": [],
                "local_modifiers": [],
                "source_chunk": literal,
                "root_dep": token.dep_,
                "origin": "local_nominal_anaphora",
                "inherited_head": True,
                "antecedent_object": antecedent,
                "antecedent_mention_id": None,
                "antecedent_provenance": None,
                "selector_like": False,
                "uncertainty": False,
                "uncertainty_reason": None,
            }
        )
    return recovered


def extract(question):
    scene = scene_context(question)
    if scene_only(question):
        return {
            "state": "SCENE_ONLY",
            "entity_instances": [],
            "scene_context": True,
            "no_entity": True,
        }
    doc = nlp(question)
    candidates = []
    for chunk in doc.noun_chunks:
        if is_candidate_head(chunk.root, doc):
            candidate = make_candidate(question, doc, chunk.root, "noun_chunk")
            if candidate:
                candidates.append(candidate)
    for token in doc:
        if is_candidate_head(token, doc):
            candidate = make_candidate(question, doc, token, "all_token_head")
            if candidate:
                candidates.append(candidate)
    unique = []
    seen = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["start"],
            item["end"],
            item["head_start"] if item["head_start"] is not None else -1,
        ),
    ):
        key = (candidate["start"], candidate["end"], candidate["head_start"])
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    final = []
    for candidate in unique:
        drop = False
        for other in unique:
            if candidate is other:
                continue
            if (
                candidate["start"] >= other["start"]
                and candidate["end"] <= other["end"]
                and (
                    candidate["start"] != other["start"]
                    or candidate["end"] != other["end"]
                )
            ):
                candidate_words = candidate["text"].lower().split()
                other_words = other["text"].lower().split()
                if candidate_words == other_words[: len(candidate_words)]:
                    drop = True
                    break
                if doc[candidate["head_i"]].dep_ == "compound":
                    drop = True
                    break
        if not drop:
            final.append(candidate)
    final.extend(recover_local_anaphors(question, doc, final))
    final.sort(key=lambda item: (item["start"], item["end"], item["head_i"]))
    for index, mention in enumerate(final, 1):
        mention["mention_id"] = f"e{index}"
    for mention in final:
        antecedent = mention.pop("antecedent_object", None)
        if antecedent is not None:
            mention["antecedent_mention_id"] = antecedent["mention_id"]
            mention["antecedent_provenance"] = {
                "mention_id": antecedent["mention_id"],
                "literal_head": antecedent["head"],
                "head_start": antecedent["head_start"],
                "head_end": antecedent["head_end"],
                "inheritance_basis": "unique_local_explicit_entity_class",
            }
        mention.pop("head_i", None)
        mention.pop("head_lemma", None)
    state = "OK" if final else "NO_ENTITY"
    public_fields = (
        "mention_id",
        "literal_source_span",
        "source_start",
        "source_end",
        "clean_entity_phrase",
        "head",
        "head_start",
        "head_end",
        "local_modifiers",
        "inherited_head",
        "antecedent_mention_id",
        "antecedent_provenance",
        "selector_like",
        "uncertainty",
        "uncertainty_reason",
        "origin",
    )
    entity_instances = [
        {field: mention[field] for field in public_fields} for mention in final
    ]
    return {
        "state": state,
        "entity_instances": entity_instances,
        "scene_context": scene,
        "no_entity": not bool(final),
    }
