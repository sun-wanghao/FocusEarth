"""Source-offset-preserving removal of MME question operators."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


ALLOWED_REMOVAL_REASONS = {
    "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME",
    "COUNT_INTERROGATIVE_OPERATOR",
    "LOCATION_INTERROGATIVE_OPERATOR",
    "EXISTENCE_INTERROGATIVE_OPERATOR",
    "TARGET_INITIAL_DETERMINER_CLEANUP",
    "COUNT_COPULA_OR_EXPLETIVE_CLEANUP",
    "POSITIVE_VISIBILITY_OPERATOR_CLEANUP",
    "LOCATION_OPERATOR_CLEANUP",
    "TERMINAL_QUESTION_PUNCTUATION",
    "GRAMMATICAL_WHITESPACE_OR_PUNCTUATION_CLEANUP",
}


@dataclass
class Unit:
    char: str
    source_index: int


class MappedText:
    """Text whose surviving characters retain offsets into its source string."""

    def __init__(self, text: str):
        self.source = text
        self.units = [Unit(char, index) for index, char in enumerate(text)]
        self.operations: list[dict[str, Any]] = []

    @property
    def text(self) -> str:
        return "".join((unit.char for unit in self.units))

    def _source_segments(self, start: int, end: int) -> list[tuple[int, int, str]]:
        units = self.units[start:end]
        if not units:
            return []
        groups: list[list[Unit]] = [[units[0]]]
        for unit in units[1:]:
            if unit.source_index == groups[-1][-1].source_index + 1:
                groups[-1].append(unit)
            else:
                groups.append([unit])
        return [
            (g[0].source_index, g[-1].source_index + 1, "".join((x.char for x in g)))
            for g in groups
        ]

    def remove(self, start: int, end: int, reason: str, rule_id: str) -> None:
        if start == end:
            return
        if reason not in ALLOWED_REMOVAL_REASONS:
            raise RuntimeError(f"unregistered removal reason: {reason}")
        for source_start, source_end, text in self._source_segments(start, end):
            self.operations.append(
                {
                    "stage": "SOURCE_DESCRIPTION",
                    "operation": "DELETE",
                    "rule_id": rule_id,
                    "reason": reason,
                    "text": text,
                    "source_start": source_start,
                    "source_end": source_end,
                }
            )
        del self.units[start:end]

    def remove_match(
        self, match: re.Match[str], reason: str, rule_id: str, group: int | str = 0
    ) -> None:
        start, end = match.span(group)
        self.remove(start, end, reason, rule_id)

    def remove_regex(
        self, pattern: str, reason: str, rule_id: str, flags: int = re.I, start: int = 0
    ) -> bool:
        match = re.search(pattern, self.text[start:], flags)
        if not match:
            return False
        absolute_start = start + match.start()
        absolute_end = start + match.end()
        self.remove(absolute_start, absolute_end, reason, rule_id)
        return True

    def normalize(self, stage: str) -> None:
        reason = "GRAMMATICAL_WHITESPACE_OR_PUNCTUATION_CLEANUP"
        while True:
            match = re.search("\\?\\s*$", self.text)
            if not match:
                break
            self.remove(
                match.start(),
                match.end(),
                "TERMINAL_QUESTION_PUNCTUATION",
                f"{stage}_DROP_TERMINAL_QUESTION",
            )
        match = re.match("^[\\s,]+", self.text)
        if match:
            self.remove(match.start(), match.end(), reason, f"{stage}_TRIM_LEFT")
        match = re.search("[\\s,]+$", self.text)
        if match:
            self.remove(match.start(), match.end(), reason, f"{stage}_TRIM_RIGHT")
        while True:
            match = re.search("\\s{2,}", self.text)
            if not match:
                break
            self.remove(
                match.start() + 1, match.end(), reason, f"{stage}_COLLAPSE_WHITESPACE"
            )
        match = re.search(",\\s*$", self.text)
        if match:
            self.remove(
                match.start(), match.end(), reason, f"{stage}_DROP_DANGLING_COMMA"
            )


def remove_group_reverse(
    mt: MappedText,
    match: re.Match[str],
    groups: list[int | str],
    reason: str,
    rule_id: str,
) -> None:
    spans = sorted(((match.start(group), match.end(group)) for group in groups))
    for start, end in reversed(spans):
        mt.remove(start, end, reason, rule_id)


def strip_target_determiner(mt: MappedText) -> None:
    match = re.match("^(?:the|a|an)\\s+", mt.text, re.I)
    if match:
        mt.remove_match(
            match,
            "TARGET_INITIAL_DETERMINER_CLEANUP",
            "SOURCE_DROP_INITIAL_TARGET_DETERMINER",
        )
        return
    match = re.search("(?<=,)\\s*(?:the|a|an)\\s+", mt.text, re.I)
    if match:
        text = match.group(0)
        leading = len(text) - len(text.lstrip())
        start = match.start() + min(1, leading)
        mt.remove(
            start,
            match.end(),
            "TARGET_INITIAL_DETERMINER_CLEANUP",
            "SOURCE_DROP_EMBEDDED_TARGET_DETERMINER",
        )


def remove_first_aux_after(mt: MappedText, start: int, rule: str) -> None:
    segment = mt.text[start:]
    match = re.search("\\b(?:is|are)\\s+there\\b", segment, re.I)
    if match:
        mt.remove(
            start + match.start(),
            start + match.end(),
            "COUNT_COPULA_OR_EXPLETIVE_CLEANUP",
            rule,
        )
        return
    match = re.search("\\b(?:is|are)\\b", segment, re.I)
    if match:
        mt.remove(
            start + match.start(),
            start + match.end(),
            "COUNT_COPULA_OR_EXPLETIVE_CLEANUP",
            rule,
        )


def remove_question_operators(category: str, question: str) -> dict[str, Any]:
    """Remove supported question frames while preserving source offsets."""
    mt = MappedText(question)
    question_form = "UNHANDLED"
    matched_rule = ""
    ambiguous = False
    failure_reason = ""
    matched = False
    match = re.search("\\bIs there any\\s+", mt.text, re.I)
    if match:
        mt.remove_match(
            match, "EXISTENCE_INTERROGATIVE_OPERATOR", "SOURCE_EXISTENCE_IS_THERE_ANY"
        )
        question_form, matched_rule, matched = (
            "EXISTENCE",
            "SOURCE_EXISTENCE_IS_THERE_ANY",
            True,
        )
    if not matched:
        match = re.search("\\bHow man(?P<ending>y|g)\\s+", mt.text, re.I)
        if match:
            typo = match.group("ending").lower() == "g"
            start = match.start()
            mt.remove_match(
                match,
                "COUNT_INTERROGATIVE_OPERATOR",
                (
                    "SOURCE_COUNT_HOW_MANY"
                    if not typo
                    else "SOURCE_COUNT_HOW_MANG_EXPLICIT_TYPO"
                ),
            )
            remove_first_aux_after(mt, start, "SOURCE_COUNT_DROP_COPULA_OR_THERE")
            match_visible = re.search("\\bcan be seen\\b", mt.text[start:], re.I)
            if match_visible:
                mt.remove(
                    start + match_visible.start(),
                    start + match_visible.end(),
                    "POSITIVE_VISIBILITY_OPERATOR_CLEANUP",
                    "SOURCE_COUNT_DROP_POSITIVE_CAN_BE_SEEN",
                )
            question_form, matched_rule, matched = (
                "COUNT",
                "SOURCE_COUNT_HOW_MANY",
                True,
            )
            if typo:
                ambiguous = True
                failure_reason = "Known dataset typo 'How mang' was interpreted as the count operator; this interpretation is explicit, not silent."
    if not matched:
        match = re.search(
            "\\bWhat is the (?P<aspect>background|top|filling|ground) color of\\b",
            mt.text,
            re.I,
        )
        if match:
            color_start = match.start("aspect") + len(match.group("aspect"))
            color_end = match.end()
            color_match = re.search(
                "\\s+color(?=\\s+of\\b)", mt.text[color_start:color_end], re.I
            )
            if not color_match:
                raise RuntimeError("aspect-color subspan not found")
            mt.remove(
                color_start + color_match.start(),
                color_start + color_match.end(),
                "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME",
                "SOURCE_COLOR_ASPECT_PRESERVE",
            )
            prefix = re.search("\\bWhat is the\\s+", mt.text, re.I)
            assert prefix
            mt.remove_match(
                prefix,
                "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME",
                "SOURCE_COLOR_ASPECT_PRESERVE",
            )
            question_form, matched_rule, matched = (
                "COLOR_ATTRIBUTE",
                "SOURCE_COLOR_ASPECT_PRESERVE",
                True,
            )
    if not matched:
        color_prefixes = [
            ("\\bWhat is the main color tone of\\s+", "SOURCE_COLOR_MAIN_TONE_OF"),
            ("\\bWhat is the main color of\\s+", "SOURCE_COLOR_MAIN_OF"),
            ("\\bWhat main color is\\s+", "SOURCE_COLOR_MAIN_IS"),
            ("\\bWhat is the colou?r of\\s+", "SOURCE_COLOR_OF"),
            ("\\bWhat's the colou?r of\\s+", "SOURCE_COLOR_CONTRACTION_OF"),
            ("\\bWhat are the main colors of\\s+", "SOURCE_COLORS_MAIN_OF"),
            ("\\bWhat are the colors of\\s+", "SOURCE_COLORS_OF"),
            ("\\bWhat color appears of\\s+", "SOURCE_COLOR_APPEARS_OF"),
        ]
        for pattern, rule in color_prefixes:
            match = re.search(pattern, mt.text, re.I)
            if match:
                mt.remove_match(match, "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME", rule)
                question_form = (
                    "COLOR_SET_ATTRIBUTE" if "COLORS" in rule else "COLOR_ATTRIBUTE"
                )
                matched_rule, matched = (rule, True)
                if rule == "SOURCE_COLOR_APPEARS_OF":
                    ambiguous = True
                    failure_reason = "Malformed color template 'What color appears of' was reduced conservatively."
                break
    if not matched:
        match = re.search(
            "\\b(?:Which|What) color(?: of car)? has\\s+(?:the\\s+)?", mt.text, re.I
        )
        if match:
            mt.remove_match(
                match,
                "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME",
                "SOURCE_COLOR_GROUP_COMPARISON",
            )
            question_form, matched_rule, matched = (
                "COLOR_GROUP_COMPARISON",
                "SOURCE_COLOR_GROUP_COMPARISON",
                True,
            )
            ambiguous = True
            failure_reason = "The color is the grouping/answer variable; the remaining comparative evidence is preserved but does not name a single local target."
    if not matched:
        direct_patterns = [
            ("\\bWhat colors are\\s+", "SOURCE_COLORS_ARE", "COLOR_SET_ATTRIBUTE"),
            ("\\bWhat color are\\s+", "SOURCE_COLOR_ARE", "COLOR_ATTRIBUTE"),
            ("\\bWhat color is\\s+", "SOURCE_COLOR_IS", "COLOR_ATTRIBUTE"),
            ("\\bWhat color of\\s+", "SOURCE_COLOR_OF_NOMINAL", "COLOR_ATTRIBUTE"),
        ]
        for pattern, rule, form in direct_patterns:
            match = re.search(pattern, mt.text, re.I)
            if match:
                mt.remove_match(match, "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME", rule)
                question_form, matched_rule, matched = (form, rule, True)
                break
    if not matched:
        match = re.search(
            "\\bWhat color\\s+(?P<subject>.+?)\\s+(?P<aux>is|are)\\b", mt.text, re.I
        )
        if match:
            remove_group_reverse(
                mt,
                match,
                ["aux"],
                "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME",
                "SOURCE_COLOR_NOMINAL_COPULA",
            )
            prefix = re.search("\\bWhat color\\s+", mt.text, re.I)
            assert prefix
            mt.remove_match(
                prefix,
                "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME",
                "SOURCE_COLOR_NOMINAL_COPULA",
            )
            mt.remove_regex(
                "\\bthere\\b",
                "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME",
                "SOURCE_COLOR_NOMINAL_DROP_THERE",
            )
            question_form, matched_rule, matched = (
                "COLOR_ATTRIBUTE",
                "SOURCE_COLOR_NOMINAL_COPULA",
                True,
            )
    if not matched:
        location_prefixes = [
            (
                "\\bIn which area of the picture\\s+(?:is|are)\\s+",
                "SOURCE_LOCATION_IN_WHICH_AREA_OF_PICTURE",
            ),
            (
                "\\bIn which area\\s+(?:is|are)(?:\\s+there)?\\s+",
                "SOURCE_LOCATION_IN_WHICH_AREA",
            ),
            (
                "\\bWhat area of the map\\s+(?:is|are)\\s+",
                "SOURCE_LOCATION_WHAT_AREA_OF_MAP",
            ),
            ("\\bWhat area\\s+(?:is|are)\\s+", "SOURCE_LOCATION_WHAT_AREA"),
            ("\\bWhich area\\s+(?:is|are)\\s+", "SOURCE_LOCATION_WHICH_AREA"),
            ("\\bWhat is the location of\\s+", "SOURCE_LOCATION_WHAT_IS_LOCATION_OF"),
            ("\\bWhere\\s+(?:is|are)\\s+", "SOURCE_LOCATION_WHERE_COPULA"),
        ]
        for pattern, rule in location_prefixes:
            match = re.search(pattern, mt.text, re.I)
            if match:
                mt.remove_match(match, "LOCATION_INTERROGATIVE_OPERATOR", rule)
                question_form, matched_rule, matched = ("LOCATION", rule, True)
                break
    if not matched:
        match = re.search("\\bWhere\\s+(?P<do>do|does)\\s+", mt.text, re.I)
        if match:
            mt.remove_match(
                match, "LOCATION_INTERROGATIVE_OPERATOR", "SOURCE_LOCATION_WHERE_DOES"
            )
            fit = re.search("\\bfit\\b", mt.text, re.I)
            if fit:
                mt.remove_match(
                    fit, "LOCATION_OPERATOR_CLEANUP", "SOURCE_LOCATION_DROP_FIT"
                )
            question_form, matched_rule, matched = (
                "LOCATION",
                "SOURCE_LOCATION_WHERE_DOES",
                True,
            )
    if not matched:
        match = re.search(
            "\\bWhere\\s+(?P<subject>.+?)\\s+(?P<aux>is|are)\\s+(?=(?:in|on|at|near|beside|under|above|below)\\b)",
            mt.text,
            re.I,
        )
        if match:
            remove_group_reverse(
                mt,
                match,
                ["aux"],
                "LOCATION_INTERROGATIVE_OPERATOR",
                "SOURCE_LOCATION_MALFORMED_WHERE_SUBJECT_IS",
            )
            prefix = re.search("\\bWhere\\s+", mt.text, re.I)
            assert prefix
            mt.remove_match(
                prefix,
                "LOCATION_INTERROGATIVE_OPERATOR",
                "SOURCE_LOCATION_MALFORMED_WHERE_SUBJECT_IS",
            )
            question_form, matched_rule, matched = (
                "LOCATION",
                "SOURCE_LOCATION_MALFORMED_WHERE_SUBJECT_IS",
                True,
            )
            ambiguous = True
            failure_reason = "Malformed location word order was handled by deletion only; no content words were reordered."
    if not matched:
        match = re.search(
            "\\bWhich position in the picture is full of\\s+", mt.text, re.I
        )
        if match:
            mt.remove_match(
                match,
                "LOCATION_INTERROGATIVE_OPERATOR",
                "SOURCE_LOCATION_WHICH_POSITION_FULL_OF",
            )
            question_form, matched_rule, matched = (
                "LOCATION_SELECTOR",
                "SOURCE_LOCATION_WHICH_POSITION_FULL_OF",
                True,
            )
            ambiguous = True
            failure_reason = "The answer variable is a position; only the explicit visual evidence after 'full of' is representable without inventing a region."
    if not matched:
        match = re.search("\\bWhich\\s+(?=water area\\b)", mt.text, re.I)
        if match:
            mt.remove_match(
                match,
                "LOCATION_INTERROGATIVE_OPERATOR",
                "SOURCE_LOCATION_WHICH_WATER_AREA",
            )
            question_form, matched_rule, matched = (
                "LOCATION_SELECTOR",
                "SOURCE_LOCATION_WHICH_WATER_AREA",
                True,
            )
            ambiguous = True
            failure_reason = "The selected water area is an answer variable; the explicit class and selector clause are retained without resolving it."
    if not matched and category.lower() in {"color", "position"}:
        match = re.search("\\bWhat is\\s+", mt.text, re.I)
        if match:
            mt.remove_match(
                match,
                "ANSWER_VARIABLE_AND_INTERROGATIVE_FRAME",
                "SOURCE_ANOMALOUS_WHAT_IS",
            )
            question_form, matched_rule, matched = (
                "ANOMALOUS_WHAT_IS",
                "SOURCE_ANOMALOUS_WHAT_IS",
                True,
            )
            ambiguous = True
            failure_reason = "The dataset category and surface question do not expose the expected answer variable cleanly; only the generic interrogative frame was removed."
    if not matched:
        return {
            "source_description": "",
            "question_form": "FAILED_UNHANDLED",
            "matched_rule": "NO_RULE_MATCH",
            "operations": [],
            "ambiguous": False,
            "failed": True,
            "failure_reason": "No deterministic answer-variable frame matched; output was fail-closed rather than semantically repaired.",
        }
    match = re.search("\\blocated\\s+in\\s*$|\\blocated\\s*$", mt.text, re.I)
    if match:
        mt.remove_match(
            match, "LOCATION_OPERATOR_CLEANUP", "SOURCE_DROP_DANGLING_LOCATED"
        )
    strip_target_determiner(mt)
    mt.normalize("SOURCE_DESCRIPTION")
    match = re.search("\\blocated\\s+in\\s*$|\\blocated\\s*$", mt.text, re.I)
    if match:
        mt.remove_match(
            match,
            "LOCATION_OPERATOR_CLEANUP",
            "SOURCE_DROP_DANGLING_LOCATED_AFTER_PUNCTUATION",
        )
        mt.normalize("SOURCE_DESCRIPTION")
    if not mt.text:
        return {
            "source_description": "",
            "question_form": question_form,
            "matched_rule": matched_rule,
            "operations": mt.operations,
            "ambiguous": ambiguous,
            "failed": True,
            "failure_reason": "Answer-variable removal left no faithful visual text.",
        }
    return {
        "source_description": mt.text,
        "question_form": question_form,
        "matched_rule": matched_rule,
        "operations": mt.operations,
        "ambiguous": ambiguous,
        "failed": False,
        "failure_reason": failure_reason,
    }
