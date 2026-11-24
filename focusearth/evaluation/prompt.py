"""Supplement-aligned public MME answering prompt."""

from __future__ import annotations

from collections.abc import Sequence

PROMPT_PROTOCOL = "mme_supplement_evidence_final_answer"
VISUAL_DESCRIPTION = (
    "Image 1 is a single full-scene image obtained by bilinearly resizing the source "
    "image. Outlined regions mark selected evidence locations at their original "
    "relative positions; they indicate locations only "
    "and do not encode the answer."
)
RESPONSE_FORMAT = (
    "Use the following response format:\n\n"
    "Evidence: [A concise, image-grounded observation-based explanation]\n\n"
    "Final answer: [A single MME option letter]"
)


def canonical_mme_question(question: str, answer_choices: Sequence[str]) -> str:
    """Retain the original question and choice strings, including blank choices."""
    if not isinstance(question, str) or not question:
        raise ValueError("question must be a nonempty string")
    if len(answer_choices) != 5 or any(
        not isinstance(choice, str) for choice in answer_choices
    ):
        raise ValueError("MME requires the five original option strings")
    return question + " The choices are listed below:\n" + "\n".join(answer_choices)


def build_mme_prompt(question: str, answer_choices: Sequence[str]) -> str:
    """Render the supplement's representative prompt before chat-template wrapping.

    TeX paragraph breaks become blank lines. The question placeholder retains
    original question and choice strings, including option labels.
    """
    return (
        "You are an intelligent remote sensing analyst.\n\n"
        + VISUAL_DESCRIPTION
        + "\n\nQuestion: "
        + canonical_mme_question(question, answer_choices)
        + "\n\n"
        + RESPONSE_FORMAT
    )
