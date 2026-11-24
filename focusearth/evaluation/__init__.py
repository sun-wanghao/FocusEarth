"""MME-RealWorld-RS loading, answering prompt, and option-letter evaluation."""

from .mme_dataset import MMESample, load_mme_dataset
from .mme_metrics import compute_metrics, parse_mme_answer, score_response
from .prompt import build_mme_prompt

__all__ = [
    "MMESample",
    "load_mme_dataset",
    "compute_metrics",
    "parse_mme_answer",
    "score_response",
    "build_mme_prompt",
]
