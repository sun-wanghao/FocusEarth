"""Population-normalized, entropy-matched fusion of four encoder distributions."""

from __future__ import annotations

from typing import Any

import numpy as np


ENCODER_COUNT = 4
VIEW_COUNT = 5
CANDIDATE_COUNT = 36
REGION_COUNT = 6
STANDARD_DEVIATION_FLOOR = 1e-6
TEMPERATURE_MIN = 0.25
TEMPERATURE_MAX = 4.0
ENTROPY_SEARCH_STEPS = 40


def _entropy(values: np.ndarray) -> np.ndarray:
    """Natural-log entropy with the 0*log(0)=0 convention."""
    values = np.asarray(values, np.float64)
    return -np.sum(
        np.where(values > 0, values * np.log(np.maximum(values, 1e-300)), 0.0),
        axis=-1,
    )


def rank_top_regions(fused: np.ndarray) -> np.ndarray:
    """Stable descending Top-6; lower zero-based patch index wins exact ties."""
    fused = np.asarray(fused)
    if fused.shape[-1] != CANDIDATE_COUNT:
        raise ValueError(f"expected final patch axis 36, got {fused.shape}")
    return np.argsort(-fused, axis=-1, kind="stable")[..., :REGION_COUNT]


def _stable_softmax(values: np.ndarray) -> np.ndarray:
    """Stable float64 softmax used by entropy matching."""
    z = np.asarray(values, float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def entropy_matched_fusion(raw_scores: np.ndarray) -> dict[str, Any]:
    """Four-scorer entropy-matching rule on [...,4,5,36] raw views.

    Each scorer distribution is entropy-matched to the median temperature-one
    entropy. All four matched distributions receive equal aggregation weight.
    """
    raw = np.asarray(raw_scores, np.float32)
    if raw.shape[-3:] != (ENCODER_COUNT, VIEW_COUNT, CANDIDATE_COUNT):
        raise ValueError(f"expected [...,4,5,36], got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("raw scores must be finite")
    leading = raw.shape[:-3]
    flat = raw.reshape(-1, ENCODER_COUNT, VIEW_COUNT, CANDIDATE_COUNT)
    fused_rows = []
    temperatures_rows = []
    targets = []
    for item in flat:
        score = item.max(axis=-2)
        z = (
            (score - score.mean(-1, keepdims=True))
            / np.maximum(
                score.std(-1, keepdims=True), np.float32(STANDARD_DEVIATION_FLOOR)
            )
        ).astype(float)
        target = float(np.median(_entropy(_stable_softmax(z))))
        matched = []
        temperatures = []
        for model in range(ENCODER_COUNT):
            lo, hi = TEMPERATURE_MIN, TEMPERATURE_MAX
            for _ in range(ENTROPY_SEARCH_STEPS):
                mid = (lo + hi) / 2.0
                h = float(_entropy(_stable_softmax(z[model] / mid)))
                if h < target:
                    lo = mid
                else:
                    hi = mid
            candidates = [
                (lo, _stable_softmax(z[model] / lo)),
                (hi, _stable_softmax(z[model] / hi)),
            ]
            temperature, distribution = min(
                candidates, key=lambda x: abs(float(_entropy(x[1])) - target)
            )
            temperatures.append(temperature)
            matched.append(distribution)
        fused_rows.append(np.mean(matched, axis=0))
        temperatures_rows.append(temperatures)
        targets.append(target)
    fused = np.asarray(fused_rows, np.float64).reshape(leading + (CANDIDATE_COUNT,))
    temperatures = np.asarray(temperatures_rows, np.float64).reshape(
        leading + (ENCODER_COUNT,)
    )
    target_entropy = np.asarray(targets, np.float64).reshape(leading)
    if not np.isfinite(fused).all() or not np.allclose(
        fused.sum(axis=-1), 1.0, rtol=0, atol=1e-12
    ):
        raise FloatingPointError("non-finite or unnormalized entropy-matched output")
    return {
        "fused": fused,
        "top_region_indices": rank_top_regions(fused),
        "temperatures": temperatures,
        "target_entropy": target_entropy,
    }
