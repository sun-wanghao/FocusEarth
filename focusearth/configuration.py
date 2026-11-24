"""MME configuration loading with explicit checks for fixed method settings."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .evaluation.mme_dataset import CATEGORY_COUNTS
from .models.image_text_encoders import MODEL_ORDER
from .models.qwen3_vl import GENERATION, MODEL_ID, MODEL_REVISION
from .region_prioritization.candidate_regions import VIEW_ORDER
from .region_prioritization.entropy_matching import (
    ENTROPY_SEARCH_STEPS,
    REGION_COUNT,
    STANDARD_DEVIATION_FLOOR,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
)
from .visual_representation.canvas_geometry import ALIGNED_CANVAS_SCALE
from .visual_representation.token_budget import RHO


FIXED_SETTINGS = {
    "dataset": {
        "name": "MME-RealWorld-RS",
        "question_count": sum(CATEGORY_COUNTS.values()),
    },
    "region_prioritization": {
        "candidate_grid": [6, 6],
        "expansion_per_side": 0.05,
        "views": list(VIEW_ORDER),
        "fusion": "entropy_matched_equal_weight",
        "temperature_range": [TEMPERATURE_MIN, TEMPERATURE_MAX],
        "entropy_search_steps": ENTROPY_SEARCH_STEPS,
        "standard_deviation_floor": STANDARD_DEVIATION_FLOOR,
    },
    "region_selection": {"region_count": REGION_COUNT},
    "visual_representation": {"canvas_scale": ALIGNED_CANVAS_SCALE, "rho": RHO},
    "answer_model": {
        "repository": MODEL_ID,
        "revision": MODEL_REVISION,
        "prompt_protocol": "mme_supplement_evidence_final_answer",
        "dtype": "bfloat16",
        "attention": "sdpa",
    },
    "generation": GENERATION,
}
_USER_SETTINGS = {
    "dataset": {"root", "annotations"},
    "scoring_encoders": {"checkpoints", "device", "image_batch_size"},
    "answer_model": {"checkpoint"},
    "output": {"directory"},
}


def load_configuration(
    path: str | Path,
    *,
    data_root: str | None = None,
    annotations: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Load paths and runtime settings, rejecting changes to the method protocol.

    Relative paths are resolved from the working directory. Dataset and output
    CLI arguments override the corresponding configuration paths.
    """
    configuration = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise ValueError("configuration must be a YAML mapping")
    configuration = deepcopy(configuration)
    expected_groups = set(FIXED_SETTINGS) | set(_USER_SETTINGS)
    if set(configuration) != expected_groups:
        raise ValueError(f"configuration groups must be {sorted(expected_groups)}")
    for group in expected_groups:
        section = configuration[group]
        if not isinstance(section, dict):
            raise ValueError(f"{group} must be a mapping")
        fixed = FIXED_SETTINGS.get(group, {})
        allowed = set(fixed) | _USER_SETTINGS.get(group, set())
        if set(section) != allowed:
            raise ValueError(f"{group} keys must be {sorted(allowed)}")
        for key, expected in fixed.items():
            if section[key] != expected:
                raise ValueError(f"{group}.{key} is fixed at {expected!r}")
    for group, key, override in (
        ("dataset", "root", data_root),
        ("dataset", "annotations", annotations),
        ("output", "directory", output_dir),
    ):
        if override is not None:
            configuration[group][key] = override
    encoders = configuration["scoring_encoders"]
    if not isinstance(encoders["checkpoints"], dict) or set(
        encoders["checkpoints"]
    ) != set(MODEL_ORDER):
        raise ValueError(f"scoring checkpoint keys must be {MODEL_ORDER}")
    if (
        type(encoders["image_batch_size"]) is not int
        or encoders["image_batch_size"] <= 0
    ):
        raise ValueError("scoring_encoders.image_batch_size must be a positive integer")
    if not isinstance(encoders["device"], str) or not encoders["device"].strip():
        raise ValueError("scoring_encoders.device must be a nonempty device name")
    paths = [
        ("dataset", "root"),
        ("dataset", "annotations"),
        ("answer_model", "checkpoint"),
        ("output", "directory"),
    ]
    for group, key in paths:
        value = configuration[group][key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{group}.{key} must be supplied")
        configuration[group][key] = str(Path(value).expanduser())
    for name, value in encoders["checkpoints"].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"scoring_encoders.checkpoints.{name} must be supplied")
        encoders["checkpoints"][name] = str(Path(value).expanduser())
    return configuration


def configuration_for_manifest(configuration: dict[str, Any]) -> dict[str, Any]:
    """Retain effective settings while replacing local paths with basenames."""
    result = deepcopy(configuration)
    for group, key in (
        ("dataset", "root"),
        ("dataset", "annotations"),
        ("answer_model", "checkpoint"),
        ("output", "directory"),
    ):
        result[group][key] = Path(result[group][key]).name
    result["scoring_encoders"]["checkpoints"] = {
        name: Path(value).name
        for name, value in result["scoring_encoders"]["checkpoints"].items()
    }
    return result
