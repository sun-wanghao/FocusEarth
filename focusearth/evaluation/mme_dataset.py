"""Read the official MME-RealWorld annotation format without dataset rewriting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
from typing import Any

CATEGORY_COUNTS = {"color": 1255, "count": 1226, "position": 1257}


@dataclass(frozen=True)
class MMESample:
    """One remote-sensing question in the annotation's original occurrence order."""

    question_id: str
    category: str
    question: str
    answer_choices: tuple[str, ...]
    ground_truth: str
    image_relative_path: str
    image_path: Path
    annotation_index: int

    def identity(self) -> dict[str, Any]:
        """Dataset identity fields independent of a caller's filesystem root."""
        return {
            "question_id": self.question_id,
            "category": self.category,
            "question": self.question,
            "answer_choices": list(self.answer_choices),
            "ground_truth": self.ground_truth,
            "image_relative_path": self.image_relative_path,
            "annotation_index": self.annotation_index,
        }


def _safe_image_path(data_root: Path, relative: str) -> Path:
    # The public annotation uses POSIX relative paths. Lexical containment permits
    # legitimate owner-created symlinks while rejecting traversal and private paths.
    part = PurePosixPath(relative)
    if not relative or part.is_absolute() or ".." in part.parts or "\\" in relative:
        raise ValueError(f"image path must be a safe relative POSIX path: {relative!r}")
    return data_root.joinpath(*part.parts)


def load_mme_dataset(
    data_root: str | Path,
    annotations: str | Path,
    *,
    strict_population: bool = True,
    verify_images: bool = True,
) -> list[MMESample]:
    """Load the official JSON list, retaining only ``Subtask=Remote Sensing``.

    ``data_root`` contains paths such as ``remote_sensing/example.png``. Ground
    truth and original option strings are preserved. The default requires the
    canonical 3,738-question population and exact category counts. Disabling that
    check is intended for synthetic tests and must not label a subset a full run.
    Images and annotations are supplied by the caller.
    """
    data_root = Path(data_root).expanduser().absolute()
    annotation_path = Path(annotations).expanduser()
    values = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("MME annotations must be the official JSON list")
    output: list[MMESample] = []
    seen: set[str] = set()
    required = (
        "Question_id",
        "Image",
        "Text",
        "Answer choices",
        "Ground truth",
        "Category",
    )
    for ordinal, row in enumerate(values):
        if not isinstance(row, dict):
            raise ValueError(f"annotation item {ordinal} is not an object")
        if row.get("Subtask") != "Remote Sensing":
            continue
        if any(key not in row for key in required):
            raise ValueError(
                f"remote-sensing item {ordinal} is missing required fields"
            )
        qid, relative, question, category, truth = (
            row["Question_id"],
            row["Image"],
            row["Text"],
            row["Category"],
            row["Ground truth"],
        )
        if any(
            not isinstance(value, str) or not value
            for value in (qid, relative, question, category, truth)
        ):
            raise ValueError(
                f"remote-sensing item {ordinal} has invalid text identity fields"
            )
        if qid in seen:
            raise ValueError(f"duplicate question ID: {qid}")
        if category not in CATEGORY_COUNTS:
            raise ValueError(f"unsupported remote-sensing category: {category!r}")
        if truth not in "ABCDE" or len(truth) != 1:
            raise ValueError(f"invalid ground-truth option for {qid}: {truth!r}")
        choices = row["Answer choices"]
        if (
            not isinstance(choices, list)
            or len(choices) != 5
            or any(not isinstance(x, str) for x in choices)
        ):
            raise ValueError(f"expected five original answer choices for {qid}")
        # Keep blank option content: some official options consist only of labels.
        if any(
            not choice.startswith(f"({chr(65+i)})") for i, choice in enumerate(choices)
        ):
            raise ValueError(f"option letters must retain original A–E order for {qid}")
        image_path = _safe_image_path(data_root, relative)
        if verify_images and not image_path.is_file():
            raise FileNotFoundError(f"missing MME source image for {qid}: {image_path}")
        seen.add(qid)
        output.append(
            MMESample(
                qid,
                category,
                question,
                tuple(choices),
                truth,
                relative,
                image_path,
                ordinal,
            )
        )
    if not output:
        raise ValueError("no Remote Sensing questions found in annotations")
    counts = dict(Counter(sample.category for sample in output))
    if strict_population and (len(output) != 3738 or counts != CATEGORY_COUNTS):
        raise ValueError(
            f"expected the canonical 3,738-question population {CATEGORY_COUNTS}; got {len(output)} {counts}"
        )
    return output
