#!/usr/bin/env python3
"""Run complete FocusEarth inference and evaluation on MME-RealWorld-RS."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from focusearth.configuration import configuration_for_manifest, load_configuration
from focusearth.evaluation.mme_dataset import load_mme_dataset
from focusearth.evaluation.mme_metrics import compute_metrics, score_response
from focusearth.models.qwen3_vl import MODEL_ID, MODEL_REVISION, load_qwen3_vl
from focusearth.models.image_text_encoders import CHECKPOINT_HASHES, ImageTextEncoders
from focusearth.io_utils import file_sha256
from focusearth.pipeline import prioritize_sample, answer_prioritized_sample


def _jsonl_write(handle, value):
    handle.write(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )
    handle.flush()


def build_argument_parser() -> argparse.ArgumentParser:
    """Expose the MME configuration, dataset, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/mme_realworld_rs.yaml"),
        help="MME configuration with external checkpoint paths",
    )
    parser.add_argument("--data-root", help="override dataset.root")
    parser.add_argument("--annotations", help="override dataset.annotations")
    parser.add_argument(
        "--output-dir", help="override output.directory; must not exist"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Score regions, answer all MME questions, and write evaluation artifacts."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        config = load_configuration(
            args.config,
            data_root=args.data_root,
            annotations=args.annotations,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    data_root = config["dataset"]["root"]
    annotations = config["dataset"]["annotations"]
    qwen_path = config["answer_model"]["checkpoint"]
    checkpoints = config["scoring_encoders"]["checkpoints"]
    device = config["scoring_encoders"]["device"]
    image_batch_size = config["scoring_encoders"]["image_batch_size"]
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    output = Path(config["output"]["directory"]).absolute()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output path: {output}")
    output.mkdir(parents=True)
    samples = load_mme_dataset(data_root, annotations)
    scorer = ImageTextEncoders(checkpoints, device=device, batch_size=image_batch_size)
    checkpoint_hashes = dict(scorer.checkpoint_hashes)
    predictions_path, traces_path = (
        output / "predictions.jsonl",
        output / "traces.jsonl",
    )
    priority_path = output / ".priority_work.jsonl"
    with priority_path.open("x", encoding="utf-8") as work:
        for index, sample in enumerate(samples, 1):
            try:
                row = {
                    "question_id": sample.question_id,
                    "priority": prioritize_sample(sample, scorer),
                }
            except Exception as exc:
                row = {
                    "question_id": sample.question_id,
                    "priority_error": f"{type(exc).__name__}: {exc}",
                }
            _jsonl_write(work, row)
            if index % 25 == 0:
                print(f"scored and selected {index}/{len(samples)}", flush=True)
    scorer.close()
    del scorer
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    model, processor = load_qwen3_vl(qwen_path)
    priorities = {
        row["question_id"]: row
        for row in (
            json.loads(line)
            for line in priority_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    with predictions_path.open("x", encoding="utf-8") as predictions, traces_path.open(
        "x", encoding="utf-8"
    ) as traces:
        for index, sample in enumerate(samples, 1):
            try:
                priority = priorities[sample.question_id]
                if "priority_error" in priority:
                    raise RuntimeError(priority["priority_error"])
                prediction, trace = answer_prioritized_sample(
                    sample, priority["priority"], model, processor
                )
                prediction.update(
                    score_response(prediction["raw_response"], sample.ground_truth)
                )
                prediction["status"] = "OK"
            except Exception as exc:
                prediction = {
                    "question_id": sample.question_id,
                    "category": sample.category,
                    "question": sample.question,
                    "ground_truth": sample.ground_truth,
                    "raw_response": "",
                    "extracted_candidate": "",
                    "extraction_method": "execution_error",
                    "answer": None,
                    "valid": False,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": "ERROR",
                }
                trace = {
                    "question_id": sample.question_id,
                    "status": "ERROR",
                    "error": prediction["error"],
                }
            _jsonl_write(predictions, prediction)
            _jsonl_write(traces, trace)
            if index % 25 == 0:
                print(f"completed {index}/{len(samples)}", flush=True)
    priority_path.unlink()
    records = [
        json.loads(line) for line in predictions_path.read_text().splitlines() if line
    ]
    results = compute_metrics(records, expected_rows=len(samples))
    (output / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    try:
        versions = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "open_clip_torch",
                "Pillow",
                "numpy",
                "spacy",
            )
        }
    except importlib.metadata.PackageNotFoundError:
        versions = {}
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "effective_configuration": configuration_for_manifest(config),
        "config_sha256": file_sha256(args.config),
        "benchmark": "MME-RealWorld-RS",
        "question_count": len(samples),
        "category_counts": dict(Counter(row.category for row in samples)),
        "annotation_sha256": file_sha256(annotations),
        "scoring_models": {
            key: {
                "checkpoint_file": Path(value).name,
                "sha256": checkpoint_hashes[key],
                "reference_sha256": CHECKPOINT_HASHES[key],
            }
            for key, value in checkpoints.items()
        },
        "answer_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "checkpoint_directory_name": Path(qwen_path).name,
            "prompt_protocol": "mme_supplement_evidence_final_answer",
            "dtype": "bfloat16",
            "attention": "sdpa",
        },
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 64,
            "use_cache": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "libraries": versions,
            "device_request": device,
            "image_batch_size": image_batch_size,
        },
        "scoring_cost_boundary": "Four-encoder, five-view region scoring runs offline before Qwen answering for each question; it is included in this complete pipeline run.",
        "results": results,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
