"""Configuration and command-output contracts without model execution."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from focusearth.configuration import configuration_for_manifest, load_configuration
from focusearth.models.image_text_encoders import MODEL_ORDER

ROOT = Path(__file__).resolve().parents[1]


def fixture_configuration() -> dict:
    configuration = yaml.safe_load((ROOT / "configs/mme_realworld_rs.yaml").read_text())
    configuration["dataset"].update(root="dataset", annotations="annotations.json")
    configuration["answer_model"]["checkpoint"] = "models/qwen"
    configuration["output"]["directory"] = "outputs/mme"
    configuration["scoring_encoders"]["checkpoints"] = {
        name: f"models/{name}.pt" for name in MODEL_ORDER
    }
    return configuration


def load_entry_point():
    spec = importlib.util.spec_from_file_location(
        "mme_command", ROOT / "scripts/eval_mme_realworld_rs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ConfigurationTests(unittest.TestCase):
    def test_external_paths_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configuration.yaml"
            path.write_text(yaml.safe_dump(fixture_configuration()))
            loaded = load_configuration(
                path, data_root="other_dataset", output_dir="other_output"
            )
            self.assertEqual(loaded["dataset"]["root"], "other_dataset")
            self.assertEqual(loaded["output"]["directory"], "other_output")
            recorded = configuration_for_manifest(loaded)
            self.assertEqual(recorded["answer_model"]["checkpoint"], "qwen")
            self.assertEqual(
                recorded["scoring_encoders"]["checkpoints"]["CLIP"], "CLIP.pt"
            )
            self.assertEqual(loaded["answer_model"]["checkpoint"], "models/qwen")

    def test_method_settings_and_unknown_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "configuration.yaml"
            for group, key, value in (
                ("visual_representation", "rho", 0.6),
                ("generation", "max_new_tokens", 65),
                ("region_selection", "unused_setting", True),
            ):
                configuration = deepcopy(fixture_configuration())
                configuration[group][key] = value
                path.write_text(yaml.safe_dump(configuration))
                with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                    load_configuration(path)

    def test_documented_cli_and_output_artifacts(self):
        command = load_entry_point()
        parser = command.build_argument_parser()
        arguments = parser.parse_args(
            [
                "--config",
                "configuration.yaml",
                "--data-root",
                "dataset",
                "--output-dir",
                "outputs/mme",
            ]
        )
        self.assertEqual(arguments.data_root, "dataset")
        sample = SimpleNamespace(
            question_id="synthetic",
            category="color",
            question="Which color?",
            ground_truth="B",
        )

        class EncoderStub:
            def __init__(self, *args, **kwargs):
                self.checkpoint_hashes = {
                    name: "synthetic_hash" for name in MODEL_ORDER
                }

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            configuration = fixture_configuration()
            configuration["output"]["directory"] = str(output)
            path = Path(directory) / "configuration.yaml"
            path.write_text(yaml.safe_dump(configuration))
            raw = "Evidence: A blue roof.\nFinal answer: B"
            prediction = {
                "question_id": sample.question_id,
                "category": sample.category,
                "question": sample.question,
                "ground_truth": "B",
                "raw_response": raw,
            }
            with patch.object(
                command, "load_mme_dataset", return_value=[sample]
            ), patch.object(command, "ImageTextEncoders", EncoderStub), patch.object(
                command, "file_sha256", return_value="synthetic_hash"
            ), patch.object(
                command, "prioritize_sample", return_value={}
            ), patch.object(
                command, "load_qwen3_vl", return_value=(object(), object())
            ), patch.object(
                command,
                "answer_prioritized_sample",
                return_value=(prediction, {"question_id": "synthetic"}),
            ), patch.dict(
                sys.modules,
                {
                    "torch": SimpleNamespace(
                        cuda=SimpleNamespace(is_available=lambda: False)
                    )
                },
            ):
                self.assertEqual(command.main(["--config", str(path)]), 0)
            self.assertEqual(
                {p.name for p in output.iterdir()},
                {
                    "predictions.jsonl",
                    "traces.jsonl",
                    "results.json",
                    "run_manifest.json",
                },
            )
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(
                manifest["scoring_models"]["CLIP"]["sha256"], "synthetic_hash"
            )
            self.assertEqual(manifest["generation"]["max_new_tokens"], 64)
            self.assertEqual(manifest["question_count"], 1)
            with self.assertRaises(FileExistsError):
                command.main(["--config", str(path)])


if __name__ == "__main__":
    unittest.main()
