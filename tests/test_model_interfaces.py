"""Stub-only Qwen and scorer interface checks; no model weights are loaded."""

import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from focusearth.models.qwen3_vl import build_processor_batch, generate_answer
from focusearth.region_prioritization.encoder_scoring import cosine_scores
from focusearth.visual_representation.region_protected_compression import (
    build_mapping_plan,
)


class FakeProcessor:
    image_token = "<|image_pad|>"

    class Tokenizer:
        @staticmethod
        def convert_tokens_to_ids(token):
            return 77

    tokenizer = Tokenizer()

    def apply_chat_template(self, message, **kwargs):
        assert message[0]["content"][0]["type"] == "image"
        assert message[0]["content"][1]["type"] == "text"
        return "rendered user image and prompt"

    def __call__(self, **kwargs):
        import torch

        return {
            "input_ids": torch.tensor([[1] + [77] * 16 + [2]]),
            "attention_mask": torch.ones((1, 18), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 256, 256)),
            "image_grid_thw": torch.tensor([[1, 8, 8]]),
        }

    def batch_decode(self, outputs, **kwargs):
        assert kwargs == {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
        return ["A"]


class FakeModel:
    def __init__(self):
        import torch

        self.model = SimpleNamespace(
            get_image_features=lambda *a, **k: None,
            get_rope_index=lambda *a, **k: None,
            rope_deltas=None,
        )
        self._parameter = torch.zeros(1)

    def parameters(self):
        return iter([self._parameter])

    def generate(self, input_ids, **kwargs):
        import torch

        generation = {
            key: kwargs.pop(key)
            for key in ("do_sample", "temperature", "max_new_tokens", "use_cache")
        }
        assert generation == {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": 64,
            "use_cache": True,
        }
        assert (
            "attention_mask" in kwargs
            and "pixel_values" in kwargs
            and "image_grid_thw" in kwargs
        )
        return torch.cat(
            (input_ids, torch.tensor([[3]], device=input_ids.device)), dim=1
        )


class ModelInterfaceTests(unittest.TestCase):
    def test_encoder_einsum_shape_and_values(self):
        image = np.ones((36, 5, 512), dtype=np.float32)
        text = np.ones((2, 512), dtype=np.float32)
        scores = cosine_scores(image, text)
        self.assertEqual(scores.shape, (2, 5, 36))
        np.testing.assert_array_equal(
            scores, np.full((2, 5, 36), 512, dtype=np.float32)
        )

    def test_processor_grid_and_generation_placeholder_contract(self):
        import torch

        processor = FakeProcessor()
        image = Image.new("RGB", (256, 256), "white")
        batch, grid = build_processor_batch(processor, image, "choose A")
        self.assertEqual(grid, [1, 8, 8])
        plan = build_mapping_plan(4, 4, [5], 8)
        self.assertTrue(plan["valid"], plan.get("failure"))
        answer, trace = generate_answer(FakeModel(), processor, batch, plan)
        self.assertEqual(answer, "A")
        self.assertEqual(trace["rendered_input_length"], 10)
        self.assertEqual(trace["generated_sequence_length"], 11)


if __name__ == "__main__":
    unittest.main()
