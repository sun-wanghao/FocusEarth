"""Native ViT-B/32 preprocessing and five-view cosine scoring.

Checkpoints are supplied by the caller. Loading uses pickle-capable torch APIs,
as required by the original checkpoints; use only trusted checkpoint files.
The candidate axis is row-major and the view axis follows ``VIEW_ORDER``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io_utils import file_sha256
from ..region_prioritization.encoder_scoring import candidate_views, cosine_scores

MODEL_ORDER = ("CLIP", "GeoRSCLIP", "RemoteCLIP", "SkyCLIP-50")

CHECKPOINT_HASHES = dict(
    zip(
        MODEL_ORDER,
        (
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af",
            "129bafaa6a097b8be52e2babf27d24f0a934dae919201e538dc698611bd1ea01",
            "60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4",
            "69ba9fa7089322a83e9164fe515a222bcece2a715888db5749ab5ce4d1602313",
        ),
    )
)


class ImageTextEncoders:
    """Load and execute all four frozen scorers without a precomputed bank.

    Models are loaded lazily and cached across questions. FP32 encoding uses
    batches of 128 image views by default and 256 text descriptions.
    """

    def __init__(
        self,
        checkpoints: dict[str, str],
        device: str = "cuda",
        batch_size: int = 128,
        verify_hashes: bool = True,
    ):
        if set(checkpoints) != set(MODEL_ORDER):
            raise ValueError(f"checkpoint keys must be {MODEL_ORDER}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.checkpoints = {
            key: str(Path(value).resolve()) for key, value in checkpoints.items()
        }
        self.device, self.batch_size = device, batch_size
        self._models: dict[str, tuple[Any, Any, Any]] = {}
        self.checkpoint_hashes = {}
        for key, path in self.checkpoints.items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"missing {key} checkpoint: {path}")
            value = file_sha256(path)
            if verify_hashes and value != CHECKPOINT_HASHES[key]:
                raise ValueError(
                    f"{key} checkpoint identity differs from the reference checkpoint"
                )
            self.checkpoint_hashes[key] = value

    def _load(self, kind: str) -> tuple[Any, Any, Any]:
        import open_clip
        import torch

        if kind in self._models:
            return self._models[kind]
        if kind == "CLIP":
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained=self.checkpoints[kind],
                weights_only=False,
                force_quick_gelu=True,
            )
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=None
            )
            state = torch.load(
                self.checkpoints[kind], map_location="cpu", weights_only=False
            )
            state = state.get("state_dict", state) if isinstance(state, dict) else state
            state = {
                str(k)[7:] if str(k).startswith("module.") else str(k): v
                for k, v in state.items()
            }
            model.load_state_dict(state, strict=True)
        result = (
            model.eval().to(self.device),
            preprocess,
            open_clip.get_tokenizer("ViT-B-32"),
        )
        self._models[kind] = result
        return result

    def score(self, image, descriptions: list[str]) -> np.ndarray:
        """Compute [description, encoder=4, view=5, candidate=36] raw scores."""
        import torch

        if not descriptions or any(
            not isinstance(x, str) or not x.strip() for x in descriptions
        ):
            raise ValueError("nonempty descriptions are required")
        outputs = []
        for kind in MODEL_ORDER:
            model, preprocess, tokenizer = self._load(kind)
            image_rows = []
            pending = []
            with torch.inference_mode():
                for crop in candidate_views(image):
                    try:
                        pending.append(preprocess(crop))
                    finally:
                        crop.close()
                    if len(pending) == self.batch_size:
                        encoded = model.encode_image(
                            torch.stack(pending).to(self.device)
                        )
                        encoded = encoded / encoded.norm(
                            dim=-1, keepdim=True
                        ).clamp_min(1e-12)
                        image_rows.append(encoded.float().cpu().numpy())
                        pending.clear()
                if pending:
                    encoded = model.encode_image(torch.stack(pending).to(self.device))
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(
                        1e-12
                    )
                    image_rows.append(encoded.float().cpu().numpy())
                text_rows = []
                for start in range(0, len(descriptions), 256):
                    tokens = tokenizer(descriptions[start : start + 256]).to(
                        self.device
                    )
                    encoded = model.encode_text(tokens)
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(
                        1e-12
                    )
                    text_rows.append(encoded.float().cpu().numpy())
            outputs.append(
                cosine_scores(
                    np.concatenate(image_rows).reshape(36, 5, 512),
                    np.concatenate(text_rows),
                )
            )
        return np.stack(outputs, axis=1)

    def close(self) -> None:
        """Release cached encoder models after region scoring."""
        self._models.clear()
