"""Five-view crop order and normalized image-text cosine scoring."""

from __future__ import annotations

import numpy as np

from .candidate_regions import VIEW_ORDER, candidate_boxes, view_box


def cosine_scores(image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    """Return FP32 [description, view, candidate] scores from normalized features.

    Normalized embeddings use NumPy einsum with FP32 accumulation.
    """
    image = np.asarray(image_features, dtype=np.float32)
    text = np.asarray(text_features, dtype=np.float32)
    if image.shape != (36, 5, 512) or text.ndim != 2 or text.shape[1] != 512:
        raise ValueError("expected image [36,5,512] and text [descriptions,512]")
    if not np.isfinite(image).all() or not np.isfinite(text).all():
        raise ValueError("embeddings must be finite")
    return np.einsum("td,pvd->tvp", text, image, dtype=np.float32)


def candidate_views(image):
    """Yield source crops in candidate-major, then five-view order.

    Cropping precedes the encoder's native resize/center-crop transform. No
    intermediate candidate resize is performed before the corner crops.
    The caller owns and must close each yielded PIL image.
    """
    for box in candidate_boxes(*image.size):
        for view in VIEW_ORDER:
            yield image.crop(view_box(box, view))
