"""Text direction classifier (0° vs 180°).

Used to flip upside-down text regions before recognition.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from PIL import Image


class ClsPreProcess:
    """Resize to (C, 48, 192) and normalise to [-1, 1].

    Normalisation: (x/255 − 0.5)/0.5  ≡  x × (1/127.5) − 1
    """

    HEIGHT = 48
    WIDTH  = 192

    def __call__(self, imgs: List[np.ndarray]) -> np.ndarray:
        batch = [self._process(img) for img in imgs]
        return np.stack(batch, axis=0).astype(np.float32)  # (N, 3, 48, W)

    def _process(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        target_w = min(self.WIDTH, int(math.ceil(self.HEIGHT * w / h)))
        pil = Image.fromarray(img).resize((target_w, self.HEIGHT), Image.BILINEAR)
        # Fused normalise: (x/255 − 0.5)/0.5  ==  x/127.5 − 1
        arr = np.asarray(pil, dtype=np.float32) * (1.0 / 127.5) - 1.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        pad = np.zeros((3, self.HEIGHT, self.WIDTH), dtype=np.float32)
        pad[:, :, :target_w] = arr
        return pad


def apply_cls(
    imgs: List[np.ndarray],
    preds: np.ndarray,
    thresh: float = 0.9,
) -> List[np.ndarray]:
    """Rotate images predicted as 180° back to upright.

    Parameters
    ----------
    imgs:
        Cropped text-region images (H×W×3 uint8).
    preds:
        Classifier output, shape (N, 2).  Class 0 = 0°, class 1 = 180°.
    thresh:
        Confidence threshold; below this the image is left unchanged.
    """
    out = []
    for img, pred in zip(imgs, preds):
        label = int(pred.argmax())
        score = float(pred[label])
        if label == 1 and score >= thresh:
            # rotate 180°
            img = img[::-1, ::-1, :]
        out.append(img)
    return out
