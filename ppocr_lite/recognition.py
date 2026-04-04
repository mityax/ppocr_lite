"""Text recognition using PP-OCR CTC models.

Pre-processing: PIL resize + numpy normalise.
Post-processing: CTC greedy decode from the ONNX model's embedded character list.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

class RecPreProcess:
    """Resize to fixed height (48) and normalise to [-1, 1].

    PP-OCRv4/v5 recognition models use rec_img_shape = [3, 48, 320].
    """

    HEIGHT = 48
    MAX_WIDTH = 320

    def resize_norm(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        """Resize *img* to (3, HEIGHT, img_w) and normalise to [-1, 1].

        *img_w* is determined by ``max_wh_ratio`` across the current batch
        (matching PaddleOCR's batching logic) but capped at MAX_WIDTH.

        Normalisation: (x/255 − 0.5)/0.5  ≡  x × (1/127.5) − 1
        """
        h, w = img.shape[:2]
        img_w    = min(int(self.HEIGHT * max_wh_ratio), self.MAX_WIDTH)
        resized_w = min(img_w, int(math.ceil(self.HEIGHT * w / h)))

        pil = Image.fromarray(img).resize((resized_w, self.HEIGHT), Image.BILINEAR)
        # np.asarray avoids an extra copy vs np.array when PIL returns a
        # read-only buffer; the subsequent arithmetic materialises a new array.
        arr = np.asarray(pil, dtype=np.float32) * (1.0 / 127.5) - 1.0  # HWC [-1,1]
        arr = arr.transpose(2, 0, 1)  # → CHW

        out = np.zeros((3, self.HEIGHT, img_w), dtype=np.float32)
        out[:, :, :resized_w] = arr
        return out


# ---------------------------------------------------------------------------
# CTC greedy decode
# ---------------------------------------------------------------------------

class CTCDecoder:
    """Greedy CTC decoder with blank-token removal and duplicate-collapse."""

    def __init__(self, characters: List[str]) -> None:
        # characters must NOT yet contain 'blank' or ' '; we prepend them
        self.chars = ["blank", *characters, " "]

    @classmethod
    def from_model_metadata(cls, meta: dict) -> "CTCDecoder":
        raw = meta.get("character", "")
        chars = raw.splitlines()
        return cls(chars)

    @classmethod
    def from_file(cls, path: Path) -> "CTCDecoder":
        chars = path.read_bytes().decode("utf-8").splitlines()
        return cls(chars)

    def decode(
        self, preds: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Decode a batch of CTC outputs.

        Parameters
        ----------
        preds:
            Shape (N, T, C) float32.

        Returns
        -------
        List of (text, mean_confidence) pairs.
        """
        indices = preds.argmax(axis=2)   # (N, T)
        probs   = preds.max(axis=2)      # (N, T)
        results = []
        n_chars = len(self.chars)

        for idx_seq, prob_seq in zip(indices, probs):
            chars: List[str] = []
            confs: List[float] = []
            prev = 0  # blank index
            for tok, p in zip(idx_seq.tolist(), prob_seq.tolist()):
                if tok == 0:          # blank — resets duplicate suppression
                    prev = 0
                    continue
                if tok == prev:       # consecutive duplicate → collapse
                    continue
                prev = tok
                if tok < n_chars:
                    chars.append(self.chars[tok])
                    confs.append(p)
            text  = "".join(chars)
            score = sum(confs) / len(confs) if confs else 0.0
            results.append((text, round(score, 5)))

        return results
