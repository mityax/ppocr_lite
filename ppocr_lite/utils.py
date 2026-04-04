"""Miscellaneous utilities: image I/O, region cropping, box sorting."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import List, Tuple, Union

import numpy as np
from PIL import Image

ImageInput = Union[str, Path, np.ndarray, "Image.Image"]


def load_image(src: ImageInput) -> np.ndarray:
    """Return an H×W×3 uint8 numpy array (RGB) from any supported input."""
    if isinstance(src, np.ndarray):
        if src.ndim == 2:
            src = np.stack([src] * 3, axis=-1)
        elif src.shape[2] == 4:
            src = src[:, :, :3]
        return src
    if isinstance(src, Image.Image):
        return np.array(src.convert("RGB"), dtype=np.uint8)
    pil = Image.open(src).convert("RGB")
    return np.array(pil, dtype=np.uint8)


def crop_region(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Crop the quadrilateral *box* (4×2 int, TL→TR→BR→BL) from *img*.

    For screenshots the boxes are almost always axis-aligned, so a
    simple bounding-rect crop is used.  A perspective warp would add a
    cv2 dependency for essentially zero benefit in this use-case.
    """
    x0 = int(np.clip(box[:, 0].min(), 0, img.shape[1] - 1))
    y0 = int(np.clip(box[:, 1].min(), 0, img.shape[0] - 1))
    x1 = int(np.clip(box[:, 0].max(), 0, img.shape[1]))
    y1 = int(np.clip(box[:, 1].max(), 0, img.shape[0]))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        # fall back to 1×1 black patch rather than empty array
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return crop


def sort_boxes(boxes: np.ndarray, y_thresh: int = 10) -> np.ndarray:
    """Sort detected boxes reading-order: top-to-bottom, left-to-right.

    Two boxes belong to the same "line" when their top-left y coordinates
    differ by less than *y_thresh* pixels.
    """

    if len(boxes) == 0:
        return boxes

    y_coords = boxes[:, 0, 1]
    y_order  = np.argsort(y_coords, kind="stable")
    boxes_s  = boxes[y_order]
    y_s      = y_coords[y_order]

    dy = np.diff(y_s)
    line_ids = np.concatenate([[0], np.cumsum(dy >= y_thresh)])

    x_coords = boxes_s[:, 0, 0]
    final    = np.lexsort((x_coords, line_ids))
    return boxes_s[final]


class FastONNXRunner:
    """Thin wrapper around an ONNX InferenceSession.

    Only calls ``np.ascontiguousarray`` when the tensor is actually
    non-contiguous — avoids a silent memory copy on the hot path.
    """

    def __init__(self, session) -> None:
        self.session     = session
        self.input_name  = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        if not input_tensor.flags["C_CONTIGUOUS"]:
            input_tensor = np.ascontiguousarray(input_tensor)
        return self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )[0]

_local = threading.local()
log = logging.getLogger(__name__)

@contextmanager
def log_perf(label: str, *, warn_above: float = 0.0, level=logging.DEBUG):
    depth = getattr(_local, "perf_depth", 0)
    _local.perf_depth = depth + 1
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        _local.perf_depth = depth
        if elapsed >= warn_above:
            indent = "    " * depth
            log.log(level, "%s%s took %.3f s", indent, label, elapsed)
