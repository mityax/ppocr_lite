from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

from ppocr_lite.utils import log_perf


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

class DetPreProcess:
    """Resize → normalize → NCHW float32 batch for the DB detector.

    Normalisation: (x/255 − 0.5) / 0.5  ≡  x × (1/127.5) − 1

    The result is made C-contiguous here so FastONNXRunner never needs
    an extra copy.
    """

    def __init__(self, limit_side_len: int = 960, limit_type: str = "max") -> None:
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        ratio = self._ratio(h, w)
        new_h = int(round(h * ratio / 32) * 32)
        new_w = int(round(w * ratio / 32) * 32)
        if new_h <= 0 or new_w <= 0:
            raise ValueError(f"Invalid resize target ({new_w}×{new_h})")

        pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
        # Fused normalise: (x/255 − 0.5)/0.5  ==  x/127.5 − 1
        arr = np.asarray(pil, dtype=np.float32) * (1.0 / 127.5) - 1.0
        # Transpose HWC→CHW, add batch dim, and force C-contiguous in one shot
        return np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis])

    def _ratio(self, h: int, w: int) -> float:
        lim = self.limit_side_len
        side = min(h, w) if self.limit_type == "min" else max(h, w)
        if self.limit_type == "min" and side < lim:
            return lim / side
        if self.limit_type == "max" and side > lim:
            return lim / side
        return 1.0


def _auto_limit(max_side: int) -> int:
    """Longest-side cap for the DB detector input (used with limit_type='max').

    Keeps detector tensors at a manageable size without upscaling.
    The engine chooses limit_type='min' separately for small images.
    """
    if max_side <= 1280:
        return 1280
    if max_side <= 1920:
        return 1920
    return 2560


# ---------------------------------------------------------------------------
# Post-processing  (DB – Differentiable Binarisation)
# ---------------------------------------------------------------------------

class DBPostProcess:
    """Convert the DB probability map into oriented quad boxes.

    Fast path optimized for screenshot / UI text:
    - connected-component slices instead of repeated full-image scans
    - axis-aligned bounding rectangles (O(N) min/max instead of PCA)
    - optional SciPy acceleration if available
    """

    def __init__(
        self,
        thresh: float = 0.3,
        box_thresh: float = 0.5,
        max_candidates: int = 1000,
        unclip_ratio: float = 1.6,
        min_size: int = 3,
        max_points_for_box: int = 512,
    ) -> None:
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size
        self.max_points_for_box = max_points_for_box

    def __call__(
        self,
        pred: np.ndarray,            # (1, 1, H, W) float32
        orig_shape: Tuple[int, int], # (orig_h, orig_w)
    ) -> Tuple[np.ndarray, List[float]]:
        prob = pred[0, 0]  # (H, W)
        mask = prob > self.thresh

        # slight dilation equivalent: max-pool with 2×2 kernel
        mask = _dilate2x2_bool(mask)

        orig_h, orig_w = orig_shape
        map_h, map_w = mask.shape

        components = _find_components(mask)

        boxes: List[np.ndarray] = []
        scores: List[float] = []

        for comp in components[:self.max_candidates]:
            pts = comp["pts"]  # global coords (N, 2) float32
            if pts.shape[0] < 4:
                continue

            if pts.shape[0] > self.max_points_for_box:
                step = max(1, pts.shape[0] // self.max_points_for_box)
                pts = pts[::step]

            # Axis-aligned bounding rect: exact for horizontal screen text and
            # ~5× faster than PCA (no eigendecomposition).
            # Swap to _pca_rect_quad if you need rotated-text support.
            box, sside = _axis_aligned_rect(pts)
            if sside < self.min_size:
                continue

            score = _box_score_fast(prob, box)
            if score < self.box_thresh:
                continue

            box = _unclip_quad(box, self.unclip_ratio)
            box, sside = _axis_aligned_rect(box)
            if sside < self.min_size + 2:
                continue

            # scale back to original image coordinates
            box[:, 0] = np.clip(np.round(box[:, 0] / map_w * orig_w), 0, orig_w - 1)
            box[:, 1] = np.clip(np.round(box[:, 1] / map_h * orig_h), 0, orig_h - 1)

            boxes.append(box.astype(np.int32))
            scores.append(float(score))

        if not boxes:
            return np.empty((0, 4, 2), dtype=np.int32), []

        boxes_arr = np.stack(boxes, axis=0)
        boxes_arr, scores = _filter_boxes(boxes_arr, scores, orig_h, orig_w)
        return boxes_arr, scores


# ---------------------------------------------------------------------------
# Pure-numpy image processing primitives (cv2 replacements)
# ---------------------------------------------------------------------------

def _dilate2x2_bool(mask: np.ndarray) -> np.ndarray:
    """Equivalent of cv2.dilate with a 2×2 all-ones kernel."""
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[:, :-1] |= mask[:, 1:]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


try:
    import scipy.ndimage as scipy_ndimage
except ImportError:
    scipy_ndimage = None

def _find_components(mask: np.ndarray):
    """Return connected components with point clouds in global coords."""
    if scipy_ndimage is not None:
        with log_perf("scipy_ndimage_label.nd_label"):
            labeled, n = scipy_ndimage.label(mask)
        with log_perf("scipy_ndimage_label.find_objects"):
            objs = scipy_ndimage.find_objects(labeled)
            return _components_from_labeled(labeled, objs, n)
    else:
        with log_perf("_label_numpy"):
            labeled, n = _label_numpy(mask)
        with log_perf("_find_objects_numpy"):
            objs = _find_objects_numpy(labeled, n)
        return _components_from_labeled(labeled, objs, n)


def _components_from_labeled(labeled: np.ndarray, objs, n: int):
    comps = []
    for i in range(1, n + 1):
        sl = objs[i - 1] if i - 1 < len(objs) else None
        if sl is None:
            continue
        ysl, xsl = sl
        roi = labeled[ysl, xsl] == i
        if roi.sum() < 4:
            continue
        ys, xs = np.nonzero(roi)
        xs = xs + xsl.start
        ys = ys + ysl.start
        pts = np.empty((xs.size, 2), dtype=np.float32)
        pts[:, 0] = xs
        pts[:, 1] = ys
        comps.append({"label": i, "slice": sl, "pts": pts})
    return comps


def _find_objects_numpy(labeled: np.ndarray, n: int):
    """Cheap find_objects replacement for fallback labeling."""
    objs = [None] * n
    if n == 0:
        return objs
    ys, xs = np.nonzero(labeled)
    vals = labeled[ys, xs]
    mins_y = np.full(n + 1, labeled.shape[0], dtype=np.int32)
    mins_x = np.full(n + 1, labeled.shape[1], dtype=np.int32)
    maxs_y = np.full(n + 1, -1, dtype=np.int32)
    maxs_x = np.full(n + 1, -1, dtype=np.int32)
    np.minimum.at(mins_y, vals, ys)
    np.minimum.at(mins_x, vals, xs)
    np.maximum.at(maxs_y, vals, ys)
    np.maximum.at(maxs_x, vals, xs)
    for i in range(1, n + 1):
        if maxs_y[i] >= 0:
            objs[i - 1] = (
                slice(int(mins_y[i]), int(maxs_y[i]) + 1),
                slice(int(mins_x[i]), int(maxs_x[i]) + 1),
            )
    return objs

def downscale(mask, factor=2):
    """Downscale a 2D binary mask by integer factor using block reduction."""
    h, w = mask.shape
    new_h, new_w = h // factor, w // factor
    # reshape into blocks and take max (if any pixel is 1, block is 1)
    down = mask[:new_h*factor, :new_w*factor].reshape(new_h, factor, new_w, factor)
    down = down.max(axis=(1, 3))
    return down

def upscale(labels_small, factor=2, original_shape=None):
    """Upscale labels back using nearest-neighbor replication."""
    up = np.repeat(np.repeat(labels_small, factor, axis=0), factor, axis=1)
    if original_shape:
        up = up[:original_shape[0], :original_shape[1]]
    return up

def label_numpy(mask):
    """Minimal 4-connectivity labeling."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            label += 1
            stack = [(y, x)]
            labels[y, x] = label
            while stack:
                cy, cx = stack.pop()
                if cy > 0 and mask[cy - 1, cx] and labels[cy - 1, cx] == 0:
                    labels[cy - 1, cx] = label
                    stack.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and labels[cy + 1, cx] == 0:
                    labels[cy + 1, cx] = label
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and labels[cy, cx - 1] == 0:
                    labels[cy, cx - 1] = label
                    stack.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and labels[cy, cx + 1] == 0:
                    labels[cy, cx + 1] = label
                    stack.append((cy, cx + 1))
    return labels, label

def _label_numpy(mask, factor=2):
    """Downscale → label → upscale"""
    small = downscale(mask, factor=factor)
    labels_small, n = label_numpy(small)
    labels_up = upscale(labels_small, factor=factor, original_shape=mask.shape)
    return labels_up, n

'''
def _label_numpy(mask: np.ndarray):
    """Minimal 4-connectivity connected-component labeling fallback."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            label += 1
            stack = [(y, x)]
            labels[y, x] = label
            while stack:
                cy, cx = stack.pop()
                if cy > 0 and mask[cy - 1, cx] and labels[cy - 1, cx] == 0:
                    labels[cy - 1, cx] = label
                    stack.append((cy - 1, cx))
                if cy + 1 < h and mask[cy + 1, cx] and labels[cy + 1, cx] == 0:
                    labels[cy + 1, cx] = label
                    stack.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and labels[cy, cx - 1] == 0:
                    labels[cy, cx - 1] = label
                    stack.append((cy, cx - 1))
                if cx + 1 < w and mask[cy, cx + 1] and labels[cy, cx + 1] == 0:
                    labels[cy, cx + 1] = label
                    stack.append((cy, cx + 1))
    return labels, label
'''

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _axis_aligned_rect(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fast axis-aligned bounding rectangle.

    Exact for horizontal screen / UI text and ~5× faster than _pca_rect_quad
    because it only needs min/max instead of a covariance eigendecomposition.

    Returns:
        box: (4, 2) float32 quad in TL→TR→BR→BL order
        short_side: float
    """
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    box = np.array(
        [[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]],
        dtype=np.float32,
    )
    return box, float(min(mx[0] - mn[0], mx[1] - mn[1]))


def _pca_rect_quad(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    """PCA-based oriented rectangle — retained for rotated-text use-cases.

    Use _axis_aligned_rect for screenshots (faster, equally accurate there).

    Returns:
        box: (4, 2) float32 quad in TL→TR→BR→BL order
        short_side: float
    """
    pts = pts.astype(np.float32, copy=False)
    if pts.shape[0] == 1:
        x, y = pts[0]
        return np.array([[x, y], [x+1, y], [x+1, y+1], [x, y+1]], dtype=np.float32), 1.0

    center = pts.mean(axis=0)
    centered = pts - center
    cov = centered.T @ centered / max(len(pts), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    proj = centered @ eigvecs
    mn = proj.min(axis=0)
    mx = proj.max(axis=0)
    corners = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]], dtype=np.float32)
    box = _order_quad(corners @ eigvecs.T + center)
    w = float(np.linalg.norm(box[0] - box[1]))
    h = float(np.linalg.norm(box[1] - box[2]))
    return box, min(w, h)


def _order_quad(box: np.ndarray) -> np.ndarray:
    """Sort 4 points into TL, TR, BR, BL order."""
    box = np.asarray(box, dtype=np.float32)
    x_sort = box[np.argsort(box[:, 0])]
    left = x_sort[:2][np.argsort(x_sort[:2, 1])]
    right = x_sort[2:][np.argsort(x_sort[2:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], dtype=np.float32)


def _box_score_fast(prob: np.ndarray, box: np.ndarray) -> float:
    """Fast bounding-box mean score."""
    h, w = prob.shape
    xs = np.clip(box[:, 0].astype(np.int32), 0, w - 1)
    ys = np.clip(box[:, 1].astype(np.int32), 0, h - 1)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(prob[y0:y1 + 1, x0:x1 + 1].mean())


def _unclip_quad(box: np.ndarray, ratio: float) -> np.ndarray:
    """Expand a quad outward by *ratio* using polygon area/perimeter."""
    box = np.asarray(box, dtype=np.float32)
    x, y = box[:, 0], box[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    perimeter = float(np.sum(np.linalg.norm(np.diff(box, axis=0, append=box[:1]), axis=1)))
    if perimeter < 1e-6:
        return box
    distance = area * ratio / perimeter
    center = box.mean(axis=0)
    vecs = box - center
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-6)
    return (box + vecs / norms * distance).astype(np.float32)


def _filter_boxes(
    boxes: np.ndarray, scores: List[float], img_h: int, img_w: int
) -> Tuple[np.ndarray, List[float]]:
    kept_boxes, kept_scores = [], []
    for box, score in zip(boxes, scores):
        box = _order_quad(box)
        box[:, 0] = np.clip(box[:, 0], 0, img_w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, img_h - 1)
        w = float(np.linalg.norm(box[0] - box[1]))
        h = float(np.linalg.norm(box[1] - box[2]))
        if w <= 3 or h <= 3:
            continue
        kept_boxes.append(box.astype(np.int32))
        kept_scores.append(float(score))
    if not kept_boxes:
        return np.empty((0, 4, 2), dtype=np.int32), []
    return np.stack(kept_boxes, axis=0), kept_scores
