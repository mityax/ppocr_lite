"""PPOCRLite – the main OCR engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .classification import ClsPreProcess, apply_cls
from .detection import DBPostProcess, DetPreProcess, _auto_limit
from .models import ModelConfig
from .recognition import CTCDecoder, RecPreProcess
from .utils import ImageInput, crop_region, load_image, sort_boxes, log_perf, FastONNXRunner


@dataclass
class OCRResult:
    """Single detected text region."""
    text: str
    score: float
    box: np.ndarray  # (4, 2) int32  TL→TR→BR→BL


class PPOCRLite:
    """Lightweight PP-OCR runtime.

    Parameters
    ----------
    config:
        ``ModelConfig`` controlling which ONNX files to use.
        ``None`` → default PP-OCRv5 mobile models (auto-downloaded).
    det_thresh:
        DB binarisation threshold (lower ↔ more detections).
    det_box_thresh:
        Minimum mean probability inside a detected box.
    det_unclip_ratio:
        How much to expand detected boxes outward.
    rec_batch_size:
        Number of text crops processed in one ONNX call.  Larger values
        reduce kernel-launch overhead; 24 is a good default.
    use_cls:
        Whether to run the direction classifier.  ``None`` = use classifier
        only if *config* provides / downloads one.
    providers:
        ONNX Runtime execution providers, e.g. ``["CUDAExecutionProvider",
        "CPUExecutionProvider"]``.  ``None`` → auto-select.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        det_thresh: float = 0.3,
        det_box_thresh: float = 0.5,
        det_unclip_ratio: float = 1.6,
        rec_batch_size: int = 24,          # was 6 — fewer ONNX kernel launches
        use_cls: Optional[bool] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        cfg = (config or ModelConfig()).resolve()
        self._providers = providers or ["CPUExecutionProvider"]

        # --- Detection ---
        self._det_post = DBPostProcess(
            thresh=det_thresh,
            box_thresh=det_box_thresh,
            unclip_ratio=det_unclip_ratio,
        )
        self._det_session = FastONNXRunner(_load_session(cfg.det_model, self._providers))

        # --- Direction classifier ---
        self._cls_session = None
        self._cls_pre = None
        _want_cls = use_cls if use_cls is not None else (cfg.cls_model is not False)
        if _want_cls and cfg.cls_model and cfg.cls_model is not False:
            self._cls_session = _load_session(cfg.cls_model, self._providers)
            self._cls_pre = ClsPreProcess()

        # --- Recognition ---
        self._rec_pre     = RecPreProcess()
        self._rec_batch   = rec_batch_size
        self._rec_session = FastONNXRunner(_load_session(cfg.rec_model, self._providers))
        try:
            self._decoder = _build_decoder(self._rec_session.session)
        except RuntimeError:
            self._decoder = CTCDecoder.from_file(config.dict_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        image: ImageInput,
        return_boxes: bool = True,
    ) -> List[OCRResult]:
        """Run end-to-end OCR on *image*.

        Parameters
        ----------
        image:
            File path, ``PIL.Image``, or H×W×3 uint8 numpy array.
        return_boxes:
            When ``False`` only text and scores are returned (boxes are
            still computed but not included in the output).

        Returns
        -------
        List of :class:`OCRResult`, one per detected text region,
        in reading order (top-to-bottom, left-to-right).
        """
        img = load_image(image)
        with log_perf("_detect"):
            boxes = self._detect(img)
        if len(boxes) == 0:
            return []

        with log_perf("sort_boxes"):
            boxes = sort_boxes(boxes)

        with log_perf("crop_region[s]"):
            crops = [crop_region(img, box) for box in boxes]

        if self._cls_session is not None:
            crops = self._classify(crops)

        with log_perf("_recognize"):
            texts, scores = self._recognize(crops)

        return [
            OCRResult(text=t, score=s, box=b)
            for t, s, b in zip(texts, scores, boxes)
            if t.strip()
        ]

    def find_text_close_to(self, image: ImageInput, positions: list[tuple[float, float]], max_dist: float = 0.4):
        img = load_image(image)

        boxes = self._detect(img)

        if len(boxes) == 0:
            return []

        boxes = sort_boxes(boxes)

        boxes = filter_boxes_and_sort_by_position(
            boxes,
            tuple((t[0] * img.shape[1], t[1] * img.shape[0]) for t in positions),
            max_dist=max_dist * np.min(img.shape[:2])
        )

        crops = [crop_region(img, box) for box in boxes]

        if self._cls_session is not None:
            crops = self._classify(crops)

        texts, scores = self._recognize(crops)

        return [
            OCRResult(text=t, score=s, box=b)
            for t, s, b in zip(texts, scores, boxes)
            if t.strip()
        ]


    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]

        # --- Choose detection input size --------------------------------
        # Original code used limit_type="min", which *upscales* the image so
        # the SHORT side reaches `limit`.  On a 1920×1080 screenshot with
        # limit=2000 that blows the tensor up to ~3 550×1 984 — 3.4× more
        # pixels than necessary and the single biggest runtime cost.
        #
        # New policy:
        #   • Tiny images (short side < 960): upscale so the short side
        #     reaches 960 — keeps small captures readable for the detector.
        #   • Everything else: cap the LONG side (no upscaling) so large
        #     screenshots are fed at roughly native resolution.
        if min(h, w) < 960:
            pre = DetPreProcess(limit_side_len=960, limit_type="min")
        else:
            pre = DetPreProcess(limit_side_len=_auto_limit(max(h, w)), limit_type="max")

        with log_perf("pre"):
            inp = pre(img)

        with log_perf("_det_session.run"):
            pred = self._det_session(inp)

        with log_perf("_det_post"):
            boxes, _ = self._det_post(pred, (h, w))

        return boxes  # (N, 4, 2)

    def _classify(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        batch = self._cls_pre(crops)
        preds = self._cls_session.run(
            None, {self._cls_session.get_inputs()[0].name: batch}
        )[0]  # (N, 2)
        return apply_cls(crops, preds)

    def _recognize(
        self, crops: List[np.ndarray]
    ) -> Tuple[List[str], List[float]]:
        if not crops:
            return [], []

        # Sort by aspect ratio so same-shape images cluster → minimal padding
        ratios  = np.array([c.shape[1] / c.shape[0] for c in crops], dtype=np.float32)
        indices = np.argsort(ratios).tolist()

        results: List[Tuple[str, float]] = [("", 0.0)] * len(crops)

        for start in range(0, len(crops), self._rec_batch):
            batch_idx   = indices[start : start + self._rec_batch]
            batch_crops = [crops[i] for i in batch_idx]

            max_ratio = float(ratios[batch_idx].max())
            # resize_norm already returns float32 — no .astype() copy needed
            tensors = np.stack(
                [self._rec_pre.resize_norm(c, max_ratio) for c in batch_crops]
            )  # (B, 3, 48, W) float32

            preds = self._rec_session(tensors)  # (B, T, C)

            for local_i, decoded in enumerate(self._decoder.decode(preds)):
                results[batch_idx[local_i]] = decoded

        texts  = [r[0] for r in results]
        scores = [r[1] for r in results]
        return texts, scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_session(model_path: Path, providers: List[str]):
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
    import os

    opts = SessionOptions()
    opts.log_severity_level       = 4   # errors only
    opts.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena     = True
    n = os.cpu_count() or 1
    opts.intra_op_num_threads     = n
    opts.inter_op_num_threads     = 1
    return InferenceSession(str(model_path), sess_options=opts, providers=providers)


def _build_decoder(session) -> CTCDecoder:
    meta = session.get_modelmeta().custom_metadata_map
    if "character" in meta:
        return CTCDecoder.from_model_metadata(meta)
    raise RuntimeError(
        "Recognition model has no embedded character list. "
        "Please use an ONNX model exported by RapidAI/RapidOCR "
        "(PP-OCRv4 or PP-OCRv5)."
    )


# ---------------------------------------------------------------------------
# Find text helpers
# ---------------------------------------------------------------------------

def filter_boxes_and_sort_by_position(
        boxes: np.ndarray,
        positions: Tuple[Tuple[float, float], ...],
        max_dist: int
) -> List[OCRResult]:
    def box_center(box: np.ndarray) -> np.ndarray:
        # Compute the center of the box
        return box.mean(axis=0)

    filtered = []

    for box in boxes:
        center = box_center(box)
        # Compute distances to all positions
        dists = np.linalg.norm(np.array(positions) - center, axis=1)
        min_dist = dists.min()
        if min_dist <= max_dist:
            filtered.append((min_dist, box))

    # Sort by distance to the closest position
    filtered.sort(key=lambda x: x[0])

    # Return only the OCRResult objects
    return [b for _, b in filtered]
