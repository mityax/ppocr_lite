"""PPOCRLite – the main OCR engine."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .classification import ClsPreProcess, apply_cls
from .detection import DBPostProcess, DetPreProcess, _auto_limit
from .models import ModelConfig
from .recognition import CTCDecoder, RecPreProcess
from .structs import OCRResult, BBox
from .text_handling import arrange_text, merge_phrase_boxes, merge_phrase_boxes_fuzzy
from .utils import ImageInput, crop_region, load_image, log_perf, FastONNXRunner


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
            self._decoder = CTCDecoder.from_file(cfg.dict_path)

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

    def find_text_close_to(self, image: ImageInput, positions: list[tuple[float, float]], max_dist: float = 0.4,
                           early_exit_if_all_found: list[str] | None = None):
        img = load_image(image)

        boxes = self._detect(img)

        if len(boxes) == 0:
            return []

        boxes = filter_boxes_and_sort_by_proximity(
            boxes,
            tuple((t[0] * img.shape[1], t[1] * img.shape[0]) for t in positions),
            max_dist=max_dist * np.min(img.shape[:2])
        )

        crops = [crop_region(img, box) for box in boxes]

        if early_exit_if_all_found:
            # Use a smaller batch size of three to support early exit:
            texts, scores = [], []
            for start in range(0, len(crops), 3):
                batch = crops[start:start+3]

                if self._cls_session is not None:
                    batch = self._classify(batch)

                txts, scrs = self._recognize(batch)

                texts.extend(txts)
                scores.extend(scrs)

                if all(w in texts for w in early_exit_if_all_found):
                    # All words found; no need to recognize further
                    break
        else:
            if self._cls_session is not None:
                crops = self._classify(crops)
            texts, scores = self._recognize(crops)

        return [
            OCRResult(text=t, score=s, box=b)
            for t, s, b in zip(texts, scores, boxes)
            if t.strip()
        ]

    def check_contains(self, image: ImageInput, phrases: list[str],
                       position_hints: list[tuple[float, float]] | None = None, position_max_dist: float | None = None,
                       fuzzy_match_min_similarity: float = 0.9,
                       line_cy_threshold: float = 0.2, word_dist_threshold: float = 5.0) -> list[OCRResult | None]:
        """
        For each phrase in [phrases], check whether the phrase is found in the given image.

        Parameters
        ----------
        phrases:
            A list of phrases to search for (newlines in a phrase are not supported).
        position_hints:
            Helps to search more efficiently by recognizing boxes close to a position hint first. Given in
            coordinates relative to the image size [0-1].
        position_max_dist:
            If given, boxes whose center is further away from any location hint than [location_max_dist]
            will be ignored. This value is relative to the shorter image side.
        fuzzy_match_min_similarity:
            If greater than 0, fall back to fuzzy string matching if an exact match is not found. Can be
            set to a value between 0 and 1, indicating the allowed mismatch ratio (as interpreted by
            difflib.SequenceMatcher).
        line_cy_threshold:
            The maximum allowed vertical distance between word center points in one line, relative to line
            height.
        word_dist_threshold:
            The maximum horizontal distance between words to be considered part of the same line, relative
            to line height (as a proxy for font size).

        Returns
        -------
        A list with one item for each phrase, that is either an OCRResult or None, if no match was found.
        """

        img = load_image(image)
        res = [None for _ in phrases]

        with log_perf("_detect"):
            boxes = self._detect(img)

        if len(boxes) == 0:
            return res

        if position_hints is not None:
            boxes = filter_boxes_and_sort_by_proximity(
                boxes,
                tuple((t[0] * img.shape[1], t[1] * img.shape[0]) for t in position_hints),
                max_dist=(position_max_dist * np.min(img.shape[:2])) if position_max_dist else np.max(img.shape[:2])
            )
        #else:
        #    boxes = sorted(boxes, key=lambda b: min(abs(b.width / b.height - len(p) * x) for p in phrases))

        crops = [crop_region(img, box) for box in boxes]

        # Use a smaller batch size of three to support early exit:
        ocr_results = []

        with log_perf("_recognize and matching loop"):
            for start in range(0, len(crops), 6):
                batch = crops[start:start + 6]

                if self._cls_session is not None:
                    batch = self._classify(batch)

                with log_perf("_recognize"):
                    txts, scrs = self._recognize(batch)

                with log_perf("matching"):
                    ocr_results.extend([
                        OCRResult(text=t, score=s, box=b)
                        for t, s, b in zip(txts, scrs, boxes[start:start+6], strict=True)
                        if t.strip()
                    ])

                    # Try simple matching first:
                    for p_idx, p in enumerate(phrases):
                        for r in ocr_results:
                            if p.strip().lower() in r.text.strip().lower():
                                res[p_idx] = r

                    if any(r is None for r in res):
                        arranged = arrange_text(
                            ocr_results,
                            line_cy_threshold=line_cy_threshold,
                            word_dist_threshold=word_dist_threshold,
                        )

                        for p_idx, p in enumerate(phrases):
                            if res[p_idx] is not None:
                                continue

                            if r := merge_phrase_boxes(arranged, p.split()):
                                res[p_idx] = r
                            elif 0 < fuzzy_match_min_similarity < 1 and (r := merge_phrase_boxes_fuzzy(arranged, p.split(), cutoff=fuzzy_match_min_similarity)):
                                res[p_idx] = r
                    else:
                        # we've found all phrases; no need to dig further
                        break

        return res



    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _detect(self, img: np.ndarray) -> list[BBox]:
        h, w = img.shape[:2]

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

        # each box is (top-left, top-right, bottom-right, bottom-left) so far

        boxes = [
            BBox(
                x=min(coord[0] for coord in b),
                y=min(coord[1] for coord in b),
                width=max(coord[0] for coord in b) - min(coord[0] for coord in b),
                height=max(coord[1] for coord in b) - min(coord[1] for coord in b),
            )
            for b in boxes
        ]

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

def filter_boxes_and_sort_by_proximity(
        boxes: list[BBox],
        positions: Tuple[Tuple[float, float], ...],
        max_dist: int
) -> List[BBox]:
    filtered = []
    pos_arr = np.array(positions)

    for box in boxes:
        center = np.array((box.cx, box.cy))
        # Compute distances to all positions
        dists = np.linalg.norm(pos_arr - center, axis=1)
        min_dist = dists.min()
        if min_dist <= max_dist:
            filtered.append((min_dist, box))

    # Sort by distance to the closest position
    filtered.sort(key=lambda x: x[0])

    # Return only the [BBox]es
    return [b for _, b in filtered]
