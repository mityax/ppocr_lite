# oversimplifiedocr

A **lightweight** OCR runtime built on PaddleOCR models (PP-OCRv5 / v4) and
ONNX Runtime.

| Concern | Decision |
|---|---|
| OpenCV | ❌ not used |
| PyTorch / TensorFlow / PaddlePaddle | ❌ not used |
| Shapely / pyclipper | ❌ not used |
| numpy | ✅ required |
| Pillow | ✅ required |
| onnxruntime | ✅ required |
| scipy | ⚡ optional (faster connected-component labelling) |

## Motivation

The standard PaddleOCR Python package drags in the full PaddlePaddle deep-
learning framework plus OpenCV.  This library strips everything down to just
the inference path: image pre-processing with PIL/numpy, ONNX Runtime for
model execution, and a pure-numpy DB post-processor.  The result is a small
Docker image that starts quickly and stays fast.

## Installation

```bash
pip install onnxruntime pillow numpy           # core
pip install scipy                              # optional – faster detection
```

For GPU inference replace `onnxruntime` with `onnxruntime-gpu`.

## Model files

Download PP-OCRv5 mobile ONNX models from the
[PaddleOCR ONNX releases](https://github.com/PaddlePaddle/PaddleOCR) or
export your own.  You need three files:

| File | Purpose |
|---|---|
| `det.onnx` | DB text detector |
| `rec.onnx` | SVTR/CRNN text recogniser |
| `ppocrv5_dict.txt` (or equivalent) | Character dictionary |

Optionally:

| File | Purpose |
|---|---|
| `cls.onnx` | Angle classifier (detects 180° rotated text) |

## Quick start

```python
from oversimplifiedocr import OCREngine

engine = OCREngine(
    det_model_path      = "models/ppocrv5/det/det.onnx",
    rec_model_path      = "models/ppocrv5/rec/rec.onnx",
    rec_char_dict_path  = "models/ppocrv5/ppocrv5_dict.txt",
)

# Accepts: file path, PIL.Image, or numpy array (HWC RGB)
results = engine.run("screenshot.png")

for box, (text, score) in results:
    print(f"[{score:.2f}] {text}")
    # box = [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]  (TL → TR → BR → BL)

# Or just get the text:
print(engine.get_text("screenshot.png"))
```

## Engine options

```python
engine = OCREngine(
    det_model_path      = "...",
    rec_model_path      = "...",
    rec_char_dict_path  = "...",

    # --- Detection ---
    det_limit_side      = 960,      # max long edge after resize
    det_limit_type      = "max",    # 'max' or 'min'
    det_thresh          = 0.3,      # DB binarisation threshold
    det_box_thresh      = 0.6,      # min box confidence
    det_unclip_ratio    = 1.5,      # box expansion factor
    det_use_dilation    = False,    # dilate binary map before labelling

    # --- Recognition ---
    rec_img_h           = 48,       # model input height
    rec_img_w           = 320,      # model max input width
    rec_batch_size      = 6,        # crops per inference call

    # --- Angle classifier (optional) ---
    cls_model_path      = None,     # set to enable
    use_angle_cls       = False,
    cls_thresh          = 0.9,

    # --- Output ---
    drop_score          = 0.5,      # discard results below this confidence

    # --- Input format ---
    bgr_input           = False,    # True if your numpy arrays are BGR (OpenCV)

    # --- Hardware ---
    use_gpu             = False,
    gpu_id              = 0,
)
```

## Accepting OpenCV images

```python
import cv2
frame = cv2.imread("screenshot.png")   # BGR uint8

engine = OCREngine(..., bgr_input=True)
results = engine.run(frame)
```

## Docker

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir \
        onnxruntime \
        pillow \
        numpy \
        scipy      # optional but recommended

COPY models/ /app/models/
COPY oversimplifiedocr/ /app/oversimplifiedocr/

WORKDIR /app
```

## Architecture

```
OCREngine.run(image)
│
├─ TextDetector(image)           ← det.onnx
│   ├─ resize_for_det()          PIL resize, keep aspect, multiple-of-32
│   ├─ normalize_for_det()       ImageNet mean/std, BGR channel order
│   ├─ ONNX inference            → probability map (B,1,H,W)
│   └─ DBPostProcess             pure numpy
│       ├─ threshold             binary mask
│       ├─ connected components  scipy.ndimage.label or BFS fallback
│       ├─ _min_area_rect        rotating-calipers on convex hull
│       └─ _unclip               expand box from centroid
│
├─ sort_boxes()                  reading-order sort
│
├─ crop_text_region()            PIL.Image.QUAD perspective crop
│
├─ (TextClassifier)              cls.onnx – optional 180° correction
│
└─ TextRecognizer(crops)         rec.onnx
    ├─ resize_norm_img()         aspect-ratio-preserving resize + pad
    ├─ ONNX inference            → logit sequence (B,T,C)
    └─ CTCDecoder.decode()       greedy CTC → (text, confidence)
```

## Design notes

### Why no OpenCV?

OpenCV is ~80 MB when installed and brings in many shared libraries.
Every cv2 operation used by the original code has a direct equivalent:

| cv2 | Replacement |
|---|---|
| `cv2.imread` | `PIL.Image.open` |
| `cv2.resize` | `PIL.Image.resize` |
| `cv2.findContours` | numpy BFS / `scipy.ndimage.label` |
| `cv2.minAreaRect` + `cv2.boxPoints` | rotating calipers in numpy |
| `cv2.warpPerspective` | `PIL.Image.transform(QUAD)` |
| `cv2.rotate` | `np.rot90` |
| `cv2.dilate` | numpy bit-shift dilation |

### Why no pyclipper / shapely?

The standard DB post-processor uses pyclipper to expand each detected box
outward (the "unclip" step) and shapely for polygon area computation.
For desktop screenshot OCR – where text boxes are nearly rectangular and
never perspective-distorted – a much simpler centroid-based scaling
produces identical results:

```
expanded = centroid + (box - centroid) * unclip_ratio
```

This exact formula is exact for rectangles and a very close approximation
for the slightly-rotated boxes that occasionally appear in desktop UIs.

### Connected components without scipy

scipy is listed as an optional dependency.  When absent, the library falls
back to a pure-Python BFS labeller.  For typical desktop screenshot detection
maps (640 × 640) with a moderate number of text regions this is fast enough
(< 1 s).  Install scipy for production use:

```bash
pip install scipy
```
