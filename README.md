# ppocr-lite

A lightweight PaddlePaddle-OCR runtime for images like screenshots. 

| Dependency | Role |
|---|---|
| `numpy` | All numerical computation |
| `Pillow` | Image I/O and resize |
| `onnxruntime` | Model inference |
| `scipy` *(optional)* | Faster connected-component labelling |

No OpenCV, deep-learning framework or utility libraries.

---

## Install

```bash
pip install ppocr-lite        # CPU
pip install ppocr-lite[gpu]   # GPU (uses onnxruntime-gpu)
pip install ppocr-lite[fast]  # + scipy for faster CC labelling
```

Models (PP-OCRv5 mobile det/rec + v2 direction cls) can be auto-downloaded to `~/.cache/ppocr_lite/` on
first use, or manually downloaded and configured.

**Automatically downloaded models** come from [RapidOCR](https://github.com/RapidAI/RapidOCR/tree/main) and are downloaded from huggingface 
(see [here](ppocr_lite/models.py) for details).

To **manually download models** see [their huggingface](https://huggingface.co/monkt/paddleocr-onnx) - you'll 
need one `det.onnx` (for text detection), one `rec.onnx` (for text recognition) and the corresponding `dict.txt` 
(the model-output-to-character mapping). The mobile (= smaller) models as 
[shipped by OnnxOCR](https://github.com/jingsongliujing/OnnxOCR/tree/main/onnxocr/models/ppocrv5) also work 
quite well.

---

## Quick Start

```python
from ppocr_lite import PPOCRLite

ocr_engine = PPOCRLite()

for result in ocr_engine.run("screenshot.png"):
    print(f"{result.score:.2f}  {result.text}")
    # result.box is a np.ndarray (4, 2) - top-left, top-right, bottom-right, bottom-left
```

### Use Your Own Models

```python
from ppocr_lite import PPOCRLite, ModelConfig
from pathlib import Path

ocr_engine = PPOCRLite(
    ModelConfig(
        det_model=Path("models/PP-OCRv5/det.onnx"),
        rec_model=Path("models/PP-OCRv5/rec.onnx"),
        dict_path=Path("models/PP-OCRv5/dict.txt"),
        cls_model=False,   # skip direction classifier
    )
)
```

### GPU inference

```python
ocr_engine = PPOCRLite(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

---

## Manage Downloaded Models

A few utility functions are available to configure from and to where models are downloaded:

```python
from ppocr_lite import models

models.set_cache_directory("./my-cache-dir")
models.get_cache_directory()  # -> pathlib.Path

models.list_downloaded_models() # -> list[pathlib.Path]
models.download_default_models()
models.download_model("https://huggingface.co/me/my-repo/resolve/main/my-model.onnx?download=true")
```

Of course you are entirely free to not use the built-in model management functionality and instead do
everything yourself – just configure your engine on initialization as described above.

---

## Design Notes

This project is very similar to the excellent [RapidOCR](https://github.com/RapidAI/RapidOCR/tree/main),
but more lightweight. Notably, it does not depend on OpenCV (which weighs around 200MB) and uses
numpy-based alternatives instead. This does not hurt performance much, at least in my humble tests.

Please be aware that many of those numpy-based alternatives are only really feasible because this project
assumes non-distorted input images (screenshots, clean document scans, …). I have not tested it, but I'd
assume it doesn't work nearly as well on inputs like perspective-distorted real-world photographs.

### What's different here?

* **Detection post-processing** – contour finding is replaced with
  scipy `ndimage.label` (or a numpy fallback).  The minimum-area
  rectangle is simplified under the assumption of non-perspective
  distorted input.  Polygon offset ("unclip") is done analytically 
  using the area/perimeter ratio and a per-vertex outward push — accurate 
  enough for near-rectangular screenshot text.

* **Resize** – PIL `BILINEAR` instead of `cv2.resize`.  The two are
  numerically equivalent for the precision required by OCR.

* **Crop** – axis-aligned bounding-rect crop instead of a perspective
  warp. Screenshot text is always axis-aligned, making this lossless.

* **No config YAML, no omegaconf** – plain Python dataclasses.

### Limitations vs. full PaddleOCR

* No perspective correction
* Direction classifier is only a 0°/180° binary; no 90°/270° support.

---

## License

This project is GPL-3.0-or-later licensed. Note that the licenses of models
(self-brought or auto-downloaded) will likely differ; refer to their creators
for more information.
