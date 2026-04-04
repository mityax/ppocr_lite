"""ppocr_lite – a lightweight PP-OCR runtime built on onnxruntime + numpy + PIL.

No OpenCV, no deep-learning framework required.

Typical usage::

    from ppocr_lite import PPOCRLite

    ocr = PPOCRLite()          # auto-downloads models on first run
    results = ocr("screenshot.png")
    for text, score, box in results:
        print(f"{score:.2f}  {text}  {box}")
"""

from .engine import PPOCRLite
from .models import ModelConfig

__all__ = ["PPOCRLite", "ModelConfig"]
__version__ = "0.1.0"
