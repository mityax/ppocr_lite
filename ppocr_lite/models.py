"""Model configuration and automatic download helpers."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Default model URLs (PP-OCRv5 mobile + v2 direction classifier)
# From: https://github.com/RapidAI/RapidOCR/tree/main
# ---------------------------------------------------------------------------

_DET_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0"
    "/onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx"
)
_DET_SHA256 = "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"

_REC_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0"
    "/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile_infer.onnx"
)
_REC_SHA256 = "5825fc7ebf84ae7a412be049820b4d86d77620f204a041697b0494669b1742c5"

_CLS_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.7.0"
    "/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx"
)
_CLS_SHA256 = "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"

_DEFAULT_CACHE = Path.home() / ".cache" / "ppocr_lite"


@dataclass
class ModelConfig:
    """Paths to ONNX model files.

    Unset paths (``None``) trigger auto-download to *cache_dir*.

    Parameters
    ----------
    det_model:
        Path to PP-OCRv5 detection model.  ``None`` → auto-download.
    rec_model:
        Path to PP-OCRv5 recognition model.  ``None`` → auto-download.
    cls_model:
        Path to direction-classifier model.  ``None`` → auto-download.
        Set to ``False`` to disable direction classification entirely.
    dict_path:
        Path to the recognizer output dictionary (dict.txt)
    cache_dir:
        Directory used when downloading models.
    """

    det_model: Path | None = None
    rec_model: Path | None = None
    cls_model: Path | None | bool = None  # False = disabled
    dict_path: Path | None = None
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CACHE)

    def resolve(self) -> "ModelConfig":
        """Return a copy with all ``None`` paths replaced by downloaded files."""
        cache = Path(self.cache_dir)
        cache.mkdir(parents=True, exist_ok=True)

        det = self.det_model or _ensure(_DET_URL, _DET_SHA256, cache)
        rec = self.rec_model or _ensure(_REC_URL, _REC_SHA256, cache)

        if self.cls_model is False:
            cls: Path | bool = False
        else:
            cls = self.cls_model or _ensure(_CLS_URL, _CLS_SHA256, cache)

        return ModelConfig(
            det_model=det,
            rec_model=rec,
            cls_model=cls,
            cache_dir=cache,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure(url: str, sha256: str, cache_dir: Path) -> Path:
    dest = cache_dir / Path(url).name
    if dest.exists() and _sha256(dest) == sha256:
        return dest
    print(f"[ppocr_lite] Downloading {dest.name} …", flush=True)
    _download(url, dest)
    actual = _sha256(dest)
    if actual != sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {dest.name}\n"
            f"  expected: {sha256}\n"
            f"  got:      {actual}"
        )
    return dest


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1 << 20  # 1 MiB
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                fh.write(data)
                downloaded += len(data)
                if total:
                    pct = downloaded * 100 // total
                    print(f"\r  {pct:3d}%", end="", flush=True)
        print()
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
