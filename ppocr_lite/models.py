"""Model configuration and automatic download helpers."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Default model URLs (PP-OCRv5 mobile + v2 direction classifier)
# From: https://github.com/RapidAI/RapidOCR/tree/main
# ---------------------------------------------------------------------------

_DET_URL = "https://huggingface.co/ilaylow/PP_OCRv5_mobile_onnx/resolve/main/ppocrv5_det.onnx?download=true"
_REC_URL = "https://huggingface.co/ilaylow/PP_OCRv5_mobile_onnx/resolve/main/ppocrv5_rec.onnx?download=true"
_DICT_URL = "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/chinese/dict.txt?download=true"  # appears to be the appropriate dict for the above _REC_URL

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

        det = self.det_model or _ensure(_DET_URL, cache)
        rec = self.rec_model or _ensure(_REC_URL, cache)
        dic = self.dict_path or _ensure(_DICT_URL, cache)

        if self.cls_model is False:
            cls: Path | bool = False
        else:
            cls = self.cls_model

        return ModelConfig(
            det_model=det,
            rec_model=rec,
            dict_path=dic,
            cls_model=cls,
            cache_dir=cache,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure(url: str, cache_dir: Path) -> Path:
    dest = cache_dir / Path(url).name
    if dest.exists():
        return dest
    print(f"[ppocr_lite] Downloading {dest.name} …", flush=True)
    _download(url, dest)
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
