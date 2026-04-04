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

_cache_dir = Path.home() / ".cache" / "ppocr_lite"


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
    cache_dir: Path = field(default_factory=lambda: _cache_dir)

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


def list_downloaded_models() -> list[Path]:
    """
    List all models present in the default cache directory.
    """
    if not _cache_dir.exists():
        return []

    return list(_cache_dir.iterdir())


def download_default_models():
    """
    Download all default models to the default cache directory.
    """
    ModelConfig().resolve()


def download_model(url: str, name: str | None = None):
    """
    Download a model from the given [url] to the default cache directory.

    If [name] is given, it will be used as the file name; otherwise, the file name will be derived from [url].
    """
    _ensure(url, _cache_dir, name)


def get_cache_directory():
    """
    Get the default cache directory.
    """

    return _cache_dir


def set_cache_directory(pth: Path):
    """
    Change the default cache directory. This must be done before using the cache or built-in model
    management.
    """

    global _cache_dir

    _cache_dir = pth


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure(url: str, cache_dir: Path, name: str | None = None) -> Path:
    dest = cache_dir / (name or Path(url).name)

    dest.parent.mkdir(parents=True, exist_ok=True)

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
