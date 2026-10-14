"""OCR backends for scanned pages.

Three backends are supported and probed at import time:

``tesseract``  pytesseract + a system Tesseract binary. Highest quality when the
               right language packs are installed; nothing to compile.
``rapidocr``   rapidocr-onnxruntime. Bundled ONNX models, so no system binary is
               required -- the convenient fallback.
``none``       No backend available. Scanned pages are reported rather than
               silently skipped.

Every public function degrades gracefully: an OCR failure yields a warning on the
page, never an exception that aborts a whole ingestion run.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class OcrEngine(str, Enum):
    NONE = "none"
    TESSERACT = "tesseract"
    RAPIDOCR = "rapidocr"


@dataclass
class OcrBackendStatus:
    """Status of a single OCR backend, compatible with OcrStatus field order."""
    engine: OcrEngine
    available: bool
    detail: str = ""
    languages: list[str] = field(default_factory=list)
    name: str = ""

    def __str__(self) -> str:
        if not self.available:
            return f"unavailable ({self.detail})"
        langs = f" [langs: {', '.join(self.languages)}]" if self.languages else ""
        return f"{self.name}{langs}"


# --------------------------------------------------------------------------
# Backend probing
# --------------------------------------------------------------------------

_rapidocr_instance = None
_rapidocr_failed = False
_rapidocr_error = ""


OcrStatus = OcrBackendStatus


def _probe_tesseract() -> OcrStatus:
    try:
        import pytesseract  # noqa: F401
    except Exception as exc:
        return OcrStatus(OcrEngine.NONE, False, f"pytesseract unavailable: {exc}")

    cmd = os.getenv("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd:
        return OcrStatus(
            OcrEngine.NONE,
            False,
            "Tesseract binary not found. Install Tesseract OCR and set TESSERACT_CMD "
            "(see README > OCR).",
        )
    try:
        pytesseract.pytesseract.tesseract_cmd = cmd
        langs = list(pytesseract.get_languages(config=""))
        return OcrStatus(OcrEngine.TESSERACT, True, f"binary={cmd}", languages=langs)
    except Exception as exc:
        return OcrStatus(OcrEngine.NONE, False, f"Tesseract not usable: {exc}")


def _probe_rapidocr() -> OcrStatus:
    # Import the concrete class, not just the package: the package __init__ can
    # succeed while its real dependency (shapely) fails, which would make us
    # advertise a backend that cannot actually run.
    try:
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    except Exception as exc:
        return OcrStatus(OcrEngine.NONE, False, f"rapidocr-onnxruntime unusable: {exc}")
    # Constructing the engine loads the ONNX models. Discovering a failure here
    # is better than rediscovering it once per scanned page.
    if get_rapidocr() is None:
        return OcrStatus(OcrEngine.NONE, False, _rapidocr_error or "RapidOCR init failed")
    return OcrStatus(
        OcrEngine.RAPIDOCR,
        True,
        "bundled ONNX models (no system binary required)",
        languages=["ch", "en"],
    )


def probe(requested: str = "auto") -> OcrStatus:
    """Report which OCR backend is usable.

    ``requested`` may be ``auto``, ``tesseract``, ``rapidocr`` or ``none``.
    With ``auto`` we prefer Tesseract for quality and fall back to RapidOCR.
    """
    requested = (requested or "auto").strip().lower()

    if requested == "none":
        return OcrStatus(OcrEngine.NONE, False, "OCR disabled by configuration.")

    if requested == OcrEngine.TESSERACT.value:
        return _probe_tesseract()
    if requested == OcrEngine.RAPIDOCR.value:
        return _probe_rapidocr()

    tess = _probe_tesseract()
    if tess.available:
        return tess
    rapid = _probe_rapidocr()
    if rapid.available:
        log.info("Tesseract unavailable (%s); using RapidOCR instead.", tess.detail)
        return rapid
    return OcrStatus(OcrEngine.NONE, False, tess.detail)


def probe_all() -> dict[str, OcrBackendStatus]:
    """Report the availability of all OCR backends.

    Returns a dict with keys ``"tesseract"``, ``"rapidocr"``, and ``"none"``.
    Each value is an :class:`OcrBackendStatus` describing that backend's
    availability and the reason.
    """
    tesseract = _probe_tesseract()
    rapidocr = _probe_rapidocr()
    # "none" status is implicit: if neither Tesseract nor RapidOCR is available,
    # the system has no OCR capability.
    return {
        "tesseract": OcrBackendStatus(
            name="tesseract",
            available=tesseract.available,
            detail=tesseract.detail,
            languages=tesseract.languages,
            engine=tesseract.engine,
        ),
        "rapidocr": OcrBackendStatus(
            name="rapidocr",
            available=rapidocr.available,
            detail=rapidocr.detail,
            languages=rapidocr.languages,
            engine=rapidocr.engine,
        ),
    }


def get_rapidocr():
    """Lazily construct the (expensive) RapidOCR engine singleton."""
    global _rapidocr_instance, _rapidocr_failed, _rapidocr_error
    if _rapidocr_instance is not None or _rapidocr_failed:
        return _rapidocr_instance
    try:
        from rapidocr_onnxruntime import RapidOCR

        _rapidocr_instance = RapidOCR()
    except Exception as exc:  # pragma: no cover - environment dependent
        log.warning("RapidOCR init failed: %s", exc)
        _rapidocr_failed = True
        _rapidocr_error = f"RapidOCR engine could not initialise: {exc}"
        _rapidocr_instance = None
    return _rapidocr_instance


# --------------------------------------------------------------------------
# OCR entry point
# --------------------------------------------------------------------------


def ocr_image_bytes(
    png_bytes: bytes, status: OcrStatus, languages: str = "eng"
) -> tuple[str, Optional[str]]:
    """Run OCR over a PNG image.

    Returns ``(text, error)``. ``error`` is ``None`` on success; on failure the
    text is empty and the caller records the error as a page warning.
    """
    if not status.available:
        return "", status.detail
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes))
        img.load()
    except Exception as exc:
        return "", f"could not decode page image: {exc}"

    if status.engine is OcrEngine.TESSERACT:
        try:
            import pytesseract

            lang = languages.split("+")[0] if languages else "eng"
            if lang not in status.languages:
                lang = "eng" if "eng" in status.languages else status.languages[0]
            text = pytesseract.image_to_string(img, lang=lang)
            return text, None
        except Exception as exc:
            return "", f"tesseract failed: {exc}"

    if status.engine is OcrEngine.RAPIDOCR:
        try:
            engine = get_rapidocr()
            if engine is None:
                return "", "RapidOCR engine could not initialise"
            import numpy as np

            result, _ = engine(np.array(img))
            if not result:
                return "", None
            # RapidOCR returns [box, text, score] triples.
            return "\n".join(str(line[1]) for line in result), None
        except Exception as exc:
            return "", f"rapidocr failed: {exc}"

    return "", "no OCR backend"