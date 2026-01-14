from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from django.conf import settings

from digitrec_core.predictor import Prediction, Predictor


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictRequest:
    """Predikció kérés.

    Args:
        pixels: 784 elemű, 0..1 közötti értékek.
    """

    pixels: List[float]


def parse_pixels_json(payload: str) -> PredictRequest:
    """JSON stringből pixel lista.

    Args:
        payload: JSON string.

    Returns:
        Validált kérés.

    Raises:
        ValueError: Ha a JSON nem lista, vagy a lista nem 784 hosszú.
    """

    obj = json.loads(payload)
    if not isinstance(obj, list):
        raise ValueError("pixels JSON must be a list")
    pixels = [float(x) for x in obj]
    return PredictRequest(pixels=pixels)


_PREDICTOR: Predictor | None = None


def get_predictor() -> Predictor:
    """Lazy predictor betöltés.

    A web alkalmazás jellemzően hosszú életű process, ezért cache-eljük.

    Returns:
        Betöltött Predictor.

    Raises:
        FileNotFoundError: Ha nem található exportált modell.
    """

    global _PREDICTOR  # noqa: PLW0603
    if _PREDICTOR is not None:
        return _PREDICTOR

    artifact_dir = _resolve_artifact_dir(Path(settings.DIGITREC_ARTIFACT_DIR))
    log.info("Loading predictor", extra={"artifact_dir": str(artifact_dir)})
    _PREDICTOR = Predictor(artifact_dir)
    return _PREDICTOR


def predict(pixels: Sequence[float]) -> Prediction:
    """Predikció szolgáltatás.

    Args:
        pixels: 784 elemű pixel lista.

    Returns:
        Predikció.
    """

    return get_predictor().predict(pixels)


def predict_ui(pixels: Sequence[float]) -> Prediction:
    """Predikció UI-hoz, magyarázó adatokkal.

    Args:
        pixels: 784 elemű pixel lista.

    Returns:
        Predikció + (preprocessed + saliency) mezők.
    """

    return get_predictor().predict(pixels, include_explanations=True)


def _resolve_artifact_dir(base: Path) -> Path:
    # Ha konkrét run dir-t ad meg a user, azonnal használjuk.
    if (base / "weights.pt").exists() and (base / "model_meta.json").exists():
        return base

    # Egyébként keressük a legfrissebb run_* könyvtárat.
    candidates = [p for p in base.glob("run_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            "No exported model found. Set DIGITREC_ARTIFACT_DIR to an exported run directory."
        )

    # A név tartalmaz időbélyeget: lexikografikus rendezés elég.
    return sorted(candidates, key=lambda p: p.name)[-1]
