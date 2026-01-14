from __future__ import annotations

import json
from typing import Any, Dict, List

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from digitrec_core.predictor import Prediction

from .services import parse_pixels_json, predict, predict_ui


@require_GET
def index(request: HttpRequest) -> HttpResponse:
    """Főoldal.

    Returns:
        HTML oldal.
    """

    return render(request, "digits/index.html")


@require_GET
def health(request: HttpRequest) -> JsonResponse:
    """Egészség ellenőrző endpoint.

    Returns:
        JSON válasz.
    """

    return JsonResponse({"status": "ok"})


@require_POST
def api_predict(request: HttpRequest) -> JsonResponse:
    """JSON predikció endpoint.

    Request JSON:
        {"pixels": [0..1 x 784]}

    Returns:
        {"digit": int, "probabilities": [10], "hidden_layers": [[...], [...]]}
    """

    try:
        obj: Any = json.loads(request.body.decode("utf-8"))
        pixels = obj.get("pixels")
        if not isinstance(pixels, list):
            raise ValueError("pixels must be a list")
        pred = predict([float(x) for x in pixels])
        return JsonResponse(
            {
                "digit": pred.digit,
                "probabilities": pred.probabilities,
                "hidden_layers": pred.hidden_layers,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return JsonResponse({"error": str(exc)}, status=400)


@require_POST
def ui_predict(request: HttpRequest) -> HttpResponse:
    """HTMX predikció: HTML fragment.

    Form fields:
        pixels_json: JSON lista (784 float)

    Returns:
        HTML fragment.
    """

    try:
        pixels_json = request.POST.get("pixels_json", "")
        req = parse_pixels_json(pixels_json)
        pred = predict_ui(req.pixels)
        return render(request, "digits/_prediction_fragment.html", _ui_context(pred))
    except Exception as exc:  # noqa: BLE001
        # HTMX esetén a 4xx státuszkód alapértelmezetten megakadályozhatja a swap-et.
        # A felhasználónak viszont hasznosabb, ha a fragmentben látja a hibaüzenetet.
        return render(request, "digits/_prediction_fragment.html", {"error": str(exc)})


def _ui_context(pred: Prediction) -> Dict[str, object]:
    """UI-hoz szükséges kontextus összeállítása.

    Megjegyzés:
        A template lokalizáció miatt a lebegőpontos értékek vesszővel is formázódhatnak.
        CSS-ben ez érvénytelen (pl. width: 0,1%). Ezért a CSS-hez külön, ponttal
        formázott stringeket adunk át.

    Args:
        pred: A predikció (számjegy, valószínűségek, rejtett rétegek).

    Returns:
        Template context dict.
    """

    probs: List[Dict[str, object]] = []
    for i, p in enumerate(pred.probabilities):
        prob = float(p)
        pct = prob * 100.0
        probs.append(
            {
                "digit": i,
                "prob": prob,
                "pct": pct,
                # CSS-hez: ponttal formázott százalék (pl. "92.5")
                "pct_css": f"{pct:.1f}",
            }
        )

    layers: List[Dict[str, object]] = []
    for idx, layer in enumerate(pred.hidden_layers, start=1):
        vals = [float(v) for v in layer]
        # Normalizálás: layer maximuma alapján (ReLU miatt sok 0 lehet).
        mx = max(vals) if vals else 0.0
        cells: List[Dict[str, object]] = []
        for v in vals:
            if mx <= 0.0 or v <= 0.0:
                alpha = 0.0
            else:
                norm = min(1.0, max(0.0, v / mx))
                # Enyhe gamma a kontrasztért
                norm = norm ** 0.5
                alpha = 0.05 + 0.95 * norm
            cells.append({"v": v, "alpha_css": f"{alpha:.3f}"})

        cols = 16
        layers.append({"idx": idx, "cols": cols, "cells": cells})

    pre_list = pred.preprocessed or []
    sal_list = pred.saliency or []

    return {
        "pred_digit": pred.digit,
        "probs": probs,
        "layers": layers,
        "has_explain": bool(pre_list) and bool(sal_list),
        "preprocessed_json": json.dumps(pre_list, ensure_ascii=False, separators=(",", ":")),
        "saliency_json": json.dumps(sal_list, ensure_ascii=False, separators=(",", ":")),
    }
