from __future__ import annotations

import json


def test_health_endpoint(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_api_predict_returns_json(client, settings, artifact_dir) -> None:
    # A digits.services cache-eli a Predictor-t, ezért reseteljük.
    import digits.services as services

    services._PREDICTOR = None
    settings.DIGITREC_ARTIFACT_DIR = artifact_dir

    pixels = [0.0] * 784
    payload = {"pixels": pixels}

    resp = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 200
    body = resp.json()
    assert "digit" in body
    assert isinstance(body["probabilities"], list)
    assert len(body["probabilities"]) == 10


def test_ui_predict_renders_fragment(client, settings, artifact_dir) -> None:
    import digits.services as services

    services._PREDICTOR = None
    settings.DIGITREC_ARTIFACT_DIR = artifact_dir

    pixels = [0.0] * 784
    resp = client.post("/ui/predict", data={"pixels_json": json.dumps(pixels)})
    assert resp.status_code == 200
    html = resp.content.decode("utf-8")
    assert "Rejtett rétegek" in html
    assert "Mi látszik a modellnek?" in html
