from __future__ import annotations

import json
from pathlib import Path

import torch

from digitrec_core.model import make_model
from digitrec_core.predictor import Predictor


def _make_artifact_dir(tmp_path: Path) -> Path:
    out = tmp_path / "artifact"
    out.mkdir(parents=True, exist_ok=True)

    model = make_model(hidden_sizes=(32, 16), dropout=0.0)
    torch.save(model.state_dict(), out / "weights.pt")

    (out / "model_meta.json").write_text(json.dumps({"hidden_sizes": [32, 16], "dropout": 0.0}), encoding="utf-8")
    return out


def test_predictor_returns_probabilities(tmp_path: Path) -> None:
    artifact = _make_artifact_dir(tmp_path)
    pred = Predictor(artifact, device="cpu")

    pixels = [0.0] * 784
    pixels[0] = 1.0
    result = pred.predict(pixels)

    assert 0 <= result.digit <= 9
    assert len(result.probabilities) == 10
    assert abs(sum(result.probabilities) - 1.0) < 1e-5
    assert len(result.hidden_layers) == 2
    assert len(result.hidden_layers[0]) == 32
    assert len(result.hidden_layers[1]) == 16


def test_predictor_explanations_include_saliency(tmp_path: Path) -> None:
    artifact = _make_artifact_dir(tmp_path)
    pred = Predictor(artifact, device="cpu")

    pixels = [0.0] * 784
    pixels[100] = 1.0
    result = pred.predict(pixels, include_explanations=True)

    assert result.preprocessed is not None
    assert result.saliency is not None
    assert len(result.preprocessed) == 784
    assert len(result.saliency) == 784
    assert all(0.0 <= x <= 1.0 for x in result.saliency)
