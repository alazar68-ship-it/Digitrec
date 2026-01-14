from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from .model import ForwardTrace, MLPNet, make_model
from .preprocessing import PreprocessMeta, preprocess_canvas_pixels


@dataclass(frozen=True)
class Prediction:
    """Predikció eredmény.

    Args:
        digit: A legvalószínűbb számjegy.
        probabilities: 10 elemű valószínűségek.
        hidden_layers: Rejtett rétegek aktivációi, 0..1-re normálva.
        preprocessed: A modell által ténylegesen látott 28x28 kép (784 elem), 0..1.
        saliency: 28x28 saliency térkép (784 elem), 0..1.
    """

    digit: int
    probabilities: List[float]
    hidden_layers: List[List[float]]
    preprocessed: Optional[List[float]] = None
    saliency: Optional[List[float]] = None


class Predictor:
    """Modell betöltése és inference."""

    def __init__(self, artifact_dir: Path, *, device: str = "auto") -> None:
        self._artifact_dir = Path(artifact_dir)
        if not self._artifact_dir.exists():
            raise FileNotFoundError(f"Artifact dir not found: {self._artifact_dir}")

        meta = json.loads((self._artifact_dir / "model_meta.json").read_text(encoding="utf-8"))
        hidden_sizes = tuple(int(x) for x in meta.get("hidden_sizes", [128, 64]))
        dropout = float(meta.get("dropout", 0.1))

        self._device = _select_device(device)
        self._model: MLPNet = make_model(hidden_sizes=hidden_sizes, dropout=dropout).to(self._device)

        weights_path = self._artifact_dir / "weights.pt"
        state = _load_state_dict(weights_path, self._device)
        self._model.load_state_dict(state)
        self._model.eval()

        self._preprocess: PreprocessMeta = PreprocessMeta()

    def predict(self, pixels: Sequence[float], *, include_explanations: bool = False) -> Prediction:
        """Predikció 784 elemű pixel listából.

        Args:
            pixels: 0..1 értékek.

        Returns:
            Predikció eredmény.
        """

        x = preprocess_canvas_pixels(pixels).to(self._device)

        if not include_explanations:
            with torch.no_grad():
                trace: ForwardTrace = self._model.forward_with_trace(x)
                probs = torch.softmax(trace.logits, dim=1).squeeze(0).to("cpu")
                digit = int(torch.argmax(probs).item())

                hidden_norm: List[List[float]] = []
                for layer in trace.hidden_layers:
                    hidden_norm.append(_normalize_1d(layer.squeeze(0).to("cpu")))

            return Prediction(
                digit=digit,
                probabilities=[float(p) for p in probs.tolist()],
                hidden_layers=hidden_norm,
            )

        # Magyarázó mód: saliency térkép számítása gradiensből.
        x_req = x.clone().detach().requires_grad_(True)
        trace = self._model.forward_with_trace(x_req)
        probs_t = torch.softmax(trace.logits, dim=1).squeeze(0)
        digit = int(torch.argmax(probs_t).item())

        self._model.zero_grad(set_to_none=True)
        target = trace.logits[0, digit]
        target.backward()

        grad = x_req.grad
        if grad is None:
            sal = torch.zeros_like(x_req)
        else:
            # Egyszerű attribution: |grad| * |x| (28x28). A cél az intuitív, stabil hőtérkép.
            sal = (grad.abs() * x_req.detach().abs()).detach()

        pre = x_req.detach().view(-1).to("cpu")
        sal_norm = _normalize_784(sal.view(-1))

        hidden_norm = [_normalize_1d(layer.squeeze(0).detach().to("cpu")) for layer in trace.hidden_layers]
        probs = probs_t.detach().to("cpu")

        return Prediction(
            digit=digit,
            probabilities=[float(p) for p in probs.tolist()],
            hidden_layers=hidden_norm,
            preprocessed=[float(v) for v in pre.tolist()],
            saliency=sal_norm,
        )


def _normalize_784(t: torch.Tensor) -> List[float]:
    """784 elemű tensort 0..1 tartományra skáláz.

    Args:
        t: 784 elemű tensor.

    Returns:
        784 elemű float lista, 0..1.
    """

    flat = t.view(-1).to("cpu", dtype=torch.float32)
    if int(flat.numel()) != 28 * 28:
        flat = flat[: 28 * 28]
        if int(flat.numel()) < 28 * 28:
            pad = torch.zeros((28 * 28 - int(flat.numel()),), dtype=flat.dtype)
            flat = torch.cat([flat, pad], dim=0)

    mn = float(flat.min().item())
    mx = float(flat.max().item())
    if mx - mn < 1e-12:
        return [0.0 for _ in range(28 * 28)]
    out = ((flat - mn) / (mx - mn)).clamp(0.0, 1.0)
    return [float(x) for x in out.tolist()]


def _load_state_dict(path: Path, device: torch.device) -> dict:
    """State dict betöltése biztonságosabb módban.

    A PyTorch újabb verzióiban a `weights_only=True` opció csökkenti a pickle
    alapú betöltés kockázatát. Régebbi verzióknál fallback a klasszikus
    betöltésre.

    Args:
        path: A súlyfájl útvonala.
        device: Cél eszköz.

    Returns:
        A modell state_dict-ja.
    """

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _normalize_1d(t: torch.Tensor) -> List[float]:
    if t.ndim != 1:
        t = t.view(-1)
    mn = float(t.min().item())
    mx = float(t.max().item())
    if mx - mn < 1e-12:
        return [0.0 for _ in range(int(t.numel()))]
    out = ((t - mn) / (mx - mn)).clamp(0, 1)
    return [float(x) for x in out.tolist()]


def _select_device(choice: str) -> torch.device:
    c = str(choice).lower().strip()
    if c == "cpu":
        return torch.device("cpu")
    if c == "cuda":
        return torch.device("cuda")
    if c == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device: {choice}")
