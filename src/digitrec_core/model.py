from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ForwardTrace:
    """Forward pass során rögzített aktivációk.

    Args:
        hidden_layers: Rejtett rétegek kimenetei, ReLU után.
        logits: A kimeneti logits (10).
    """

    hidden_layers: List[torch.Tensor]
    logits: torch.Tensor


class MLPNet(nn.Module):
    """Egyszerű, könnyen vizualizálható MLP MNIST-re.

    A CNN jobb pontosságot adhatna, de a web UI-ban kért "perceptron" vizualizáció
    MLP-nél átláthatóbb.
    """

    def __init__(self, hidden_sizes: Tuple[int, ...], dropout: float) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        in_features = 28 * 28

        for idx, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_features = h

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_features, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 784]
        feats = self.feature_extractor(x)
        return self.classifier(feats)

    def forward_with_trace(self, x: torch.Tensor) -> ForwardTrace:
        """Forward + köztes aktivációk.

        Args:
            x: Input [B, 784] float32, 0..1.

        Returns:
            A rejtett réteg aktivációk és logits.
        """

        hidden: List[torch.Tensor] = []
        cur = x
        for layer in self.feature_extractor:
            cur = layer(cur)
            if isinstance(layer, nn.ReLU):
                # ReLU után rögzítünk: [B, H]
                hidden.append(cur.clone())
        logits = self.classifier(cur)
        return ForwardTrace(hidden_layers=hidden, logits=logits)


def make_model(hidden_sizes: Tuple[int, ...], dropout: float) -> MLPNet:
    """Modell létrehozása.

    Args:
        hidden_sizes: Rejtett rétegek méretei.
        dropout: Dropout arány.

    Returns:
        Inizializált modell.
    """

    return MLPNet(hidden_sizes=hidden_sizes, dropout=dropout)
