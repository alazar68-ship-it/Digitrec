from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import torch


def load_mnist_processed(
    root: Path,
    mirrors: Iterable[str] | None = None,
    *,
    logger: logging.Logger | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """MNIST betöltése a torchvision beépített letöltőjével.

    A feladatban a cél az, hogy a MNIST letöltését ne egy saját URL-listás letöltő végezze,
    hanem a PyTorch ökoszisztémában szokásos megoldás: ``torchvision.datasets.MNIST``.

    Megjegyzés:
        A ``torchvision`` a letöltött fájlokat a megadott ``root`` alá cache-eli, és
        csak hiány esetén tölt. A dataset belsőleg a nyers adatot uint8 formában tárolja
        (``.data``), amit itt közvetlenül adunk vissza. A normalizálást (0..1) a tanítási
        pipeline végzi, így a web UI preprocess is ugyanazt a logikát követi.

    Args:
        root: A MNIST cache gyökérkönyvtára (például ``data``).
        mirrors: Kompatibilitási paraméter. A torchvision saját mirror listát használ;
            itt csak azért szerepel, hogy a korábbi konfigurációk változtatás nélkül
            működjenek.
        logger: Opcionális logger.

    Returns:
        (train_images_uint8, train_labels, test_images_uint8, test_labels)
        - images: (N, 28, 28) uint8
        - labels: (N,) int64

    Raises:
        RuntimeError: Ha a ``torchvision`` nincs telepítve.
    """
    _ = mirrors  # Kompatibilitás: a paramétert megtartjuk, de a torchvision intézi a tükröket.

    log = logger or logging.getLogger(__name__)

    try:
        from torchvision import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "A MNIST letöltéséhez/betöltéséhez a 'torchvision' csomag szükséges. "
            "Telepítés: pip install torchvision  (vagy CUDA-s index URL-lel együtt: "
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX)."
        ) from exc

    log.info("Ensuring MNIST is available via torchvision cache", extra={"root": str(root)})

    # A download=True biztosítja, hogy hiány esetén automatikusan letöltse és kicsomagolja.
    train_ds = datasets.MNIST(root=str(root), train=True, download=True)
    test_ds = datasets.MNIST(root=str(root), train=False, download=True)

    # A torchvision MNIST belső tárolója torch Tensor, ezért itt nincs szükség NumPy-ra.
    tr_i = train_ds.data.clone()
    tr_y = train_ds.targets.clone()
    te_i = test_ds.data.clone()
    te_y = test_ds.targets.clone()

    # Védő ellenőrzések: alak és dtype.
    if tr_i.dtype != torch.uint8 or te_i.dtype != torch.uint8:
        tr_i = tr_i.to(torch.uint8)
        te_i = te_i.to(torch.uint8)
    if tr_y.dtype != torch.int64:
        tr_y = tr_y.to(torch.int64)
    if te_y.dtype != torch.int64:
        te_y = te_y.to(torch.int64)

    return tr_i, tr_y, te_i, te_y
