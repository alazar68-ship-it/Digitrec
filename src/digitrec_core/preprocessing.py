from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


@dataclass(frozen=True)
class PreprocessMeta:
    """Előfeldolgozási meta információ.

    Args:
        image_size: Képméret (sor, oszlop).
        flatten: Ha True, a kimenet 784 hosszú vektor.
        value_range: A várt bemeneti tartomány.
        normalization: Normalizáció leírása.
    """

    image_size: tuple[int, int] = (28, 28)
    flatten: bool = True
    value_range: str = "0..1"
    normalization: str = "x = uint8/255"


def normalize_uint8_images(images: torch.Tensor) -> torch.Tensor:
    """Uint8 MNIST képek normalizálása.

    Args:
        images: [N, 28, 28] uint8 tensor.

    Returns:
        [N, 784] float32 tensor 0..1.
    """

    if images.dtype != torch.uint8:
        raise ValueError("images must be uint8")
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("images must have shape [N,28,28]")

    x = images.to(torch.float32) / 255.0
    return x.view(x.shape[0], -1)


def coerce_pixels_list(pixels: Sequence[float]) -> torch.Tensor:
    """Webről érkező pixel lista validálása és tensorrá alakítása.

    Args:
        pixels: 784 elemű lista, elemenként 0..1.

    Returns:
        [1, 784] float32 tensor.
    """

    if len(pixels) != 28 * 28:
        raise ValueError("pixels must have length 784")

    # Gyors validálás: float és tartomány.
    out: List[float] = []
    for v in pixels:
        fv = float(v)
        if fv < 0.0 or fv > 1.0:
            raise ValueError("pixel values must be in [0,1]")
        out.append(fv)

    t = torch.tensor(out, dtype=torch.float32)
    return t.view(1, -1)



import torch.nn.functional as F


def preprocess_canvas_pixels(pixels: Sequence[float], *, threshold: float = 0.10) -> torch.Tensor:
    """Canvasról érkező 28x28 pixelek MNIST-szerű normalizálása.

    A felhasználói rajz tipikusan nincs középre igazítva és változó vastagságú.
    Ez a függvény megpróbálja az MNIST-hez hasonlóvá tenni:
      - küszöbölés a háttér zaj kiszűrésére,
      - bounding box alapú crop,
      - átméretezés ~20x20-as területre, majd 28x28-ra padding,
      - tömegközéppont szerinti eltolás középre.

    Args:
        pixels: 784 elemű lista, 0..1 közötti floatok.
        threshold: Maszk küszöb (0..1). Ennél kisebb értékeket háttérnek tekintjük.

    Returns:
        1x784 float tensor, 0..1 tartományban.
    """

    flat = coerce_pixels_list(pixels)  # 1x784
    img = flat.view(1, 1, 28, 28).clamp(0.0, 1.0)

    if float(img.max().item()) <= 0.0:
        return img.view(1, -1)

    # Maszk a nem üres régióra.
    mask = (img[0, 0] > float(threshold))
    if bool(mask.any().item()):
        ys, xs = mask.nonzero(as_tuple=True)
        y0 = int(ys.min().item())
        y1 = int(ys.max().item())
        x0 = int(xs.min().item())
        x1 = int(xs.max().item())

        # Kicsi ráhagyás a vágáshoz.
        margin = 2
        y0 = max(0, y0 - margin)
        y1 = min(27, y1 + margin)
        x0 = max(0, x0 - margin)
        x1 = min(27, x1 + margin)

        crop = img[:, :, y0 : y1 + 1, x0 : x1 + 1]
        h = int(crop.shape[-2])
        w = int(crop.shape[-1])

        # Átméretezés úgy, hogy a nagyobbik dimenzió 20 legyen.
        if h >= w:
            new_h = 20
            new_w = max(1, int(round(w * 20.0 / float(h))))
        else:
            new_w = 20
            new_h = max(1, int(round(h * 20.0 / float(w))))

        resized = F.interpolate(crop, size=(new_h, new_w), mode="bilinear", align_corners=False)

        canvas = torch.zeros((1, 1, 28, 28), dtype=resized.dtype, device=resized.device)
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        canvas[:, :, top : top + new_h, left : left + new_w] = resized
    else:
        canvas = img

    canvas = _shift_to_center_of_mass(canvas)
    canvas = canvas.clamp(0.0, 1.0)
    return canvas.view(1, -1)


def _shift_to_center_of_mass(img: torch.Tensor) -> torch.Tensor:
    """Tömegközéppont szerint középre igazít (egész pixelekkel).

    Args:
        img: 1x1x28x28 float tensor.

    Returns:
        Eltolt 1x1x28x28 tensor.
    """

    if img.ndim != 4 or img.shape[-2:] != (28, 28):
        raise ValueError("img must be 1x1x28x28")

    plane = img[0, 0]
    mass = float(plane.sum().item())
    if mass <= 1e-8:
        return img

    ys = torch.arange(28, device=img.device, dtype=plane.dtype).view(28, 1)
    xs = torch.arange(28, device=img.device, dtype=plane.dtype).view(1, 28)

    cy = float((plane * ys).sum().item() / mass)
    cx = float((plane * xs).sum().item() / mass)

    target = 13.5
    shift_y = int(round(target - cy))
    shift_x = int(round(target - cx))

    return torch.roll(img, shifts=(shift_y, shift_x), dims=(2, 3))
