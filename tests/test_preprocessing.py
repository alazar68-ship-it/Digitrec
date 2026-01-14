from __future__ import annotations

import pytest
import torch

from digitrec_core.preprocessing import coerce_pixels_list, normalize_uint8_images


def test_normalize_uint8_images_shape_and_range() -> None:
    imgs = torch.zeros((2, 28, 28), dtype=torch.uint8)
    imgs[0, 0, 0] = 255

    out = normalize_uint8_images(imgs)
    assert out.shape == (2, 784)
    assert out.dtype == torch.float32
    assert out.max().item() == pytest.approx(1.0)
    assert out.min().item() == pytest.approx(0.0)


def test_coerce_pixels_list_validates_length() -> None:
    with pytest.raises(ValueError):
        coerce_pixels_list([0.0] * 10)


def test_coerce_pixels_list_validates_range() -> None:
    bad = [0.0] * 784
    bad[0] = 1.5
    with pytest.raises(ValueError):
        coerce_pixels_list(bad)
