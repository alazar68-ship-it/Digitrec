from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
import torch

from digitrec_core.model import make_model


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    """Egy minimális, tesztekhez használható artifact könyvtárat hoz létre."""

    out = tmp_path / "artifact"
    out.mkdir(parents=True, exist_ok=True)

    model = make_model(hidden_sizes=(32, 16), dropout=0.0)
    torch.save(model.state_dict(), out / "weights.pt")
    (out / "model_meta.json").write_text(json.dumps({"hidden_sizes": [32, 16], "dropout": 0.0}), encoding="utf-8")
    return out


@pytest.fixture(scope="session", autouse=True)
def _django_bootstrap() -> None:
    """Django inicializálás pytest-django nélkül.

    A projekt célja a minimális dev függőség, ezért a Django tesztklienshez
    nem támaszkodunk a pytest-django pluginra.
    """

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "digitrec_web.settings")

    # Django csak akkor importálható, ha a pythonpath be van állítva (pyproject.toml).
    import django

    django.setup()


@pytest.fixture()
def client() -> Any:
    """Django teszt kliens."""

    from django.test import Client

    return Client()


@pytest.fixture()
def settings() -> Iterator[Any]:
    """Egyszerű settings wrapper, visszaállítással.

    Returns:
        Olyan objektum, amire attribútumként lehet beállítani a settings mezőket.
    """

    from django.conf import settings as dj_settings

    original: Dict[str, Any] = {}
    had: Dict[str, bool] = {}

    class _Wrapper:
        def __getattr__(self, item: str) -> Any:
            return getattr(dj_settings, item)

        def __setattr__(self, key: str, value: Any) -> None:
            if key not in original:
                had[key] = hasattr(dj_settings, key)
                original[key] = getattr(dj_settings, key, None)
            setattr(dj_settings, key, value)

    w = _Wrapper()
    yield w

    for k, v in original.items():
        if had.get(k, False):
            setattr(dj_settings, k, v)
        else:
            try:
                delattr(dj_settings, k)
            except Exception:
                # Biztonsági fallback: ha nem törölhető, legalább None-ra állítjuk.
                setattr(dj_settings, k, None)
