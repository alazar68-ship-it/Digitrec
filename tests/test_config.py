from __future__ import annotations

import json
from pathlib import Path

import pytest

from digitrec_core.config import ConfigError, load_config


def test_load_config_ok(tmp_path: Path) -> None:
    cfg_p = tmp_path / "cfg.json"
    cfg_p.write_text(json.dumps({"training": {"epochs": 1}, "export": {}}), encoding="utf-8")

    cfg = load_config(cfg_p)
    assert cfg.training.epochs == 1


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.json")


def test_load_config_rejects_invalid_device(tmp_path: Path) -> None:
    cfg_p = tmp_path / "cfg.json"
    cfg_p.write_text(json.dumps({"training": {"device": "bad"}}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_p)
