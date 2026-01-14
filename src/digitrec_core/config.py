from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConfigError(ValueError):
    """Konfigurációs hiba.

    Args:
        message: Emberbarát hibaüzenet.
    """


@dataclass(frozen=True)
class TrainingConfig:
    """Tanítási konfiguráció.

    Args:
        data_dir: A MNIST letöltésének és cache-ének gyökérkönyvtára.
        artifacts_dir: Ide mentjük a tanítási artefaktokat (súlyok, meta, riportok).
        seed: Reprodukálhatósági seed.
        epochs: Epoch-ok száma.
        batch_size: Batch méret.
        lr: Tanulási ráta.
        hidden_sizes: Rejtett rétegek neuronszámai.
        dropout: Dropout arány (0..1).
        device: "auto" | "cpu" | "cuda".
        num_workers: DataLoader worker-ek száma.
        log_level: Log szint.
        mnist_mirrors: MNIST letöltési tükör-URL-ek (sorrendben próbáljuk).
        mnist_base_url: (Kompatibilitás) Egyetlen alap URL. Ha a mnist_mirrors nincs megadva a configban, ezt használjuk.
    """

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    seed: int = 1337
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    hidden_sizes: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    device: str = "auto"
    num_workers: int = 0
    log_level: str = "INFO"
    log_level: str = "INFO"
    mnist_mirrors: tuple[str, ...] = (
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://azureopendatastorage.blob.core.windows.net/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    )
    mnist_base_url: str = "https://ossci-datasets.s3.amazonaws.com/mnist/"


@dataclass(frozen=True)
class ExportConfig:
    """Export konfiguráció.

    Args:
        tag: Opcionális címke az export könyvtár nevében.
        include_training_report: Ha True, JSON riportot is mentünk a futásról.
    """

    tag: str = ""
    include_training_report: bool = True


@dataclass(frozen=True)
class AppConfig:
    """Fő konfiguráció.

    Args:
        training: Tanítási paraméterek.
        export: Export paraméterek.
    """

    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def load_config(path: Path) -> AppConfig:
    """Konfiguráció betöltése JSON-ból.

    Args:
        path: JSON fájl elérési útja.

    Returns:
        A beolvasott konfiguráció.

    Raises:
        ConfigError: Ha a fájl hiányzik, nem JSON, vagy érvénytelen értékeket tartalmaz.
    """

    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid JSON in config: {p}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Top-level JSON must be an object")

    training = _parse_training(raw.get("training", {}))
    export = _parse_export(raw.get("export", {}))

    return AppConfig(training=training, export=export)


def _parse_training(obj: Any) -> TrainingConfig:
    if obj is None:
        return TrainingConfig()
    if not isinstance(obj, dict):
        raise ConfigError("training must be an object")

    def g(key: str, default: Any) -> Any:
        return obj.get(key, default)

    data_dir = Path(str(g("data_dir", "data")))
    artifacts_dir = Path(str(g("artifacts_dir", "artifacts")))

    hidden_sizes = tuple(int(x) for x in _ensure_seq(g("hidden_sizes", [128, 64]), "hidden_sizes"))


    def _norm_url(u: str) -> str:
        s = str(u).strip()
        if not s:
            return s
        return s if s.endswith("/") else s + "/"

    mirrors_raw = obj.get("mnist_mirrors", None)
    if mirrors_raw is None:
        mirrors_raw = obj.get("mnist_base_urls", None)

    if mirrors_raw is None:
        mirrors = (str(g("mnist_base_url", "https://ossci-datasets.s3.amazonaws.com/mnist/")),)
    else:
        mirrors = tuple(str(x) for x in _ensure_seq(mirrors_raw, "mnist_mirrors"))

    mirrors = tuple(_norm_url(m) for m in mirrors if str(m).strip())
    base_url = mirrors[0] if mirrors else "https://ossci-datasets.s3.amazonaws.com/mnist/"
    cfg = TrainingConfig(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        seed=int(g("seed", 1337)),
        epochs=int(g("epochs", 5)),
        batch_size=int(g("batch_size", 128)),
        lr=float(g("lr", 1e-3)),
        hidden_sizes=hidden_sizes,
        dropout=float(g("dropout", 0.1)),
        device=str(g("device", "auto")),
        num_workers=int(g("num_workers", 0)),
        log_level=str(g("log_level", "INFO")),
        mnist_mirrors=mirrors,
        mnist_base_url=base_url,
    )

    _validate_training(cfg)
    return cfg


def _parse_export(obj: Any) -> ExportConfig:
    if obj is None:
        return ExportConfig()
    if not isinstance(obj, dict):
        raise ConfigError("export must be an object")

    cfg = ExportConfig(
        tag=str(obj.get("tag", "")),
        include_training_report=bool(obj.get("include_training_report", True)),
    )
    return cfg


def _validate_training(cfg: TrainingConfig) -> None:
    if cfg.epochs <= 0:
        raise ConfigError("epochs must be > 0")
    if cfg.batch_size <= 0:
        raise ConfigError("batch_size must be > 0")
    if cfg.lr <= 0:
        raise ConfigError("lr must be > 0")
    if not (0.0 <= cfg.dropout < 1.0):
        raise ConfigError("dropout must be in [0, 1)")
    if not cfg.hidden_sizes:
        raise ConfigError("hidden_sizes must contain at least one layer")
    if any(h <= 0 for h in cfg.hidden_sizes):
        raise ConfigError("hidden_sizes values must be > 0")
    if cfg.device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("device must be one of: auto, cpu, cuda")

    if not cfg.mnist_mirrors:
        raise ConfigError("mnist_mirrors must not be empty")
    for u in cfg.mnist_mirrors:
        if not (u.startswith("http://") or u.startswith("https://")):
            raise ConfigError("mnist_mirrors entries must be http(s) URLs")


def _ensure_seq(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    raise ConfigError(f"{name} must be an array")
