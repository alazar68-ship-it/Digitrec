from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


def _to_jsonable(value: Any) -> Any:
    """JSON-serializálható formára alakítás.

    A cél, hogy a strukturált log ne omoljon össze akkor sem, ha az "extra"
    mezőkben nem JSON-serializálható típus szerepel (pl. torch.device).
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    # Gyakori konténerek rekurzív kezelése.
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    # torch.device támogatás (torch import csak akkor, ha tényleg kell)
    try:  # pragma: no cover
        import torch  # type: ignore

        if isinstance(value, torch.device):
            return str(value)
    except Exception:
        pass

    # "Utolsó mentsvár": stringgé alakítás.
    try:
        return str(value)
    except Exception:  # pragma: no cover
        return repr(value)


@dataclass(frozen=True)
class LogConfig:
    """Logging konfiguráció.

    Args:
        level: Log szint (pl. "DEBUG", "INFO").
        json_lines: Ha True, JSONL (soronként JSON) formátumban ír.
        utc_timestamps: Ha True, az időbélyegek UTC-ben lesznek.
    """

    level: str = "INFO"
    json_lines: bool = True
    utc_timestamps: bool = True


class JsonLineFormatter(logging.Formatter):
    """Egyszerű, stabil JSONL log formázó.

    A cél a jól géppel feldolgozható, strukturált log, külső függőség nélkül.
    """

    def __init__(self, *, utc_timestamps: bool) -> None:
        super().__init__()
        self._utc = utc_timestamps

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: MutableMapping[str, Any] = {
            "ts": self._format_ts(record.created),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Ha extra mezők érkeznek (logger.info(..., extra={...})), azokat is beemeljük.
        for k, v in record.__dict__.items():
            if k in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                continue
            if k.startswith("_"):
                continue
            payload[k] = _to_jsonable(v)

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _format_ts(self, created: float) -> str:
        if self._utc:
            dt = datetime.fromtimestamp(created, tz=timezone.utc)
        else:
            dt = datetime.fromtimestamp(created)
        return dt.isoformat(timespec="milliseconds")


def configure_logging(config: LogConfig | None = None) -> None:
    """Alkalmazásszintű logging inicializálás.

    Args:
        config: Logging konfiguráció. Ha None, környezeti változókból és alapértelmezésből épül fel.

    Returns:
        None.
    """

    cfg = config or _config_from_env()

    root = logging.getLogger()
    # Többszöri hívásnál ne halmozzuk a handler-eket.
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(_coerce_level(cfg.level))

    handler = logging.StreamHandler(stream=sys.stdout)
    if cfg.json_lines:
        handler.setFormatter(JsonLineFormatter(utc_timestamps=cfg.utc_timestamps))
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root.addHandler(handler)


def _coerce_level(level: str) -> int:
    name = str(level).upper().strip()
    return getattr(logging, name, logging.INFO)


def _config_from_env() -> LogConfig:
    level = os.environ.get("DIGITREC_LOG_LEVEL", "INFO")
    json_lines = os.environ.get("DIGITREC_LOG_JSON", "1") not in {"0", "false", "False"}
    utc = os.environ.get("DIGITREC_LOG_UTC", "1") not in {"0", "false", "False"}
    return LogConfig(level=level, json_lines=json_lines, utc_timestamps=utc)


def _to_jsonable(value: Any) -> Any:
    """Biztonságos JSON-serializálható értékké alakítás.

    A logging extra mezőkbe gyakran kerülhetnek olyan objektumok (pl. torch.device,
    Path), amelyeket a json modul alapból nem tud serializálni. Itt kis költséggel
    stabilan stringgé/listává/dict-té alakítjuk őket.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]

    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}

    # torch.device tipikusan ide kerül a training során.
    try:  # pragma: no cover
        import torch

        if isinstance(value, torch.device):
            return str(value)
    except Exception:
        pass

    # Végső fallback: string.
    return str(value)
