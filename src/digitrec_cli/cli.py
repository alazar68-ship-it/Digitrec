from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from digitrec_core.config import AppConfig, ConfigError, load_config
from digitrec_core.loggingx import LogConfig, configure_logging
from digitrec_core.predictor import Predictor
from digitrec_core.training import train_and_export


class ExitCode:
    """Egységes exit code-ok.

    A cél, hogy automatizált környezetben (CI, script) is könnyen kezelhető legyen.
    """

    OK = 0
    BAD_ARGS = 2
    CONFIG_ERROR = 3
    RUNTIME_ERROR = 10


def main(argv: Optional[List[str]] = None) -> int:
    """CLI belépési pont.

    Args:
        argv: Parancssori argumentumok (a programnév nélkül).

    Returns:
        Exit code.
    """

    parser = _build_parser()
    ns = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # Logging alapból is működjön, még config előtt.
    configure_logging(LogConfig(level=getattr(ns, "log_level", "INFO")))

    log = logging.getLogger("digitrec")

    try:
        if ns.command == "train":
            return _cmd_train(ns, log)
        if ns.command == "predict":
            return _cmd_predict(ns, log)
        if ns.command == "licenses":
            return _cmd_licenses(ns, log)

        parser.print_help()
        return ExitCode.BAD_ARGS
    except ConfigError as exc:
        log.error("Config error", extra={"error": str(exc)})
        return ExitCode.CONFIG_ERROR
    except KeyboardInterrupt:
        log.error("Interrupted")
        return ExitCode.RUNTIME_ERROR
    except SystemExit as exc:
        # argparse vagy explicit SystemExit
        code = int(getattr(exc, "code", ExitCode.RUNTIME_ERROR) or ExitCode.RUNTIME_ERROR)
        return code
    except Exception as exc:  # noqa: BLE001
        log.exception("Unhandled error", extra={"error": str(exc)})
        return ExitCode.RUNTIME_ERROR


def _cmd_train(ns: argparse.Namespace, log: logging.Logger) -> int:
    cfg = _load_and_apply_overrides(Path(ns.config), ns)

    # A configból is állítsuk a logszintet.
    configure_logging(LogConfig(level=cfg.training.log_level))
    log.info("Starting training", extra={"config": str(ns.config)})

    artifact_dir = train_and_export(cfg.training, cfg.export, logger=log)

    # Rövid, emberi összegzés a futás végén.
    report_path = artifact_dir / "report.json"
    if report_path.exists():
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        log.info(
            "Run summary",
            extra={
                "artifact_dir": str(artifact_dir),
                "train_loss": rep.get("train_loss"),
                "test_loss": rep.get("test_loss"),
                "test_accuracy": rep.get("test_accuracy"),
                "epochs": rep.get("epochs"),
            },
        )

    print(str(artifact_dir))
    return ExitCode.OK


def _cmd_predict(ns: argparse.Namespace, log: logging.Logger) -> int:
    predictor = Predictor(Path(ns.artifact_dir), device=ns.device)

    pixels: List[float]
    if ns.pixels_json:
        pixels = json.loads(ns.pixels_json)
    else:
        pixels = json.loads(Path(ns.pixels_file).read_text(encoding="utf-8"))

    pred = predictor.predict(pixels)
    payload = {
        "digit": pred.digit,
        "probabilities": pred.probabilities,
        "hidden_layers": pred.hidden_layers,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return ExitCode.OK


def _cmd_licenses(ns: argparse.Namespace, log: logging.Logger) -> int:
    out = Path(ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    md = _render_third_party_licenses()
    out.write_text(md, encoding="utf-8")

    log.info("Wrote third-party licenses", extra={"path": str(out)})
    print(str(out))
    return ExitCode.OK


def _load_and_apply_overrides(config_path: Path, ns: argparse.Namespace) -> AppConfig:
    cfg = load_config(config_path)

    # Minimál override: a CLI legyen gyorsan használható config módosítás nélkül.
    t = cfg.training
    overrides = {
        "epochs": ns.epochs,
        "batch_size": ns.batch_size,
        "lr": ns.lr,
        "device": ns.device,
    }

    # dataclass immutábilis -> új példány.
    t2 = t
    for k, v in overrides.items():
        if v is None:
            continue
        t2 = t2.__class__(**{**t2.__dict__, k: v})

    return cfg.__class__(training=t2, export=cfg.export)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="digitrec", description="MNIST digit recognition pipeline")
    p.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")

    sub = p.add_subparsers(dest="command")

    train = sub.add_parser("train", help="Download MNIST, train, evaluate, export")
    train.add_argument("--config", required=True, help="Path to JSON config")
    train.add_argument("--epochs", type=int, default=None, help="Override epochs")
    train.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    train.add_argument("--lr", type=float, default=None, help="Override learning rate")
    train.add_argument("--device", default="auto", help="auto|cpu|cuda")

    predict = sub.add_parser("predict", help="Run inference for a 28x28 image represented as a JSON list")
    predict.add_argument("--artifact-dir", required=True, help="Exported artifact directory")
    group = predict.add_mutually_exclusive_group(required=True)
    group.add_argument("--pixels-json", help="Inline JSON list with 784 floats")
    group.add_argument("--pixels-file", help="Path to a JSON file with 784 floats")
    predict.add_argument("--device", default="auto", help="auto|cpu|cuda")

    lic = sub.add_parser("licenses", help="Generate docs/third_party_licenses.md")
    lic.add_argument("--output", default="docs/third_party_licenses.md", help="Output path")

    return p


def _render_third_party_licenses() -> str:
    """Third-party licence lista generálása.

    Fontos: nem másoljuk be a licencszövegeket, csak a tényeket listázzuk.

    Returns:
        Markdown szöveg.
    """

    from importlib.metadata import PackageNotFoundError, version

    def v(pkg: str) -> str:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return "(not installed)"

    # Megjegyzés: Python csomagok licence mezője gyakran hiányos.
    # Itt "best effort" listát adunk a fő futásidejű függőségekről.
    items = [
        ("Django", v("Django"), "BSD-3-Clause"),
        ("torch", v("torch"), "BSD-3-Clause"),
        ("HTMX", "(CDN)", "BSD-2-Clause"),
    ]

    lines = [
        "# Third-party licenses",
        "",
        "This project aims for Apache-2.0 compatibility. The list below is a best-effort inventory.",
        "",
        "| Dependency | Version | License | Notes |",
        "|---|---:|---|---|",
    ]

    for name, ver, lic in items:
        note = "" if name != "HTMX" else "Loaded from CDN; not vendored into the repo."
        lines.append(f"| {name} | {ver} | {lic} | {note} |")

    lines.append("")
    lines.append("If you add dependencies, update this file.")
    return "\n".join(lines)
