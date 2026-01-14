from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from .config import ExportConfig, TrainingConfig
from .mnist_data import load_mnist_processed
from .model import MLPNet, make_model
from .preprocessing import PreprocessMeta, normalize_uint8_images


@dataclass(frozen=True)
class TrainingMetrics:
    """Összegzett metrikák.

    Args:
        train_loss: Átlagos train loss.
        test_loss: Átlagos test loss.
        test_accuracy: Teszt pontosság (0..1).
        epochs: Lefuttatott epoch-ok száma.
        samples_train: Train minták száma.
        samples_test: Teszt minták száma.
    """

    train_loss: float
    test_loss: float
    test_accuracy: float
    epochs: int
    samples_train: int
    samples_test: int


class FlatTensorDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Flattenelt MNIST tensordataset.

    A képeket uint8 formában tároljuk, és __getitem__ során normalizálunk.
    """

    def __init__(self, images_uint8: torch.Tensor, labels: torch.Tensor) -> None:
        if images_uint8.ndim != 3:
            raise ValueError("images must be [N,28,28]")
        if labels.ndim != 1:
            raise ValueError("labels must be [N]")
        if images_uint8.shape[0] != labels.shape[0]:
            raise ValueError("images/labels length mismatch")

        self._images = images_uint8
        self._labels = labels.to(torch.int64)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._images[idx : idx + 1]  # [1,28,28]
        x = normalize_uint8_images(img).squeeze(0)  # [784]
        y = self._labels[idx]
        return x, y


def train_and_export(cfg: TrainingConfig, export_cfg: ExportConfig, *, logger: logging.Logger | None = None) -> Path:
    """MNIST tanítás + teszt + export.

    Args:
        cfg: Tanítási konfiguráció.
        export_cfg: Export konfiguráció.
        logger: Logger.

    Returns:
        Az exportált artefakt könyvtár.
    """

    log = logger or logging.getLogger(__name__)
    _seed_everything(cfg.seed)

    log.info("Loading MNIST")
    tr_i, tr_y, te_i, te_y = load_mnist_processed(cfg.data_dir, cfg.mnist_mirrors, logger=log)

    images = torch.cat([tr_i, te_i], dim=0)
    labels = torch.cat([tr_y, te_y], dim=0)

    full_ds = FlatTensorDataset(images, labels)
    n_total = len(full_ds)
    n_train = int(0.9 * n_total)
    n_test = n_total - n_train

    log.info("Splitting dataset", extra={"total": n_total, "train": n_train, "test": n_test})
    gen = torch.Generator().manual_seed(cfg.seed)
    ds_train, ds_test = random_split(full_ds, [n_train, n_test], generator=gen)

    device = _select_device(cfg.device)
    _log_device(device, log)

    train_loader = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = make_model(hidden_sizes=cfg.hidden_sizes, dropout=cfg.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    train_loss = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optim.step()

            batch_sz = int(x.shape[0])
            running += float(loss.item()) * batch_sz
            n_seen += batch_sz

        train_loss = running / max(1, n_seen)
        test_loss, test_acc = evaluate(model, test_loader, device)

        log.info(
            "Epoch completed",
            extra={
                "epoch": epoch,
                "epochs": cfg.epochs,
                "train_loss": round(train_loss, 6),
                "test_loss": round(test_loss, 6),
                "test_acc": round(test_acc, 6),
            },
        )

    metrics = TrainingMetrics(
        train_loss=float(train_loss),
        test_loss=float(test_loss),
        test_accuracy=float(test_acc),
        epochs=int(cfg.epochs),
        samples_train=int(n_train),
        samples_test=int(n_test),
    )

    artefact_dir = _export_run(cfg, export_cfg, model, metrics, PreprocessMeta(), logger=log)
    log.info("Export completed", extra={"artifact_dir": str(artefact_dir)})
    return artefact_dir


@torch.no_grad()
def evaluate(model: MLPNet, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Kiértékelés.

    Args:
        model: Modell.
        loader: Dataloader.
        device: Device.

    Returns:
        (loss, accuracy)
    """

    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item()) * int(x.shape[0])
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.shape[0])

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def _export_run(
    cfg: TrainingConfig,
    export_cfg: ExportConfig,
    model: MLPNet,
    metrics: TrainingMetrics,
    preprocess_meta: PreprocessMeta,
    *,
    logger: logging.Logger,
) -> Path:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tag = f"_{export_cfg.tag.strip()}" if export_cfg.tag.strip() else ""
    out_dir = cfg.artifacts_dir / f"run_{stamp}{tag}"
    out_dir.mkdir(parents=True, exist_ok=False)

    # Súlyok
    torch.save(model.state_dict(), out_dir / "weights.pt")

    # Konfig és meta
    (out_dir / "training_config.json").write_text(
        json.dumps(_training_cfg_to_json(cfg), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "preprocess_meta.json").write_text(
        json.dumps(asdict(preprocess_meta), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if export_cfg.include_training_report:
        (out_dir / "report.json").write_text(
            json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Minimálisan szükséges információ a web/CLI inference-hez.
    model_meta = {
        "arch": "mlp",
        "hidden_sizes": list(cfg.hidden_sizes),
        "dropout": cfg.dropout,
    }
    (out_dir / "model_meta.json").write_text(json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Artifacts written", extra={"dir": str(out_dir)})
    return out_dir


def _training_cfg_to_json(cfg: TrainingConfig) -> Dict[str, object]:
    return {
        "data_dir": str(cfg.data_dir),
        "artifacts_dir": str(cfg.artifacts_dir),
        "seed": cfg.seed,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "hidden_sizes": list(cfg.hidden_sizes),
        "dropout": cfg.dropout,
        "device": cfg.device,
        "num_workers": cfg.num_workers,
        "log_level": cfg.log_level,
        "mnist_base_url": cfg.mnist_base_url,
    }


def _log_device(device: torch.device, log: logging.Logger) -> None:
    """Eszköz logolása (CPU/GPU).

    Args:
        device: Kiválasztott eszköz.
        log: Logger.
    """

    if device.type == 'cuda' and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            name = None
        log.info('Using CUDA device', extra={'device': str(device), 'gpu_name': name, 'torch_cuda': torch.version.cuda})
    else:
        log.info('Using CPU device', extra={'device': str(device)})


def _select_device(choice: str) -> torch.device:
    c = str(choice).lower().strip()
    if c == "cpu":
        return torch.device("cpu")
    if c == "cuda":
        return torch.device("cuda")
    if c == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device: {choice}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
