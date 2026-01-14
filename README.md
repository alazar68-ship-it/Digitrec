# Szám felismerés (MNIST)

A repository két komponensből áll:

- **CLI (Python)**: MNIST letöltés, modell tanítás, tesztelés és export.
- **Web (Django + HTMX)**: egérrel rajzolható 28×28-as digit canvas, folyamatos predikció, valószínűségek és a rejtett rétegek aktivációinak vizualizációja.

## Quickstart

### (1) Install

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

pip install -r requirements-dev.txt  # optional, tests
```

**CUDA / GPU note**

If you want PyTorch to use your NVIDIA GPU, install a CUDA-enabled build of **both** `torch` and `torchvision` (example for cu130):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

You can verify GPU support with:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```


### (2) Train & export

```bash
digitrec train --config configs/train_default.json
# Output: path to the artifact directory (e.g. artifacts/run_YYYYmmdd_HHMMSS_baseline)
```

### (3) Run the web UI

Set the exported artifact directory:

```bash
# Windows
set DIGITREC_ARTIFACT_DIR=artifacts\run_..._baseline
# Linux/macOS
# export DIGITREC_ARTIFACT_DIR=artifacts/run_..._baseline

python src/digitrec_web/manage.py migrate
python src/digitrec_web/manage.py runserver
```

Open: http://127.0.0.1:8000/

## Tests

```bash
pytest
```

## Project docs

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [docs/third_party_licenses.md](docs/third_party_licenses.md)

## License

Apache-2.0 (see [LICENSE](LICENSE))
