# Contributing

## Development setup

- Python 3.12+
- Create a virtual environment and install deps:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Quality gates

- Run unit tests:

```bash
pytest
```

## Adding dependencies

This project aims for minimal runtime dependencies. If you add a dependency:

1. Justify why it reduces risk or meaningfully reduces code.
2. Ensure license compatibility with Apache-2.0.
3. Update `docs/third_party_licenses.md`.

## Code style

- Type hints are required.
- Public functions/classes must have Google-style docstrings with English section headers and Hungarian explanations.
- Comments should be Hungarian.
