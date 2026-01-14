# Changelog


## 0.1.5
- Web UI: eliminated out-of-order HTMX swaps (flickering explanation canvases) by synchronizing prediction requests with `hx-sync="this:replace"`.


## 0.1.4
- Fixed saliency / explanation mode: `forward_with_trace()` no longer forces `torch.no_grad()`, so gradients are available for saliency computation.
- Web UI: `/ui/predict` now returns an error fragment with HTTP 200 to ensure HTMX swaps the panel and the user sees actionable errors.


## 0.1.3
- Web UI: fixed localized number formatting in inline CSS so probability bars and hidden-layer heatmaps render correctly.
- Predictor: added MNIST-style centering/cropping preprocessing for canvas inputs to improve real-world drawing accuracy.
- Web UI: improved Refresh button behavior with HTMX trigger + fetch fallback.
- Training: removed redundant device log entry to avoid JSON serialization issues.

## 0.1.1
- Fixed `digitrec` console-script entry point to accept no-arg invocation (compatible with `console_scripts`).
- Added NumPy as an explicit runtime dependency to avoid Torch initialization warnings.

## 0.1.2
- Fixed JSONL logger serialization when `extra` contains non-JSON types (e.g., `torch.device`).
- Normalized device logging to store `torch.device` as a string.

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.1.0] - 2026-01-02

### Added

- CLI: MNIST download, training, evaluation, export
- Web: Django+HTMX canvas, live prediction, hidden-layer visualization
- Tests: preprocessing, config, predictor, web endpoints
- Documentation: README, ARCHITECTURE, CONTRIBUTING, SECURITY, third-party licenses
