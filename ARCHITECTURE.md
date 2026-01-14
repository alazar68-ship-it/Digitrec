# Architecture

## High-level layout

```
src/
  digitrec_core/   # ML: data, model, training, inference
  digitrec_cli/    # CLI entrypoints
  digitrec_web/    # Django project + app
configs/
docs/
tests/
```

## Module boundaries

### Core

- `digitrec_core.mnist_data`: download (stream), parse IDX, cache as tensors
- `digitrec_core.preprocessing`: normalization + web-pixel validation
- `digitrec_core.model`: MLP model and traced forward pass
- `digitrec_core.training`: train/eval/export pipeline
- `digitrec_core.predictor`: load exported artifacts and serve predictions

### Web

Routes are split along a simple layering guideline:

- `digits.views`: HTTP routes (HTML and JSON)
- `digits.services`: orchestration (loading Predictor, request parsing)
- `digitrec_core.*`: deterministic ML logic

## Data flow

1. Browser draws to a canvas (280×280) and downsamples to 28×28 in JS.
2. HTMX posts the pixel list to `/ui/predict`.
3. Django calls the `Predictor`, renders an HTML fragment (probs + hidden layers).
4. The same predictor is also available through `/api/predict` as JSON.

## Security notes

- No hardcoded secrets.
- CSRF is enabled for HTML/HTMX POST routes.
- JSON endpoint validates input strictly (length and range).

