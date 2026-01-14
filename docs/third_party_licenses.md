# Third-party licenses

This repository targets Apache-2.0 compatibility. The inventory below is best-effort and includes both runtime and dev-only Python dependencies.

| Dependency | Version | License | Scope | Notes |
|---|---:|---|---|---|
| Django | 5.1.5 | BSD-3-Clause | runtime | Web framework |
| torch | (platform-specific) | BSD-3-Clause | runtime | ML runtime (CPU/CUDA wheels vary by platform) |
| torchvision | (platform-specific) | BSD-3-Clause | runtime | Datasets/utilities (MNIST download/cache) |
| HTMX | (CDN) | BSD-2-Clause | runtime | Loaded from CDN; not vendored into the repo |
| pytest | (range pinned) | MIT | dev | Test runner |

If you add or remove dependencies, update this file.
