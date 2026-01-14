#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    """Django manage.py belépési pont."""

    # A repo gyökér /src legyen import path-on.
    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"
    sys.path.insert(0, str(src_root))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "digitrec_web.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
