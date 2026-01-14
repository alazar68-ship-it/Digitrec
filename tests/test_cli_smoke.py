from __future__ import annotations

from pathlib import Path

from digitrec_cli.cli import ExitCode, main


def test_cli_licenses_writes_file(tmp_path: Path) -> None:
    out = tmp_path / "third_party_licenses.md"
    code = main(["licenses", "--output", str(out)])
    assert code == ExitCode.OK
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Third-party licenses" in text
