"""Lightweight verbose logging utilities for the consensus pipeline."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional


def _vlog(f: Optional[Any], msg: str) -> None:
    if f is None:
        return
    line = str(msg) if msg.endswith("\n") else str(msg) + "\n"
    f.write(line)
    f.flush()


def _vlog_pretty_value(f: Optional[Any], label: str, value: Any) -> None:
    if f is None:
        return
    if isinstance(value, (list, dict)):
        pretty = json.dumps(value, ensure_ascii=False, indent=2)
        lines = pretty.splitlines()
        if not lines:
            _vlog(f, f"# {label}: []")
            return
        _vlog(f, f"# {label}: {lines[0]}")
        for line in lines[1:]:
            _vlog(f, f"#   {line}")
        return
    _vlog(f, f"# {label}: {value}")


class _TeeWriter:
    """Write to both a file and stdout (used only for --test-one)."""
    def __init__(self, fobj: Any):
        self._f = fobj
    def write(self, msg: str) -> None:
        self._f.write(msg); sys.stdout.write(msg)
    def flush(self) -> None:
        self._f.flush(); sys.stdout.flush()
    def close(self) -> None:
        self._f.close()


def write_pretty_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
