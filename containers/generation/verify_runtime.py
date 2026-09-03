#!/usr/bin/env python3
"""Fail fast when the generation container differs from the tested runtime."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys


EXPECTED = {
    "numpy": "2.2.6",
    "openai": "2.30.0",
    "torch": "2.10.0+cu129",
    "transformers": "5.5.0",
    "vllm": "0.19.1rc1.dev28+g8617f8676",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Skip the host GPU check while building the image.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    print(f"python={sys.version.split()[0]}")
    if not sys.version.startswith("3.12."):
        failures.append(f"expected Python 3.12, found {sys.version.split()[0]}")

    for package, expected in EXPECTED.items():
        actual = importlib.metadata.version(package)
        print(f"{package}={actual}")
        if actual != expected:
            failures.append(f"{package}: expected {expected}, found {actual}")

    for module in ("torch", "transformers", "vllm"):
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - exercised in the image build
            failures.append(f"cannot import {module}: {exc}")

    if not args.no_cuda:
        import torch

        print(f"torch.cuda.version={torch.version.cuda}")
        print(f"torch.cuda.available={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            failures.append("CUDA is unavailable; run the container with Apptainer --nv")

    if failures:
        print("Runtime verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
