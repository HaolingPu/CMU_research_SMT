#!/usr/bin/env python3
"""Baseline2 entrypoint: pure code quorum-LCP (70%)."""

import os
import sys


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    core = os.path.join(script_dir, "llm_future_sampling_core.py")
    argv = [
        sys.executable,
        core,
        *sys.argv[1:],
        "--selection-mode",
        "lcp70_code",
        "--consensus-ratio",
        "0.7",
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
