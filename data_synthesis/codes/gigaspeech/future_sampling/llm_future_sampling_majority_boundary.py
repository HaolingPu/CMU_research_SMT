#!/usr/bin/env python3
"""Baseline4 entrypoint: majority-vote prefix with boundary rule."""

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
        "majority_vote",
        "--consensus-ratio",
        "0.6",
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
