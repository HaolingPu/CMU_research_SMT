#!/usr/bin/env python3
"""Baseline1 entrypoint: pure code LCP (100%)."""

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
        "lcp_code",
    ]
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
