#!/usr/bin/env python3
"""Entry point: delegates to :mod:`consensus_decoding.runner`.

Existing SLURM scripts invoke this script directly by path; the package
underneath holds the actual implementation, split into focused modules.
"""
from __future__ import annotations

from consensus_decoding import main


if __name__ == "__main__":
    main()
