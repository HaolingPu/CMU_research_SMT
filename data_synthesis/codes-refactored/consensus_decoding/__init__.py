"""Token-ID-level consensus decoding for streaming MT.

Two base models each generate future continuations of the observed source
prefix.  An instruct model provides next-token distributions for each
hypothesised full source, and the consensus intersection selects the
committed token.  Candidate sets are built via either top-k, min-p, or top-p.

Top-level entry point: ``consensus_decoding.runner.main``.
"""
from __future__ import annotations

from .runner import main

__all__ = ["main"]
