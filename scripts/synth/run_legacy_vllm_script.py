#!/usr/bin/env python3
"""Run a legacy vLLM script against the current structured-output API."""

from __future__ import annotations

import functools
import runpy
import sys
from pathlib import Path


def install_guided_decoding_compatibility() -> None:
    import vllm
    import vllm.sampling_params as sampling_params

    if hasattr(sampling_params, "GuidedDecodingParams"):
        return

    sampling_params.GuidedDecodingParams = sampling_params.StructuredOutputsParams
    original_init = sampling_params.SamplingParams.__init__

    @functools.wraps(original_init)
    def compatible_init(self, *args, guided_decoding=None, **kwargs):
        if guided_decoding is not None:
            if kwargs.get("structured_outputs") is not None:
                raise TypeError("guided_decoding and structured_outputs are mutually exclusive")
            kwargs["structured_outputs"] = guided_decoding
        original_init(self, *args, **kwargs)

    sampling_params.SamplingParams.__init__ = compatible_init
    vllm.SamplingParams = sampling_params.SamplingParams


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_legacy_vllm_script.py SCRIPT [ARGS ...]")

    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        raise SystemExit(f"legacy script not found: {script}")

    install_guided_decoding_compatibility()
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
