# syntax=docker/dockerfile:1.7

# Pin the CUDA base by digest so rebuilding this Git commit starts from the same
# userspace. NVIDIA drivers still come from the host through Apptainer --nv.
FROM docker.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04@sha256:d02c4310b6d57ca0b16cd80298bdb33a74187baafe2eccd8a6a16180ddc90802

ARG DEBIAN_FRONTEND=noninteractive
ARG VLLM_COMMIT=8617f8676b5ae936382ea00fa92693f59fbb9d69
ARG VLLM_VERSION=0.19.1rc1.dev28+g8617f8676
ARG VLLM_WHEEL_URL=https://wheels.vllm.ai/8617f8676b5ae936382ea00fa92693f59fbb9d69/vllm-0.19.1rc1.dev28%2Bg8617f8676-cp38-abi3-manylinux_2_31_x86_64.whl

LABEL org.opencontainers.image.source="https://github.com/HaolingPu/CMU_research_SMT"
LABEL org.opencontainers.image.description="Pinned vLLM runtime for ambiguity-consensus trajectory generation"
LABEL org.opencontainers.image.vllm.commit="${VLLM_COMMIT}"
LABEL org.opencontainers.image.vllm.version="${VLLM_VERSION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        lsof \
        procps \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/generation \
    && /opt/generation/bin/pip install --no-cache-dir pip==26.0.1

# This capture came from the exact gemma4 environment used by the production
# 40K run. Dependencies are installed without re-resolution because that tested
# environment intentionally uses Transformers 5 with a vLLM wheel whose
# conservative metadata still declares Transformers <5. The only substituted
# package is an unpublished apache-tvm-ffi prerelease; 0.1.10 is ABI-compatible.
COPY data_synthesis/migration/environment/pip_freeze_gemma4.txt /tmp/pip-freeze.txt
RUN sed -e '/^vllm==/d' \
        -e 's/^apache-tvm-ffi==0.1.10rc2$/apache-tvm-ffi==0.1.10/' \
        /tmp/pip-freeze.txt > /tmp/requirements.txt \
    && /opt/generation/bin/pip install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        -r /tmp/requirements.txt \
    && /opt/generation/bin/pip install --no-cache-dir --no-deps \
        "${VLLM_WHEEL_URL}" \
    && rm -f /tmp/pip-freeze.txt /tmp/requirements.txt

COPY containers/generation/verify_runtime.py /opt/verify_runtime.py
RUN /opt/generation/bin/python /opt/verify_runtime.py --no-cuda

ENV PATH="/opt/generation/bin:${PATH}" \
    PYTHONNOUSERSITE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface/hub

WORKDIR /workspace
CMD ["bash"]
