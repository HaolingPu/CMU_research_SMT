# Generation container

This image reproduces the Python/vLLM environment used by the 40K ambiguity
consensus decode. It serves all three models used by each two-GPU worker:

- Qwen3.8-27B-FP8 and Gemma-4-E2B-it colocated on GPU 0.
- Qwen3.6-35B-A3B-FP8 on GPU 1.

The image contains software only. Model weights, the frozen input TSV, outputs,
caches, and credentials stay on the cluster filesystem and are mounted at run
time.

## Reproducibility pins

- Base: NVIDIA CUDA 12.9.1 cuDNN runtime, pinned by OCI digest in `Dockerfile`.
- Python: 3.12.
- Packages: `data_synthesis/migration/environment/pip_freeze_gemma4.txt`.
- Compatibility substitution: `apache-tvm-ffi` 0.1.10 replaces the unpublished
  0.1.10rc2 capture; all other package versions remain unchanged.
- vLLM: commit `8617f8676b5ae936382ea00fa92693f59fbb9d69`,
  installed from its immutable per-commit wheel.
- Architecture: `linux/amd64`; the tested GPUs are NVIDIA L40S.

## Publish through GitHub

Pushing a change to this directory on `feature/home-checkout` runs
`.github/workflows/generation-image.yml`. The workflow publishes:

```text
ghcr.io/haolingpu/cmu-research-smt-generation:git-<full-git-commit>
ghcr.io/haolingpu/cmu-research-smt-generation:feature-home-checkout
```

Use the `git-<commit>` tag for a reproducible run. The workflow summary also
prints an immutable `docker://...@sha256:...` reference; record that reference
in the experiment manifest. Do not use the moving branch tag for a final run.

GitHub Container Registry packages are private unless their package visibility
is changed. Either make this package public in GitHub's package settings, or
have the mentor authenticate with a personal access token containing
`read:packages`:

```bash
echo "$GHCR_TOKEN" | apptainer registry login \
  --username <github-user> --password-stdin docker://ghcr.io
```

Do not put the token in a Slurm script, Git repository, or run manifest.

## Cache as a portable SIF

After copying the immutable reference from the workflow summary, pull it once
to a shared filesystem on the mentor's cluster:

```bash
IMAGE='docker://ghcr.io/haolingpu/cmu-research-smt-generation@sha256:<digest>'
apptainer pull /shared/containers/ambiguity-generation.sif "$IMAGE"
apptainer exec --nv /shared/containers/ambiguity-generation.sif \
  python /opt/verify_runtime.py
```

A `.sif` stored only in Haoling's private directory is not portable by itself.
It works elsewhere only after the file is copied there or both users can read
the same shared filesystem. GHCR is the distribution mechanism; the `.sif` is
the cluster-local cache.

## Local validation

On a machine with Docker:

```bash
docker build --platform linux/amd64 \
  -f containers/generation/Dockerfile \
  -t ambiguity-generation:test .
docker run --rm ambiguity-generation:test \
  python /opt/verify_runtime.py --no-cuda
```

GPU validation should happen in a one-task Slurm smoke test before launching a
large array.
