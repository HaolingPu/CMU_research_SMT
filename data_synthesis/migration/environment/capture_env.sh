#!/bin/bash
# Capture Babel software environment for migration documentation.
# Safe to re-run; writes into the directory containing this script.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

{ uname -a; echo; cat /etc/os-release 2>/dev/null; echo; hostname; } > system_info.txt

source /home/haolingp/miniconda3/etc/profile.d/conda.sh

{ python --version; pip --version 2>/dev/null; } > python_version.txt 2>&1

nvidia-smi > gpu_info.txt 2>&1 || echo "no GPU on this node ($(hostname))" > gpu_info.txt
{ which nvcc && nvcc --version; } >> gpu_info.txt 2>&1 || echo "nvcc not on PATH" >> gpu_info.txt
{ module list; } > modules.txt 2>&1 || echo "no environment modules" > modules.txt

conda env list > conda_env_list.txt

# Per-env captures for the envs the project's scripts actually activate
for env in evaluation segale SMT metricx vllm gemma4 fbk consensus; do
  prefix=$(conda env list | awk -v e="$env" '$1==e {print $NF}')
  if [ -z "$prefix" ]; then echo "env $env: NOT FOUND under this user" >> missing_envs.txt; continue; fi
  echo "== $env ($prefix) =="
  conda list -p "$prefix" > "conda_list_${env}.txt" 2>&1 || true
  "$prefix/bin/pip" freeze > "pip_freeze_${env}.txt" 2>&1 || true
  "$prefix/bin/python" - <<'EOF' > "key_versions_${env}.txt" 2>&1 || true
import sys
print("python", sys.version)
for m in ["torch","transformers","vllm","simuleval","sacrebleu","comet","unbabel_comet","numpy","datasets","peft","deepspeed","ms_swift","swift"]:
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception:
        pass
try:
    import torch
    print("torch.cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list())
except Exception as e:
    print("torch cuda info unavailable:", e)
EOF
done

# Env vars: names only for anything secret-looking, values for safe infra vars
env | sort | awk -F= '
  /KEY|TOKEN|SECRET|PASS|CRED|AUTH/ {print $1"=<REDACTED>"; next}
  /^(PATH|PYTHONPATH|LD_LIBRARY_PATH|CUDA|HF_|HUGGINGFACE|TRANSFORMERS|TORCH|SLURM_CLUSTER|CONDA|WANDB_DIR|TMPDIR|XDG)/ {print}
' > important_env_vars.txt

echo done
