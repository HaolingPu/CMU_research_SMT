#!/usr/bin/env bash
# Resource flags are supplied by submit_external_second_half.sh so this file is
# portable across clusters with different partition and GPU naming conventions.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/run_external_ambiguity_worker.sh"
