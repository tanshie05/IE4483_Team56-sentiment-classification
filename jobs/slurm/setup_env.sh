#!/bin/bash
# Optional cluster helper (Slurm). Builds the project's Conda environment on a CPU node (many clusters forbid installs on
# login nodes). Run from the repository root:  mkdir -p logs && sbatch jobs/slurm/setup_env.sh
# Installs from PyPI, or from a folder of pre-downloaded wheels (with a SHA256SUMS file) if WHEELS_DIR is set.
#SBATCH --job-name=ie4483-setup
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# ---- Edit these for your cluster -------------------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/path/to/your/project/storage}   # large scratch/project storage for caches and the environment
ENV_PATH=${ENV_PATH:-$PROJECT_DIR/envs/ie4483}               # where the Conda environment is created
CONDA_MODULE=${CONDA_MODULE:-Miniforge3}                      # environment module that provides conda ("" to skip)
WHEELS_DIR=${WHEELS_DIR:-}                                    # optional offline wheel folder; empty = install from PyPI
MIN_FREE_GB=${MIN_FREE_GB:-20}
# -----------------------------------------------------------------------------------------------------------------------

set -eo pipefail
export TMPDIR=$PROJECT_DIR/.tmp/$SLURM_JOB_ID
export PIP_CACHE_DIR=$PROJECT_DIR/.tmp/pip CONDA_PKGS_DIRS=$PROJECT_DIR/.tmp/conda XDG_CACHE_HOME=$PROJECT_DIR/.tmp/cache \
       TORCH_HOME=$PROJECT_DIR/.tmp/torch HF_HOME=$PROJECT_DIR/.tmp/hf
mkdir -p "$TMPDIR"

[ -z "$CONDA_MODULE" ] || module load "$CONDA_MODULE"
trap 'conda clean -a -y; rm -rf "$TMPDIR"' EXIT  # Leaves no conda cache, even if the install fails.
PROJECT_DIR=$PROJECT_DIR bash jobs/slurm/check_space.sh "$MIN_FREE_GB"
conda create -y -p "$ENV_PATH" python=3.13
source activate "$ENV_PATH"
if [ -n "$WHEELS_DIR" ]; then
    (cd "$WHEELS_DIR" && sha256sum --quiet -c SHA256SUMS)  # Stops here if the wheels are missing or damaged.
    pip install --no-cache-dir --no-index --find-links "$WHEELS_DIR" -r requirements.txt
else
    pip install --no-cache-dir -r requirements.txt
fi

python - <<'PYEOF'
import numpy, sklearn, sys, torch
print('python', sys.version.split()[0])
print('torch', torch.__version__, 'cuda build', torch.version.cuda)
print('numpy', numpy.__version__, 'scikit-learn', sklearn.__version__)
PYEOF
du -sh "$ENV_PATH"
