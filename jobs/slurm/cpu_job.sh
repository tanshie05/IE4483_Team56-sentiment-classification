#!/bin/bash
# Optional cluster helper (Slurm). Runs one command on a CPU node; the job carries on after SSH disconnects.
# Run from the repository root (mkdir -p logs first):  sbatch jobs/slurm/cpu_job.sh python time_inference.py
#SBATCH --job-name=ie4483-cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# ---- Edit these for your cluster -------------------------------------------------------------------------------------
PROJECT_DIR=${PROJECT_DIR:-/path/to/your/project/storage}   # large scratch/project storage for caches and the environment
ENV_PATH=${ENV_PATH:-$PROJECT_DIR/envs/ie4483}               # Conda environment built by setup_env.sh
CONDA_MODULE=${CONDA_MODULE:-Miniforge3}                      # environment module that provides conda ("" to skip)
MIN_FREE_GB=${MIN_FREE_GB:-2}                                 # check_space.sh threshold
# -----------------------------------------------------------------------------------------------------------------------

set -eo pipefail
export TMPDIR=$PROJECT_DIR/.tmp/$SLURM_JOB_ID
export PIP_CACHE_DIR=$PROJECT_DIR/.tmp/pip CONDA_PKGS_DIRS=$PROJECT_DIR/.tmp/conda XDG_CACHE_HOME=$PROJECT_DIR/.tmp/cache \
       TORCH_HOME=$PROJECT_DIR/.tmp/torch HF_HOME=$PROJECT_DIR/.tmp/hf
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export PYTHONUNBUFFERED=1 PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1

[ -z "$CONDA_MODULE" ] || module load "$CONDA_MODULE"
PROJECT_DIR=$PROJECT_DIR bash jobs/slurm/check_space.sh "$MIN_FREE_GB"
source activate "$ENV_PATH"

hostname
"$@"
