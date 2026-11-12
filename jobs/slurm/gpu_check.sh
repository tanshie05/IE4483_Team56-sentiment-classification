#!/bin/bash
# Optional cluster helper, sourced by train_gpu.sh. Checks each assigned GPU separately and keeps only the first healthy
# card visible, so all training runs on exactly one card. Stops the job if no card passes. Records the chosen card.

# ---- Edit these for your cluster -------------------------------------------------------------------------------------
REQUIRED_GPU_NAME=${REQUIRED_GPU_NAME:-}   # e.g. A6000: skip cards whose name lacks it (all results here used RTX A6000)
FAULTY_GPU_BUS=${FAULTY_GPU_BUS:-}         # PCI bus id of a known-bad card to skip, e.g. 00000000:00:1C.0
# -----------------------------------------------------------------------------------------------------------------------

gpu_fail() {
    echo "STOPPING: GPU sanity check failed on $(hostname): $1" >&2
    exit 1
}

nvidia-smi || gpu_fail "nvidia-smi did not run"
mapfile -t GPU_CARDS < <(nvidia-smi --query-gpu=index,uuid,pci.bus_id,name --format=csv,noheader)
[ "${#GPU_CARDS[@]}" -gt 0 ] || gpu_fail "no GPU is visible to this job"

CHOSEN_UUID=""
for card in "${GPU_CARDS[@]}"; do
    IFS=',' read -r index uuid bus name <<< "$card"
    index=$(echo "$index" | xargs) uuid=$(echo "$uuid" | xargs) bus=$(echo "$bus" | xargs) name=$(echo "$name" | xargs)
    echo "Checking card index=$index bus=$bus uuid=$uuid name=$name"
    if [[ -n "$FAULTY_GPU_BUS" && "${bus^^}" == "${FAULTY_GPU_BUS^^}" ]]; then
        echo "  skipped: this is the known faulty card ($FAULTY_GPU_BUS)"
        continue
    fi
    if nvidia-smi -i "$index" | grep -q 'ERR!'; then
        echo "  skipped: nvidia-smi shows ERR!"
        continue
    fi
    if [[ -n "$REQUIRED_GPU_NAME" && "$name" != *"$REQUIRED_GPU_NAME"* ]]; then
        echo "  skipped: not a $REQUIRED_GPU_NAME"
        continue
    fi
    if CUDA_VISIBLE_DEVICES="$uuid" python - <<'EOF'
import torch
assert torch.cuda.device_count() == 1, f'expected 1 visible GPU, got {torch.cuda.device_count()}'
x = torch.arange(6.0, device='cuda').reshape(2, 3)
assert (x @ x.T).sum().item() == 83.0
print('  torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'tiny CUDA operation OK')
EOF
    then
        CHOSEN_UUID=$uuid CHOSEN_BUS=$bus
        break
    fi
    echo "  skipped: the tiny PyTorch CUDA operation failed"
done
[ -n "$CHOSEN_UUID" ] || gpu_fail "none of the ${#GPU_CARDS[@]} assigned cards passed"

export CUDA_VISIBLE_DEVICES=$CHOSEN_UUID
python -c "import torch; assert torch.cuda.device_count() == 1, torch.cuda.device_count()" || gpu_fail "training would not see exactly one GPU"
echo "GPU used: node=$(hostname) slurm_job=$SLURM_JOB_ID bus=$CHOSEN_BUS uuid=$CHOSEN_UUID (1 of ${#GPU_CARDS[@]} assigned cards; training sees only this one)"
