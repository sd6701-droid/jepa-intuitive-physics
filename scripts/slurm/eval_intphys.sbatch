#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --job-name=intphys_eval
#SBATCH --output=/scratch/sd6701/jepa-intuitive-physics/logs/eval/%x_%j.log
#SBATCH --error=/scratch/sd6701/jepa-intuitive-physics/logs/eval/%x_%j.err
#SBATCH --requeue                # permit self-requeue after a crash
#SBATCH --open-mode=append       # requeued attempt appends to the SAME %j log -- required
# Runs evals/intuitive_physics on one H100. World size (= #GPUs) must divide 30
# for IntPhys; with 1 GPU every rank sees all 30 scenes per block. ViT-S is
# small enough that IntPhys finishes comfortably in the 2 h window; raise
# --time for GRASP / InfLevel (more, longer videos).
#
# Usage:
#   mkdir -p /scratch/sd6701/jepa-intuitive-physics/logs/eval
#   sbatch scripts/slurm/eval_intphys.sbatch                                   # student_intphys.yaml
#   bash scripts/slurm/submit_eval_dual.sh                                     # same, queued on 2 accounts
#   CONFIG=evals/intuitive_physics/configs/student_grasp.yaml sbatch --time=06:00:00 scripts/slurm/eval_intphys.sbatch
set -euo pipefail

# --- dual-account submission guard (see scripts/slurm/submit_eval_dual.sh) ---
# The submitter queues this script twice under two accounts and writes both job
# ids to $DUAL_GROUP_DIR/jobs. The first job to START takes an atomic lock
# (mkdir) and cancels its sibling; the loser exits 0. A requeued winner keeps its
# own lock (same SLURM_JOB_ID), so --requeue still works.
if [[ -n "${DUAL_GROUP_DIR:-}" ]]; then
  if mkdir "$DUAL_GROUP_DIR/lock" 2>/dev/null || [[ "$(cat "$DUAL_GROUP_DIR/winner" 2>/dev/null)" == "$SLURM_JOB_ID" ]]; then
    echo "$SLURM_JOB_ID" > "$DUAL_GROUP_DIR/winner"
    for _ in $(seq 1 30); do [[ -s "$DUAL_GROUP_DIR/jobs" ]] && break; sleep 2; done
    for j in $(cat "$DUAL_GROUP_DIR/jobs" 2>/dev/null); do
      [[ "$j" != "$SLURM_JOB_ID" ]] && { echo "dual: cancelling sibling job $j"; scancel "$j" || true; }
    done
    echo "dual: job $SLURM_JOB_ID (account ${SLURM_JOB_ACCOUNT:-?}) won the race"
  else
    echo "dual: sibling $(cat "$DUAL_GROUP_DIR/winner" 2>/dev/null) already running; exiting"
    exit 0
  fi
fi

export PATH="$HOME/.local/bin:$PATH"
for CONDA_SH in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                "/scratch/$USER/miniconda3/etc/profile.d/conda.sh"; do
  [[ -f "$CONDA_SH" ]] && { source "$CONDA_SH" && conda activate "${CONDA_ENV:-intphys}" || true; break; }
done

export WANDB_DIR="${WANDB_DIR:-/scratch/sd6701/jepa-intuitive-physics/logs/wandb}"; mkdir -p "$WANDB_DIR"
export INTPHYS_DATA_ROOT="${INTPHYS_DATA_ROOT:-/scratch/sd6701/datasets}"
REPO=/scratch/sd6701/jepa-intuitive-physics
CONFIG="${CONFIG:-evals/intuitive_physics/configs/student_intphys.yaml}"

cd "$REPO/evaluation_code"
NGPU=$(python -c "import torch;print(torch.cuda.device_count())")
DEVICES=$(seq -s ' ' -f 'cuda:%g' 0 $((NGPU-1)))
echo "host=$(hostname) gpus=$NGPU config=$CONFIG data=$INTPHYS_DATA_ROOT"
echo "python=$(which python) conda_env=${CONDA_DEFAULT_ENV:-none}"
# Preflight: fail fast (before touching the GPU) if the env is missing anything the eval imports.
python - <<'PY' || { echo "preflight failed: run  pip install -r requirements.txt  in env ${CONDA_DEFAULT_ENV:-?}"; exit 1; }
import importlib, sys
missing = [m for m in ("torch","torchvision","decord","einops","yaml","sklearn","scipy","pandas","timm","cv2","PIL","wandb","submitit","skimage")
           if importlib.util.find_spec(m) is None]
if missing:
    print("missing packages:", missing); sys.exit(1)
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
PY

python -m evals.main --devices $DEVICES --fname "$CONFIG"
