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
#   CONFIG=evals/intuitive_physics/configs/student_grasp.yaml sbatch --time=06:00:00 scripts/slurm/eval_intphys.sbatch
set -euo pipefail

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
python -c "import torch, decord; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

python -m evals.main --devices $DEVICES --fname "$CONFIG"
