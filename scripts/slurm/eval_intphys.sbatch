#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --gres=gpu:3
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_230_tandon_advanced
#SBATCH --job-name=intphys_eval
#SBATCH --output=/scratch/sd6701/jepa-intuitive-physics/logs/eval/%x_%j.log
#SBATCH --error=/scratch/sd6701/jepa-intuitive-physics/logs/eval/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
# Runs evals/intuitive_physics on one node. World size (= #GPUs) must divide 30
# for IntPhys (1, 2, 3, 5, 6). 3 GPUs -> 10 scenes per rank per block.
#
# Usage:
#   mkdir -p /scratch/sd6701/jepa-intuitive-physics/logs/eval
#   sbatch scripts/slurm/eval_intphys.sbatch                                   # student_intphys.yaml
#   CONFIG=evals/intuitive_physics/configs/student_grasp.yaml sbatch scripts/slurm/eval_intphys.sbatch
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
