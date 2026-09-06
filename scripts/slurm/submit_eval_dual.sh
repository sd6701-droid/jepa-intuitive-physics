#!/usr/bin/env bash
# Queue the eval on two SLURM accounts at once; whichever starts first runs and
# cancels the other (guard lives at the top of eval_intphys.sbatch).
#
#   bash scripts/slurm/submit_eval_dual.sh                       # student_intphys.yaml
#   CONFIG=evals/intuitive_physics/configs/student_grasp.yaml \
#     bash scripts/slurm/submit_eval_dual.sh --time=06:00:00     # extra args go to sbatch
#
# Accounts can be overridden: ACCOUNTS="acc1 acc2" bash scripts/slurm/submit_eval_dual.sh
set -euo pipefail
REPO=/scratch/sd6701/jepa-intuitive-physics
ACCOUNTS="${ACCOUNTS:-torch_pr_230_tandon_priority torch_pr_230_tandon_advanced}"
CONFIG="${CONFIG:-evals/intuitive_physics/configs/student_intphys.yaml}"
GROUP="$(date +%Y%m%d_%H%M%S)_$$"
GROUPDIR="$REPO/logs/eval/dual/$GROUP"
mkdir -p "$GROUPDIR" "$REPO/logs/eval" "$REPO/logs/wandb"

JOBS=()
for ACC in $ACCOUNTS; do
  JID=$(sbatch --parsable --account="$ACC" \
        --export=ALL,DUAL_GROUP_DIR="$GROUPDIR",CONFIG="$CONFIG" \
        "$@" "$REPO/scripts/slurm/eval_intphys.sbatch")
  JID="${JID%%;*}"
  echo "queued job $JID on $ACC"
  JOBS+=("$JID")
done
echo "${JOBS[*]}" > "$GROUPDIR/jobs"
echo "group dir: $GROUPDIR"
echo "cancel both: scancel ${JOBS[*]}"
squeue -u "$USER" -o "%.10i %.20a %.12P %.10T %.10M %R" -j "$(IFS=,; echo "${JOBS[*]}")"
