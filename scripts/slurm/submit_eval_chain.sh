#!/usr/bin/env bash
# Queue several eval configs IN ORDER. Each config is submitted on both accounts
# (via submit_eval_dual.sh: first to start wins, the other is cancelled) and each
# pair depends on the previous pair, so runs execute one at a time in the order
# given and never pile up against the per-user GPU quota.
#
#   bash scripts/slurm/submit_eval_chain.sh grasp/student_salt grasp/student_causal_proposed inflevel/random
#   bash scripts/slurm/submit_eval_chain.sh --time=10:00:00 grasp/student_salt inflevel/student_salt
#
# Config names are relative to evals/intuitive_physics/configs/ (".yaml" optional).
# Options starting with "--" are passed to sbatch for every job (e.g. --time).
set -euo pipefail
REPO=/scratch/sd6701/jepa-intuitive-physics
SBATCH_ARGS=(); CONFIGS=()
for a in "$@"; do
  if [[ "$a" == --* ]]; then SBATCH_ARGS+=("$a"); else CONFIGS+=("${a%.yaml}"); fi
done
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "usage: $0 [--sbatch-opts] <dataset>/<model> ..."; exit 1; }

PREV=""
for c in "${CONFIGS[@]}"; do
  CFG="evals/intuitive_physics/configs/$c.yaml"
  [[ -f "$REPO/evaluation_code/$CFG" ]] || { echo "no such config: $CFG"; exit 1; }
  NAME="${c//\//_}"                       # grasp/student_salt -> grasp_student_salt
  DEP=(); [[ -n "$PREV" ]] && DEP=(--dependency="afterany:${PREV}")
  echo "### $c  (job-name $NAME${PREV:+, after $PREV})"
  OUT=$(CONFIG="$CFG" bash "$REPO/scripts/slurm/submit_eval_dual.sh" --job-name="$NAME" ${DEP[@]+"${DEP[@]}"} ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"})
  echo "$OUT" | grep -E "queued job|to cancel this pair"
  IDS=$(echo "$OUT" | sed -n 's/^queued job \([0-9]*\) on .*/\1/p' | paste -sd: -)
  PREV="$IDS"
done
echo
squeue --me -o "%.10i %.28j %.30a %.8T %.8M %.22E %R"
