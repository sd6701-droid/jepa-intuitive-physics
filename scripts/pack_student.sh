#!/usr/bin/env bash
# Inspect the SALT student+predictor and teacher checkpoints, then pack them into
# the single file the eval expects. Run on the cluster (CPU is fine).
#
#   bash scripts/pack_student.sh              # inspect only
#   PACK=1 bash scripts/pack_student.sh       # inspect + pack
#
# Override the sub-dict keys once inspect shows them, e.g.:
#   PACK=1 STUDENT_KEY=encoder PRED_KEY=predictor TEACHER_KEY=encoder bash scripts/pack_student.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STUDENT="${STUDENT:-/scratch/sd6701/salt/p-salt/results/student_ssv2_salt_tube1/checkpoints/01266910/last/checkpoint.pth}"
TEACHER="${TEACHER:-/scratch/sd6701/salt/p-salt/results/teacher_ssv2/checkpoints/8f00b836/last/checkpoint.pth}"
OUT="${OUT:-/scratch/sd6701/jepa-intuitive-physics/checkpoints/student_ssv2_salt_tube1/student-latest.pth.tar}"
STUDENT_KEY="${STUDENT_KEY:-}"; PRED_KEY="${PRED_KEY:-}"; TEACHER_KEY="${TEACHER_KEY:-}"

cd "$REPO/evaluation_code"
echo "===== STUDENT (+predictor) ====="; python tools/inspect_checkpoint.py "$STUDENT"
echo; echo "===== TEACHER ====="; python tools/inspect_checkpoint.py "$TEACHER"

if [[ "${PACK:-0}" == "1" ]]; then
  echo; echo "===== PACK ====="
  python tools/pack_student_checkpoint.py \
    --student "$STUDENT"   ${STUDENT_KEY:+--student-key "$STUDENT_KEY"} \
    --predictor "$STUDENT" ${PRED_KEY:+--predictor-key "$PRED_KEY"} \
    --teacher "$TEACHER"   ${TEACHER_KEY:+--teacher-key "$TEACHER_KEY"} \
    --out "$OUT"
fi
