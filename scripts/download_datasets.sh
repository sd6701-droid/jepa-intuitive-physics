#!/usr/bin/env bash
# Downloads the evaluation datasets into $INTPHYS_DATA_ROOT (default: /scratch/sd6701/datasets).
# Usage: bash scripts/download_datasets.sh [intphys|grasp|inflevel|all]
set -euo pipefail
ROOT="${INTPHYS_DATA_ROOT:-/scratch/sd6701/datasets}"
WHAT="${1:-intphys}"
mkdir -p "$ROOT"
echo "Data root: $ROOT"

dl() { # url dest
  if command -v aria2c >/dev/null; then aria2c -x 8 -c -d "$(dirname "$2")" -o "$(basename "$2")" "$1";
  else curl -L -C - -o "$2" "$1"; fi
}

if [[ "$WHAT" == "intphys" || "$WHAT" == "all" ]]; then
  # IntPhys dev set (3 GB). Expected layout: IntPhys/dev/O{1,2,3}/<scene>/{1,2,3,4}/scene/*.png
  mkdir -p "$ROOT/IntPhys"
  dl https://download-intphys.cognitive-ml.fr/dev.tar.gz "$ROOT/IntPhys/dev.tar.gz"
  echo "6a72715007f2f3b0e9546bfa8d8fc39b  $ROOT/IntPhys/dev.tar.gz" | md5sum -c - || echo "WARNING: md5 mismatch"
  tar -xzf "$ROOT/IntPhys/dev.tar.gz" -C "$ROOT/IntPhys"
  ls "$ROOT/IntPhys/dev"
fi

if [[ "$WHAT" == "intphys-test" ]]; then
  # Only for leaderboard submission via evals/intphys_test (3 x 36 GB).
  mkdir -p "$ROOT/IntPhys/test"
  for B in O1 O2 O3; do
    dl "https://download-intphys.cognitive-ml.fr/test.$B.tar.gz" "$ROOT/IntPhys/test.$B.tar.gz"
    tar -xzf "$ROOT/IntPhys/test.$B.tar.gz" -C "$ROOT/IntPhys/test"
  done
fi

if [[ "$WHAT" == "grasp" || "$WHAT" == "all" ]]; then
  # GRASP videos are on Google Drive: https://drive.google.com/drive/folders/1F_9R1zLtAMQ7N_IIIio6HjEBkGuuMX4M
  # Download videos.zip manually (or with gdown), then:
  #   unzip videos.zip -d "$ROOT/GRASP"
  # Expected layout: GRASP/level2/P_<Property>/<scene>.mp4 and GRASP/level2/IP_<Property>/<scene>.mp4
  mkdir -p "$ROOT/GRASP"
  if [[ -f "$ROOT/GRASP/videos.zip" ]]; then unzip -q -o "$ROOT/GRASP/videos.zip" -d "$ROOT/GRASP"; ls "$ROOT/GRASP";
  else echo "GRASP: place videos.zip in $ROOT/GRASP (Google Drive link above) and re-run."; fi
fi

if [[ "$WHAT" == "inflevel" || "$WHAT" == "all" ]]; then
  # InfLevel-lab. Expected layout: inflevel_lab/{continuity,gravity,solidity}/*.mp4
  dl https://pub-7320908bcb5b4cdea63c22bc2a38600c.r2.dev/inflevel_lab.tar.gz "$ROOT/inflevel_lab.tar.gz"
  tar -xzf "$ROOT/inflevel_lab.tar.gz" -C "$ROOT"
  ls "$ROOT/inflevel_lab"
fi
