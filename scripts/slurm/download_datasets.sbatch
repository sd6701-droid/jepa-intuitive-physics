#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_230_tandon_advanced
#SBATCH --job-name=intphys_dl
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/sd6701/jepa-intuitive-physics/logs/data/dl_%A_%a.log
#SBATCH --error=/scratch/sd6701/jepa-intuitive-physics/logs/data/dl_%A_%a.err
#SBATCH --requeue
#SBATCH --open-mode=append
# CPU-only: no --gres. Download+extract is IO/CPU bound (gz decode), so 8 cores
# and 16 G is plenty; the cost here is wall clock, not compute.
#
# One array task per dataset file. There are only 6 independent files, so the
# array is 0-5. Intra-file parallelism comes from aria2c's -x/-s range splitting
# (8 connections per task), not from more SLURM jobs.
#
#   0 intphys-dev      ~3 GB    <- needed for evals/intuitive_physics
#   1 inflevel         ~GB      <- needed for evals/intuitive_physics
#   2 grasp            ~GB      <- needed for evals/intuitive_physics (manual, see below)
#   3 intphys-test-O1  ~36 GB   <- only for evals/intphys_test leaderboard submission
#   4 intphys-test-O2  ~36 GB
#   5 intphys-test-O3  ~36 GB
#
# Usage:
#   mkdir -p /scratch/sd6701/jepa-intuitive-physics/logs/data
#
# Datasets land in /scratch/sd6701/datasets alongside sceneflow/ssv2/etc:
#   /scratch/sd6701/datasets/IntPhys/dev/O{1,2,3}/...
#   /scratch/sd6701/datasets/GRASP/level2/...
#   /scratch/sd6701/datasets/inflevel_lab/{continuity,gravity,solidity}/...
#   sbatch --array=0-2 scripts/slurm/download_datasets.sbatch   # eval datasets only
#   sbatch --array=3-5 scripts/slurm/download_datasets.sbatch   # test set too
#   sbatch            scripts/slurm/download_datasets.sbatch   # everything

set -euo pipefail

DATA_ROOT="${INTPHYS_DATA_ROOT:-/scratch/sd6701/datasets}"
STAGE="$DATA_ROOT/.staging"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"   # 1 = keep tarballs after extraction
CONNS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$DATA_ROOT" "$STAGE"

NAMES=(intphys-dev inflevel grasp intphys-test-O1 intphys-test-O2 intphys-test-O3)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
UNIT="${NAMES[$IDX]:-}"
[[ -n "$UNIT" ]] || { echo "no unit for array index $IDX"; exit 1; }

DONE="$STAGE/$UNIT.done"
echo "[$(date -Is)] unit=$UNIT idx=$IDX root=$DATA_ROOT host=$(hostname)"
if [[ -f "$DONE" ]]; then echo "already complete, nothing to do"; exit 0; fi

# --- helpers ---------------------------------------------------------------
# Both paths resume a partial file, so a --requeue restart picks up where it
# left off instead of starting the 36 GB transfer over.
dl() { # url dest
  local url="$1" out="$2"
  mkdir -p "$(dirname "$out")"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x "$CONNS" -s "$CONNS" -k 10M \
           --max-tries=0 --retry-wait=15 --summary-interval=120 \
           -d "$(dirname "$out")" -o "$(basename "$out")" "$url"
  else
    curl -fL -C - --retry 100 --retry-delay 15 --retry-all-errors -o "$out" "$url"
  fi
}

untar() { # tarball dest
  local tgz="$1" dest="$2"
  mkdir -p "$dest"
  if command -v pigz >/dev/null 2>&1; then tar -I pigz -xf "$tgz" -C "$dest"
  else tar -xzf "$tgz" -C "$dest"; fi
}

drop() { # tarball -- free the archive unless KEEP_ARCHIVES=1
  [[ "$KEEP_ARCHIVES" == "1" ]] || rm -f "$1"
}

# --- units -----------------------------------------------------------------
case "$UNIT" in

  intphys-dev)
    TGZ="$DATA_ROOT/IntPhys/dev.tar.gz"
    dl https://download-intphys.cognitive-ml.fr/dev.tar.gz "$TGZ"
    echo "6a72715007f2f3b0e9546bfa8d8fc39b  $TGZ" | md5sum -c - || echo "WARNING: md5 mismatch"
    untar "$TGZ" "$DATA_ROOT/IntPhys"
    drop "$TGZ"
    ls "$DATA_ROOT/IntPhys/dev"
    ;;

  inflevel)
    TGZ="$DATA_ROOT/inflevel_lab.tar.gz"
    dl https://pub-7320908bcb5b4cdea63c22bc2a38600c.r2.dev/inflevel_lab.tar.gz "$TGZ"
    untar "$TGZ" "$DATA_ROOT"
    drop "$TGZ"
    ls "$DATA_ROOT/inflevel_lab"
    ;;

  grasp)
    # Google Drive folder: https://drive.google.com/drive/folders/1F_9R1zLtAMQ7N_IIIio6HjEBkGuuMX4M
    # Drive rate-limits scripted folder pulls, so this tries gdown and otherwise
    # tells you to stage videos.zip by hand. Expected layout after unzip:
    #   GRASP/level2/{P_,IP_}<Property>/<scene>.mp4
    mkdir -p "$DATA_ROOT/GRASP"
    ZIP="$DATA_ROOT/GRASP/videos.zip"
    if [[ ! -f "$ZIP" ]] && command -v gdown >/dev/null 2>&1; then
      gdown --folder --remaining-ok -O "$DATA_ROOT/GRASP" \
        https://drive.google.com/drive/folders/1F_9R1zLtAMQ7N_IIIio6HjEBkGuuMX4M || true
    fi
    if [[ -f "$ZIP" ]]; then
      unzip -q -o "$ZIP" -d "$DATA_ROOT/GRASP"
      drop "$ZIP"
      ls "$DATA_ROOT/GRASP"
    elif [[ -d "$DATA_ROOT/GRASP/level2" ]]; then
      echo "GRASP already extracted at $DATA_ROOT/GRASP/level2"
    else
      echo "GRASP: gdown unavailable or Drive refused. Download videos.zip in a"
      echo "browser, copy it to $ZIP, then re-run: sbatch --array=2 scripts/slurm/download_datasets.sbatch"
      exit 1
    fi
    ;;

  intphys-test-O*)
    B="${UNIT##*-}"          # O1 / O2 / O3
    TGZ="$DATA_ROOT/IntPhys/test.$B.tar.gz"
    dl "https://download-intphys.cognitive-ml.fr/test.$B.tar.gz" "$TGZ"
    untar "$TGZ" "$DATA_ROOT/IntPhys/test"
    drop "$TGZ"
    ls "$DATA_ROOT/IntPhys/test"
    ;;

  *) echo "unknown unit: $UNIT"; exit 1 ;;
esac

touch "$DONE"
echo "[$(date -Is)] unit=$UNIT complete"
