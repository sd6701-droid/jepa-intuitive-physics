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
    # Drive folder: https://drive.google.com/drive/folders/1F_9R1zLtAMQ7N_IIIio6HjEBkGuuMX4M
    # NOTE: videos.zip in that folder ships level1 ONLY (ColorBlack, ShapeBall,
    # ...). This eval needs level2 (Collision, Continuity, Gravity, ...), whose
    # clips live in the sibling P_*/IP_* Drive subfolders, so the whole folder
    # has to be mirrored and then reassembled into GRASP/level2/.
    L2="$DATA_ROOT/GRASP/level2"
    SRC="$DATA_ROOT/GRASP/drive_root"
    NDIRS=$(ls "$L2" 2>/dev/null | wc -l)
    if [[ "$NDIRS" -ge 32 ]]; then
      echo "GRASP already assembled: $NDIRS dirs in $L2"
    else
      command -v gdown >/dev/null 2>&1 || { echo "gdown missing: pip install --user gdown"; exit 1; }
      mkdir -p "$SRC"
      # Drive throttles folder pulls; gdown resumes, so just retry the whole call.
      for attempt in 1 2 3 4 5; do
        echo "--- gdown attempt $attempt ---"
        gdown --folder -O "$SRC" \
          https://drive.google.com/drive/folders/1F_9R1zLtAMQ7N_IIIio6HjEBkGuuMX4M && break
        echo "attempt $attempt failed, sleeping 120s"; sleep 120
      done

      # Collect the property dirs wherever gdown nested them and flatten to level2/.
      mkdir -p "$L2"
      mapfile -t PDIRS < <(find "$SRC" -maxdepth 4 -type d \( -name 'P_*' -o -name 'IP_*' \) | sort)
      echo "found ${#PDIRS[@]} property dirs under $SRC"
      for d in "${PDIRS[@]}"; do
        [[ -d "$L2/$(basename "$d")" ]] || mv "$d" "$L2/"
      done

      # level1 and the zips are not used by evals/intuitive_physics.
      rm -rf "$SRC" "$DATA_ROOT/GRASP/videos"
      NDIRS=$(ls "$L2" 2>/dev/null | wc -l)
    fi
    NMP4=$(find "$L2" -name '*.mp4' 2>/dev/null | wc -l)
    echo "GRASP: $NDIRS dirs, $NMP4 mp4 in $L2"
    [[ "$NDIRS" -ge 32 ]] || { echo "expected 32 dirs (16 P_ + 16 IP_)"; exit 1; }
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
