#!/usr/bin/env bash
# Verifies the intuitive-physics eval datasets are complete and laid out the way
# evals/intuitive_physics expects. Structural, not just "did bytes arrive":
# it walks the same paths the dataset classes walk.
#   IntPhys : <root>/IntPhys/dev/O{1,2,3}/<scene>/{1,2,3,4}/{scene/*.png,status.json}
#   GRASP   : <root>/GRASP/level2/{P_,IP_}<Property>/<scene>.mp4
#   InfLevel: <root>/inflevel_lab/{continuity,gravity,solidity}/*.mp4, checked
#             against the CSV manifests in auxiliary_data_loading_files/inflevel/
# Usage: bash scripts/check_datasets.sh [array_job_id]
set -uo pipefail
D="${INTPHYS_DATA_ROOT:-/scratch/sd6701/datasets}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUX="$REPO/evaluation_code/auxiliary_data_loading_files/inflevel"
JOBID="${1:-}"
ok=0; bad=0
pass() { echo "  OK   $*"; ok=$((ok+1)); }
fail() { echo "  MISS $*"; bad=$((bad+1)); }

echo "data root: $D"
echo
echo "=== download sentinels (.staging) ==="
if [[ -d "$D/.staging" ]]; then ls -1 "$D/.staging" | sed 's/\.done$/  complete/' | sed 's/^/  /'
else echo "  (none - no array task has finished yet)"; fi

echo
echo "=== disk usage ==="
du -sh "$D/IntPhys" "$D/GRASP" "$D/inflevel_lab" 2>/dev/null | sed 's/^/  /'
echo "  inodes under IntPhys: $(find "$D/IntPhys" -type f 2>/dev/null | wc -l)"

echo
echo "=== IntPhys ==="
for split in dev test; do
  for b in O1 O2 O3; do
    p="$D/IntPhys/$split/$b"
    [[ -d "$p" ]] || { echo "  --   $split/$b absent"; continue; }
    n=$(ls "$p" 2>/dev/null | wc -l)
    # a scene is well-formed when all 4 possibilities carry frames + status.json
    # (test scenes have no status.json - labels are held out for the leaderboard)
    broken=0
    while IFS= read -r s; do
      for poss in 1 2 3 4; do
        [[ -d "$p/$s/$poss/scene" ]] && [[ -n "$(ls -A "$p/$s/$poss/scene" 2>/dev/null)" ]] || { broken=$((broken+1)); break; }
        [[ "$split" == "test" || -f "$p/$s/$poss/status.json" ]] || { broken=$((broken+1)); break; }
      done
    done < <(ls "$p" 2>/dev/null)
    if [[ "$broken" -eq 0 && "$n" -gt 0 ]]; then pass "$split/$b  $n scenes"
    else fail "$split/$b  $n scenes, $broken incomplete"; fi
  done
done

echo
echo "=== GRASP (level2) ==="
PROPS="Collision Continuity Gravity GravityContinuity GravityInertia GravityInertia2 \
GravitySupport Inertia Inertia2 ObjectPermanence ObjectPermanence2 ObjectPermanence3 \
SolidityContinuity SolidityContinuity2 Unchangeableness Unchangeableness2"
if [[ -d "$D/GRASP/level2" ]]; then
  for prop in $PROPS; do
    np=$(ls "$D/GRASP/level2/P_$prop"/*.mp4  2>/dev/null | wc -l)
    ni=$(ls "$D/GRASP/level2/IP_$prop"/*.mp4 2>/dev/null | wc -l)
    # eval pairs P_ with IP_ by index, so the two counts must match and be nonzero
    if [[ "$np" -gt 0 && "$np" -eq "$ni" ]]; then pass "$prop  $np possible / $ni impossible"
    else fail "$prop  $np possible / $ni impossible"; fi
  done
else fail "GRASP/level2 absent"; fi

echo
echo "=== InfLevel-lab (vs CSV manifests) ==="
for prop in continuity gravity solidity; do
  csv="$AUX/$prop.csv"
  [[ -f "$csv" ]] || { fail "$prop  manifest $csv not found"; continue; }
  miss=0; tot=0
  while IFS= read -r rel; do
    tot=$((tot+1)); [[ -f "$D/inflevel_lab/$rel" ]] || miss=$((miss+1))
  done < <(awk -F, 'NR>1 && NF>6 {print $1; print $7}' "$csv")
  if [[ "$miss" -eq 0 && "$tot" -gt 0 ]]; then pass "$prop  $tot/$tot videos"
  else fail "$prop  $((tot-miss))/$tot videos ($miss missing)"; fi
done

if [[ -n "$JOBID" ]]; then
  echo
  echo "=== slurm array $JOBID ==="
  sacct -j "$JOBID" --format=JobID%16,JobName%14,State%12,Elapsed,MaxRSS,ExitCode | sed 's/^/  /'
fi

echo
echo "checks passed: $ok   failed: $bad"
[[ "$bad" -eq 0 ]]
