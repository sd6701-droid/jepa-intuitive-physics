"""Collapse the eval's per-block/per-context CSVs into the single headline number
per dataset used in the papers' comparison tables, plus their average.

The aggregation is the one in figures.ipynb (get_average_acc):

    for each Block: take the MAX of "Relative Accuracy (avg)" over that block's
    rows (the context lengths and the "Filtered" row), then average over blocks.

IntPhys averages 3 blocks (O1/O2/O3), GRASP 16 properties, InfLevel 3. The
"Avg" column is the unweighted mean of the three dataset scores.

Verified against the authors' bundled data (data_intphys.tar.gz): VideoMAEv2-g
reproduces as IntPhys 59.4 / Avg 58.4, matching the published row.

Usage:
    # point at the checkpoint folder; finds <ds>-<tag>/<write_tag>_r0.csv
    python tools/summarize_performance.py --root /path/to/checkpoints/student

    # or name the CSVs explicitly (dataset inferred from the path)
    python tools/summarize_performance.py a/intphys/performance.csv b/grasp/performance.csv

    # compare two runs side by side
    python tools/summarize_performance.py --root ckpt/vanilla --root ckpt/causal
"""
import argparse
import csv
import glob
import os
import statistics

KEY = "Relative Accuracy (avg)"
DATASETS = ("intphys", "grasp", "inflevel")
# how many blocks each dataset should contribute, as a completeness check
EXPECTED_BLOCKS = {"intphys": 3, "grasp": 16, "inflevel": 3}


def dataset_of(path):
    """Infer the dataset from the closest path component that names one.

    Scan from the filename outwards so an ancestor directory cannot win: a path
    like data_intphys/<model>/grasp/performance.csv is GRASP, not IntPhys.
    """
    for part in reversed(path.lower().replace("\\", "/").split("/")):
        for d in DATASETS:
            if d in part:
                return d
    return None


def score(path):
    """Return (score, n_blocks) for one performance CSV."""
    per_block = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            if KEY not in row or row[KEY] in (None, ""):
                raise SystemExit(f"{path}: no '{KEY}' column; got {list(row)[:6]}")
            per_block.setdefault(row["Block"], []).append(float(row[KEY]))
    if not per_block:
        raise SystemExit(f"{path}: no rows")
    return statistics.mean(max(v) for v in per_block.values()), len(per_block)


def find_csvs(root):
    """Locate one CSV per dataset under a checkpoint folder.

    eval.py writes <pretrain.folder>/intuitive_physics/<dataset>-<tag>/<write_tag>_r0.csv
    """
    found = {}
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        d = dataset_of(os.path.relpath(path, root))
        # rank>0 files duplicate rank 0's header-only output
        if d and not path.endswith(tuple(f"_r{i}.csv" for i in range(1, 64))):
            found.setdefault(d, path)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*", help="performance CSVs (dataset inferred from path)")
    ap.add_argument("--root", action="append", default=[],
                    help="checkpoint folder to search; repeatable to compare runs")
    a = ap.parse_args()

    runs = []  # (label, {dataset: path})
    for root in a.root:
        runs.append((os.path.basename(root.rstrip("/")), find_csvs(root)))
    if a.csvs:
        loose = {}
        for p in a.csvs:
            d = dataset_of(p)
            if d is None:
                raise SystemExit(f"{p}: cannot tell which dataset this is; "
                                 "put intphys/grasp/inflevel in the path")
            loose[d] = p
        runs.append(("cli", loose))
    if not runs:
        raise SystemExit("nothing to summarize: pass --root or CSV paths")

    w = max(len(label) for label, _ in runs) + 2
    print(f"{'run':<{w}}{'IntPhys':>9}{'GRASP':>8}{'InfLevel':>10}{'Avg':>8}")
    for label, paths in runs:
        cells, vals = [], []
        for d in DATASETS:
            if d not in paths:
                cells.append("--")
                continue
            s, n = score(paths[d])
            vals.append(s)
            # flag a partial run rather than reporting a score built on missing blocks
            cells.append(f"{s:.1f}" + ("" if n == EXPECTED_BLOCKS[d] else f"!{n}"))
        avg = f"{statistics.mean(vals):.1f}" if len(vals) == 3 else "--"
        print(f"{label:<{w}}{cells[0]:>9}{cells[1]:>8}{cells[2]:>10}{avg:>8}")
        for d in DATASETS:
            if d not in paths:
                print(f"{'':<{w}}  (no {d} CSV found)")

    print("\n'!n' marks a dataset whose CSV had n blocks instead of the expected "
          "3/16/3 -- an incomplete run, not a comparable score.")
    print("Avg is only printed when all three datasets are present.")


if __name__ == "__main__":
    main()
