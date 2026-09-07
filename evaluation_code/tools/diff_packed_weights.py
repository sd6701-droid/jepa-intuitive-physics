"""Compare a packed checkpoint's tensors against the p-salt sources they came
from, element by element.

verify_pack.py answers "which file did this come from" with a hash; this answers
the stricter question "is every tensor bit-identical to its source", and says
where it is not. It compares:

    packed[encoder]        vs  <student ckpt>[encoder]
    packed[predictor]      vs  <student ckpt>[predictor]
    packed[target_encoder] vs  <teacher ckpt>[encoder]

after stripping the DDP 'module.' prefix the way pack_student_checkpoint.py and
load_pretrained both do, so p-salt's 'module.backbone.X' lines up with the
packed 'backbone.X'.

Usage:
    cd evaluation_code
    python tools/diff_packed_weights.py \
        --packed  ../checkpoints/student_ssv2_salt_tube1/student-latest.pth.tar \
        --student /scratch/sd6701/salt/p-salt/results/student_ssv2_salt_tube1/checkpoints/01266910/last/checkpoint.pth \
        --teacher /scratch/sd6701/salt/p-salt/results/teacher_ssv2/checkpoints/8f00b836/last/checkpoint.pth
"""
import argparse

import torch


def strip(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}


def compare(label, a, b):
    """a = source tensors, b = packed tensors. Returns True when identical."""
    a, b = strip(a), strip(b)
    only_src = sorted(set(a) - set(b))
    only_pack = sorted(set(b) - set(a))
    shared = sorted(set(a) & set(b))

    same, differ, dtype_or_shape = [], [], []
    worst = 0.0
    for k in shared:
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            dtype_or_shape.append(k)
            continue
        if torch.equal(x, y):
            same.append(k)
        else:
            differ.append(k)
            worst = max(worst, (x.float() - y.float()).abs().max().item())

    ok = not only_src and not only_pack and not differ and not dtype_or_shape
    print(f"\n[{label}] {'IDENTICAL' if ok else 'MISMATCH'}")
    print(f"  {len(shared)} shared keys — {len(same)} bit-identical, {len(differ)} differing, "
          f"{len(dtype_or_shape)} shape/dtype mismatches")
    if only_src:
        print(f"  in source only ({len(only_src)}): {only_src[:4]}")
    if only_pack:
        print(f"  in packed only ({len(only_pack)}): {only_pack[:4]}")
    if differ:
        print(f"  differing (max abs diff {worst:.3e}): {differ[:4]}")
    if dtype_or_shape:
        k = dtype_or_shape[0]
        print(f"  shape/dtype: {k}  src {tuple(a[k].shape)}/{a[k].dtype} "
              f"vs packed {tuple(b[k].shape)}/{b[k].dtype}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--student", required=True, help="p-salt student ckpt (holds encoder + predictor)")
    ap.add_argument("--teacher", required=True, help="p-salt teacher ckpt (its 'encoder' is the target)")
    a = ap.parse_args()

    load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    packed, student, teacher = load(a.packed), load(a.student), load(a.teacher)

    print(f"packed   {a.packed}")
    print(f"student  {a.student}")
    print(f"teacher  {a.teacher}")

    results = [
        compare("encoder        (student[encoder])", student["encoder"], packed["encoder"]),
        compare("predictor      (student[predictor])", student["predictor"], packed["predictor"]),
        compare("target_encoder (teacher[encoder])", teacher["encoder"], packed["target_encoder"]),
    ]

    print()
    if all(results):
        print("Every packed tensor is bit-identical to its source. Packing is lossless.")
    else:
        print("At least one module differs from its source — the packed checkpoint is "
              "not the model you think it is.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
