"""Pack a student encoder, frozen teacher and predictor into the single
checkpoint layout the eval code expects:

    {"encoder": <student state_dict>,
     "target_encoder": <teacher state_dict>,
     "predictor": <predictor state_dict>,
     "epoch": int}

Each input may be a bare state_dict or a checkpoint dict; use --*-key to pick
the sub-dict when needed. "module." prefixes are stripped by the loader.

Example:
    python tools/pack_student_checkpoint.py \
        --student  /path/student.pth  --student-key model \
        --teacher  /path/teacher.pth \
        --predictor /path/predictor.pth \
        --out /path/checkpoints/student/student-latest.pth.tar
"""
import argparse
import os
import torch


def load_sd(path, key):
    ckpt = torch.load(path, map_location="cpu")
    if key:
        ckpt = ckpt[key]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    assert all(torch.is_tensor(v) for v in ckpt.values()), f"{path} does not look like a state_dict; pass --*-key"
    return {k.replace("module.", ""): v for k, v in ckpt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--student-key", default=None)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--teacher-key", default=None)
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--predictor-key", default=None)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {
        "encoder": load_sd(a.student, a.student_key),
        "target_encoder": load_sd(a.teacher, a.teacher_key),
        "predictor": load_sd(a.predictor, a.predictor_key),
        "epoch": a.epoch,
    }
    for k in ("encoder", "target_encoder", "predictor"):
        n = sum(v.numel() for v in out[k].values())
        print(f"{k:15s} {len(out[k]):5d} tensors  {n/1e6:8.1f}M params  e.g. {next(iter(out[k]))}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
