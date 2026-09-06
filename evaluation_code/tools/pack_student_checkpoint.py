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


CANDIDATES = {
    "encoder": ["encoder", "student", "student_encoder", "context_encoder", "model", "backbone"],
    "target_encoder": ["target_encoder", "teacher", "teacher_encoder", "ema_encoder", "encoder", "model", "backbone"],
    "predictor": ["predictor", "pred", "student_predictor"],
}


def _is_sd(d):
    return isinstance(d, dict) and len(d) > 0 and all(torch.is_tensor(v) for v in d.values())


def _strip(sd):
    return {k.replace("module.", ""): v for k, v in sd.items()}


def load_sd(path, key, role):
    """Return the state_dict for `role` (encoder / target_encoder / predictor).

    Resolution order: explicit --key; a top-level sub-dict whose name is a known
    alias for the role; a flat state_dict whose keys carry a `<alias>.` prefix
    (split by prefix); a bare state_dict (used as-is)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if key:
        sd = ckpt[key]
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        assert _is_sd(sd), f"{path}[{key}] is not a state_dict"
        print(f"  {role:15s} <- {path}[{key}]")
        return _strip(sd)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and not _is_sd(ckpt):
        ckpt = ckpt["state_dict"]
    if _is_sd(ckpt):
        sd = _strip(ckpt)
        for alias in CANDIDATES[role]:
            pre = alias + "."
            sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            if sub:
                print(f"  {role:15s} <- {path} (flat, prefix '{pre}', {len(sub)} tensors)")
                return sub
        print(f"  {role:15s} <- {path} (bare state_dict, {len(sd)} tensors)")
        return sd
    if isinstance(ckpt, dict):
        for alias in CANDIDATES[role]:
            if alias in ckpt:
                sd = ckpt[alias]
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                if _is_sd(sd):
                    print(f"  {role:15s} <- {path}['{alias}']")
                    return _strip(sd)
        raise SystemExit(f"{path}: no state_dict found for {role}; top-level keys = {list(ckpt.keys())}. Pass --{role.replace('target_encoder','teacher').replace('encoder','student')}-key.")
    raise SystemExit(f"{path}: unsupported checkpoint type {type(ckpt)}")


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

    def wrap(sd):
        # eval.py wraps every network in MultiMaskWrapper, whose state_dict keys are
        # "backbone.<name>". Store that form so the loader matches 1:1 (it also
        # aligns the prefix itself, but the file should be right on its own).
        if all(k.startswith("backbone.") for k in sd):
            return sd
        return {f"backbone.{k}": v for k, v in sd.items()}

    out = {
        "encoder": wrap(load_sd(a.student, a.student_key, "encoder")),
        "target_encoder": wrap(load_sd(a.teacher, a.teacher_key, "target_encoder")),
        "predictor": wrap(load_sd(a.predictor, a.predictor_key, "predictor")),
        "epoch": a.epoch,
    }
    for k in ("encoder", "target_encoder", "predictor"):
        n = sum(v.numel() for v in out[k].values())
        print(f"{k:15s} {len(out[k]):5d} tensors  {n/1e6:8.1f}M params  e.g. {next(iter(out[k]))}")
        pe = [v for kk, v in out[k].items() if "patch_embed" in kk and v.ndim == 5]
        if pe:
            print(f"{'':15s} patch_embed {tuple(pe[0].shape)} -> embed_dim={pe[0].shape[0]} tubelet={pe[0].shape[2]} patch={pe[0].shape[3]}")
    pe_e = [v for kk, v in out["encoder"].items() if "patch_embed" in kk and v.ndim == 5]
    pe_t = [v for kk, v in out["target_encoder"].items() if "patch_embed" in kk and v.ndim == 5]
    if pe_e and pe_t and pe_e[0].shape != pe_t[0].shape:
        raise SystemExit(f"encoder / target_encoder patch_embed differ: {tuple(pe_e[0].shape)} vs {tuple(pe_t[0].shape)} -- token grids must match")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
