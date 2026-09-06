"""Identify which source checkpoints a packed student file was actually built
from, by fingerprinting tensors rather than trusting the filename.

pack_student_checkpoint.py records no provenance, so a pack run that forgot to
override STUDENT= or TEACHER= produces a plausibly-named file holding the wrong
weights. This hashes each state_dict and reports, for every packed file, which
candidate source each of its three modules matches.

Usage:
    cd evaluation_code
    python tools/verify_pack.py \
        --packed ../checkpoints/student_ssv2_salt_tube1/student-latest.pth.tar \
        --packed ../checkpoints/student_ssv2_causal_proposed_flex_tube1/student-latest.pth.tar \
        --source /scratch/sd6701/salt/p-salt/results/student_ssv2_salt_tube1/checkpoints/01266910/last/checkpoint.pth \
        --source /scratch/sd6701/salt/p-salt/results/student_ssv2_causal_proposed_flex_tube1/checkpoints/e9434cb3/last/checkpoint.pth \
        --source /scratch/sd6701/salt/p-salt/results/teacher_ssv2/checkpoints/8f00b836/last/checkpoint.pth
"""
import argparse
import hashlib
import os

import torch


def is_sd(d):
    return isinstance(d, dict) and len(d) > 0 and all(torch.is_tensor(v) for v in d.values())


def fingerprint(sd):
    """Stable hash of a state_dict, invariant to the 'module.' prefix.

    pack_student_checkpoint.py strips 'module.' when packing, so the same
    weights must fingerprint identically before and after that transform.
    """
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.replace("module.", "").encode())
        t = sd[k].detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.float().numpy().tobytes())
    return h.hexdigest()[:16]


def state_dicts(path):
    """Yield (label, fingerprint) for every state_dict inside a checkpoint."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    if is_sd(ck):
        out["<root>"] = fingerprint(ck)
        return out
    if isinstance(ck, dict):
        for k, v in ck.items():
            if is_sd(v):
                out[k] = fingerprint(v)
            elif isinstance(v, dict) and "state_dict" in v and is_sd(v["state_dict"]):
                out[k] = fingerprint(v["state_dict"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", action="append", required=True, help="packed .pth.tar (repeatable)")
    ap.add_argument("--source", action="append", required=True, help="candidate source checkpoint (repeatable)")
    a = ap.parse_args()

    # fingerprint -> list of "file[key]" it appears in
    index = {}
    print("scanning sources...")
    for s in a.source:
        for key, fp in state_dicts(s).items():
            index.setdefault(fp, []).append(f"{os.path.basename(os.path.dirname(os.path.dirname(s)))}/{os.path.basename(s)}[{key}]")
            print(f"  {fp}  {s}[{key}]")

    for pk in a.packed:
        print(f"\n=== {pk} ===")
        for key, fp in state_dicts(pk).items():
            match = index.get(fp)
            if match:
                print(f"  {key:15s} {fp}  <- {', '.join(match)}")
            else:
                print(f"  {key:15s} {fp}  <- NO MATCH among the given sources")

    print("\nTwo packed files sharing a fingerprint for the same module were built "
          "from the same source -- check your STUDENT=/TEACHER= overrides if that "
          "is not what you intended.")


if __name__ == "__main__":
    main()
