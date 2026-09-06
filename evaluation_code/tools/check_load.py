"""Report whether a packed checkpoint actually loads into the model an eval
config builds -- without running the eval.

eval.py calls load_state_dict(..., strict=False), so a name or shape mismatch
only logs a line and leaves those tensors at their random initialisation. A run
can therefore complete, write a full CSV, and score at chance because part of
the model never received its weights. This compares the two key sets directly
and, for keys present in both, checks the tensors are actually equal after the
load -- so "loaded" means loaded, not merely "no error was raised".

Usage:
    cd evaluation_code
    python tools/check_load.py evals/intuitive_physics/configs/student_intphys.yaml
"""
import argparse
import os
import sys

import torch
import yaml


def summarize(name, model, packed):
    """Compare a built module against the state_dict the eval would feed it."""
    # eval.py strips 'module.' before loading; mirror that exactly (eval.py:604)
    packed = {k.replace("module.", ""): v for k, v in packed.items()}
    have = model.state_dict()
    missing = [k for k in have if k not in packed]
    unexpected = [k for k in packed if k not in have]
    shape_bad = [k for k in have if k in packed and packed[k].shape != have[k].shape]

    loadable = [k for k in have if k in packed and packed[k].shape == have[k].shape]
    model.load_state_dict({k: packed[k] for k in loadable}, strict=False)
    after = model.state_dict()
    # a tensor that is still not equal after loading did not take effect
    not_applied = [k for k in loadable if not torch.equal(after[k].cpu(), packed[k].cpu())]

    ok = not missing and not shape_bad and not not_applied
    print(f"\n[{name}] {'OK' if ok else 'PROBLEM'}")
    print(f"  model expects {len(have)} tensors, checkpoint offers {len(packed)}")
    print(f"  loaded {len(loadable)}  missing {len(missing)}  "
          f"shape-mismatch {len(shape_bad)}  unexpected {len(unexpected)}")
    for label, keys in (("missing", missing), ("shape-mismatch", shape_bad),
                        ("unexpected", unexpected), ("not-applied", not_applied)):
        if keys:
            print(f"  {label}: {keys[:6]}{' ...' if len(keys) > 6 else ''}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="eval yaml (e.g. evals/intuitive_physics/configs/student_intphys.yaml)")
    a = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from evals.intuitive_physics.eval import init_model

    cfg = yaml.safe_load(open(a.config))
    p = cfg["pretrain"]
    ckpt_path = os.path.join(p["folder"], p["checkpoint"])
    print(f"config     {a.config}")
    print(f"checkpoint {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    # Build + load exactly as the eval does, on CPU.
    enc, tgt, pred = init_model(
        device=torch.device("cpu"),
        pretrained=ckpt_path,
        model_name=p["model_name"],
        patch_size=p.get("patch_size", 16),
        crop_size=cfg["data"].get("resolution", 224),
        frames_per_clip=p.get("frames_per_clip", 16),
        tubelet_size=p.get("tubelet_size", 2),
        use_sdpa=p.get("use_sdpa", False),
        use_SiLU=p.get("use_silu", False),
        wide_SiLU=p.get("wide_silu", False),
        is_causal=p.get("is_causal", False),
        pred_is_causal=p.get("pred_is_causal", False),
        uniform_power=p.get("uniform_power", False),
        enc_checkpoint_key=p.get("enc_checkpoint_key", "encoder"),
        pred_checkpoint_key=p.get("pred_checkpoint_key", "predictor"),
        pred_embed_dim=p.get("pred_embed_dim", 384),
        pred_depth=p.get("pred_depth", 12),
        pred_num_heads=p.get("pred_num_heads", None),
        custom_model_module=p.get("custom_model_module"),
        custom_model_kwargs=p.get("custom_model_kwargs"),
    )

    ck = torch.load(ckpt_path, map_location="cpu")
    print(f"\ncheckpoint top-level keys: {[k for k in ck if k != 'epoch']} (epoch {ck.get('epoch')})")
    results = [
        summarize("encoder", enc, ck[p.get("enc_checkpoint_key", "encoder")]),
        summarize("target_encoder", tgt, ck["target_encoder"]),
        summarize("predictor", pred, ck[p.get("pred_checkpoint_key", "predictor")]),
    ]

    print()
    if all(results):
        print("All three modules loaded fully. A chance-level score is NOT a loading problem.")
    else:
        print("At least one module did not load fully -- those tensors are still random, "
              "and any score from this config is meaningless.")
        sys.exit(1)


if __name__ == "__main__":
    main()
