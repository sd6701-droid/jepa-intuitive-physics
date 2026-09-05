"""Print the structure of a training checkpoint so it can be mapped onto the
eval's (encoder, target_encoder, predictor) layout.

    python tools/inspect_checkpoint.py /path/checkpoint.pth [--all]

Reports: top-level keys, and for every state_dict found: #tensors, #params,
first/last key names, and inferred architecture (patch-embed shape ->
tubelet/patch/embed_dim, number of blocks, qkv width -> heads if possible).
"""
import sys
import re
import torch


def is_sd(d):
    return isinstance(d, dict) and len(d) > 0 and all(torch.is_tensor(v) for v in d.values())


def describe(name, sd, show_all=False):
    n = sum(v.numel() for v in sd.values())
    keys = list(sd.keys())
    print(f"\n[{name}]  {len(keys)} tensors, {n/1e6:.2f}M params")
    blocks = sorted({int(m.group(1)) for k in keys for m in [re.search(r"blocks\.(\d+)\.", k)] if m})
    if blocks:
        print(f"  blocks: {len(blocks)} (0..{max(blocks)})")
    for k in keys:
        v = sd[k]
        if "patch_embed" in k and v.ndim == 5:
            D, C, t, ph, pw = v.shape
            print(f"  patch_embed {k}: embed_dim={D} in_ch={C} tubelet={t} patch={ph}x{pw}")
        if k.endswith("pos_embed"):
            print(f"  pos_embed {k}: {tuple(v.shape)}")
        if "mask_token" in k:
            print(f"  {k}: {tuple(v.shape)}")
        if re.search(r"blocks\.0\.attn\.qkv\.weight", k):
            print(f"  attn qkv (block0): {tuple(v.shape)}")
        if "predictor_embed" in k or "predictor_proj" in k:
            print(f"  {k}: {tuple(v.shape)}")
    print(f"  first keys: {keys[:3]}")
    print(f"  last keys : {keys[-3:]}")
    if show_all:
        for k in keys:
            print(f"    {k:70s} {tuple(sd[k].shape)}")


def walk(obj, prefix, show_all):
    if is_sd(obj):
        describe(prefix, obj, show_all)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}.{k}" if prefix else str(k), show_all)
            else:
                summary = tuple(v.shape) if torch.is_tensor(v) else (v if isinstance(v, (int, float, str, bool)) else type(v).__name__)
                print(f"  {prefix + '.' if prefix else ''}{k}: {summary}")


if __name__ == "__main__":
    path = sys.argv[1]
    show_all = "--all" in sys.argv
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"{path}\ntop-level type: {type(ckpt).__name__}; keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else '-'}")
    walk(ckpt, "", show_all)
