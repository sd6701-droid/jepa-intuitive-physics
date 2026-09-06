"""Create a randomly initialised (untrained) encoder / target_encoder / predictor
checkpoint whose architecture is read from an eval config's `pretrain` section,
so the chance-level control is guaranteed to match the model it controls for
(the paper's "vit-*-random-N" baselines).

    python tools/make_random_checkpoint.py \
        --config evals/intuitive_physics/configs/random_intphys.yaml \
        --out /scratch/sd6701/jepa-intuitive-physics/checkpoints/random_vit_s/student-latest.pth.tar \
        --seed 0

Construction mirrors init_model() in evals/intuitive_physics/eval.py exactly:
same factories, same flags, RoPE derived from model_name. target_encoder is a
deepcopy of the random encoder (JEPA init: EMA teacher starts as a copy);
predictor is drawn independently. Attention is bi-directional unless the
config sets is_causal / pred_is_causal.
"""
import argparse
import copy
import os

import torch
import yaml

import src.models.vision_transformer as vit
import src.models.predictor as vit_pred


def build_from_config(pre, seed):
    torch.manual_seed(seed)
    model_name = pre["model_name"]
    common = dict(
        img_size=pre.get("crop_size", 224),
        patch_size=pre.get("patch_size", 16),
        num_frames=pre.get("frames_per_clip", 16),
        tubelet_size=pre.get("tubelet_size", 2),
        uniform_power=pre.get("uniform_power", False),
        use_sdpa=pre.get("use_sdpa", True),
        use_SiLU=pre.get("use_silu", False),
        wide_SiLU=pre.get("wide_silu", True),
        is_causal=pre.get("is_causal", False),
    )
    encoder = vit.__dict__[model_name](**common)
    target_encoder = copy.deepcopy(encoder)

    torch.manual_seed(seed + 20_000)
    predictor = vit_pred.vit_predictor(
        img_size=common["img_size"],
        use_mask_tokens=True,
        is_causal=pre.get("pred_is_causal", False),
        patch_size=common["patch_size"],
        num_frames=common["num_frames"],
        tubelet_size=common["tubelet_size"],
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pre.get("pred_embed_dim", 384),
        depth=pre.get("pred_depth", 12),
        num_heads=pre.get("pred_num_heads") or encoder.num_heads,
        uniform_power=common["uniform_power"],
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=common["use_sdpa"],
        use_SiLU=common["use_SiLU"],
        use_rope="rope" in model_name,
        rope_is_1D="rope1D" in model_name,
        wide_SiLU=common["wide_SiLU"],
    )
    return encoder, target_encoder, predictor, common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="eval yaml whose pretrain: section defines the architecture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    pre = cfg["pretrain"]
    encoder, target_encoder, predictor, common = build_from_config(pre, a.seed)

    # Keys carry the MultiMaskWrapper "backbone." prefix because load_pretrained
    # walks wrapper.state_dict() (see eval.py) when matching names.
    out = {
        "encoder": {f"backbone.{k}": v for k, v in encoder.state_dict().items()},
        "target_encoder": {f"backbone.{k}": v for k, v in target_encoder.state_dict().items()},
        "predictor": {f"backbone.{k}": v for k, v in predictor.state_dict().items()},
        "epoch": 0,
        "random_init": {"config": os.path.abspath(a.config), "seed": a.seed, "pretrain": pre},
    }
    print(f"architecture from {a.config}: {pre['model_name']} tubelet={common['tubelet_size']} "
          f"causal={common['is_causal']}/{pre.get('pred_is_causal', False)} "
          f"pred={pre.get('pred_depth', 12)}x{pre.get('pred_embed_dim', 384)} heads={pre.get('pred_num_heads') or encoder.num_heads} "
          f"uniform_power={common['uniform_power']} silu={common['use_SiLU']}/{common['wide_SiLU']}")
    for k in ("encoder", "target_encoder", "predictor"):
        n = sum(v.numel() for v in out[k].values())
        print(f"  {k:15s} {len(out[k]):5d} tensors  {n/1e6:6.1f}M params")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    print("wrote", a.out, "| seed", a.seed)


if __name__ == "__main__":
    main()
