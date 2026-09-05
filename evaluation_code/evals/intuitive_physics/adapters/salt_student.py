"""Adapter that builds (encoder, target_encoder, predictor) for the SALT
student/teacher setup so the intuitive-physics eval can score it.

Selected from the yaml with:
    pretrain:
      custom_model_module: evals.intuitive_physics.adapters.salt_student
      custom_model_kwargs: {...}   # forwarded to build_models()

Contract the eval relies on (see extract_losses in eval.py):
  encoder(x, [idx])          -> [tokens[B, K, D]]   x: [B,3,T,224,224], idx: [B,K] token indices to KEEP
  target_encoder(x, [idx])   -> [tokens[B, N, D]]   same, called with all N indices
  predictor(ctx, tgt, [idx_ctx], [idx_tgt]) -> [pred[B, K_tgt, D]]   one embedding per target index
Tokens are ordered time-major: index = t * (H/16 * W/16) + h * (W/16) + w, with
t in [0, T/tubelet). Encoder and target encoder must share this grid.

By default this builds the repo's own ViT + predictor (bi-directional attention,
is_causal=False). To use the p-salt predictor class instead, fill in
`build_salt_predictor` below and set custom_model_kwargs.use_salt_predictor: true.
"""
import sys
import copy
from functools import partial

import torch
import torch.nn as nn

import src.models.vision_transformer as vit
import src.models.predictor as vit_pred
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper


def build_salt_predictor(embed_dim, num_heads, **kw):
    """TODO: return the p-salt predictor so its checkpoint weights load 1:1.
    Example:
        sys.path.insert(0, "/scratch/sd6701/salt/p-salt")
        from models.predictor import Predictor   # <- adjust to the real import
        return Predictor(embed_dim=embed_dim, ...)
    The returned module is wrapped by PredictorMultiMaskWrapper, so its forward
    must be forward(ctxt, tgt, masks_ctxt, masks_tgt, mask_index=0) -> [B, K_tgt, D].
    If the p-salt forward has a different signature, wrap it in a small nn.Module here.
    """
    raise NotImplementedError("fill in build_salt_predictor() with the p-salt predictor class")


def build_models(
    device,
    model_name="vit_small",
    teacher_model_name=None,
    patch_size=16,
    crop_size=224,
    frames_per_clip=16,
    tubelet_size=2,
    teacher_tubelet_size=None,
    use_sdpa=True,
    use_SiLU=False,
    wide_SiLU=True,
    uniform_power=True,
    pred_depth=12,
    pred_embed_dim=384,
    pred_num_heads=None,
    num_mask_tokens=2,
    use_mask_tokens=True,
    use_salt_predictor=False,
    salt_predictor_kwargs=None,
):
    teacher_model_name = teacher_model_name or model_name
    teacher_tubelet_size = teacher_tubelet_size or tubelet_size
    assert teacher_tubelet_size == tubelet_size, (
        "encoder and target_encoder must produce the same token grid; "
        f"got tubelet {tubelet_size} vs {teacher_tubelet_size}")

    def make_vit(name):
        return vit.__dict__[name](
            img_size=crop_size, patch_size=patch_size, num_frames=frames_per_clip,
            tubelet_size=tubelet_size, uniform_power=uniform_power, use_sdpa=use_sdpa,
            use_SiLU=use_SiLU, wide_SiLU=wide_SiLU, is_causal=False)

    encoder = make_vit(model_name)              # frozen ViT-S student
    target_encoder = make_vit(teacher_model_name)  # frozen teacher (targets)
    embed_dim, num_heads = encoder.embed_dim, encoder.num_heads

    if use_salt_predictor:
        predictor = build_salt_predictor(embed_dim=embed_dim, num_heads=num_heads,
                                         **(salt_predictor_kwargs or {}))
    else:
        predictor = vit_pred.vit_predictor(
            img_size=crop_size, use_mask_tokens=use_mask_tokens, is_causal=False,
            patch_size=patch_size, num_frames=frames_per_clip, tubelet_size=tubelet_size,
            embed_dim=embed_dim, predictor_embed_dim=pred_embed_dim, depth=pred_depth,
            num_heads=pred_num_heads or num_heads, uniform_power=uniform_power,
            num_mask_tokens=num_mask_tokens, zero_init_mask_tokens=True, use_sdpa=use_sdpa,
            use_SiLU=use_SiLU, use_rope="rope" in model_name, rope_is_1D="rope1D" in model_name,
            wide_SiLU=wide_SiLU)

    return MultiMaskWrapper(encoder), MultiMaskWrapper(target_encoder), PredictorMultiMaskWrapper(predictor)
