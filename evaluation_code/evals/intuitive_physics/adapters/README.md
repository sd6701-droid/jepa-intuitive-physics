# Model adapters

`init_model` in `eval.py` normally builds the repo's ViT + predictor from the
`pretrain` section. Set `pretrain.custom_model_module` to a python module that
exposes

    build_models(device, **custom_model_kwargs) -> (encoder, target_encoder, predictor)

and the eval will use those instead, then load weights from the checkpoint keys
`encoder`, `target_encoder`, `predictor` (non-strict; read the log for missing keys).

Interface each module must satisfy is documented in `salt_student.py`.
