# Eval configs

```
configs/
├── default_{intphys,grasp,inflevel}.yaml   # the paper's V-JEPA ViT-H configs (untouched)
├── intphys/                                # one folder per dataset ...
├── grasp/
└── inflevel/
    ├── random.yaml                         # ... four models each:
    ├── student_salt.yaml                   #   chance-level control (random ViT-S, same arch)
    ├── student_causal_proposed.yaml        #   SALT student, bi-directional
    └── student_causal_framelocal.yaml      #   SALT causal students (proposed / framelocal)
```

All twelve share one architecture block (ViT-S RoPE, tubelet 1, predictor 12x384x16h)
and one teacher (8f00b836). Only these differ:

| axis    | field(s)                                           |
|---------|----------------------------------------------------|
| model   | `pretrain.folder`, `write_tag`, `is_causal`, `pred_is_causal`, `tag`, `wandb.tags` |
| dataset | `data.frames_per_clip`, `data.context_lengths`, `data.frame_steps`, `dataset` |

Dataset settings follow the paper's `default_*.yaml`: IntPhys/GRASP 16 frames at
frame step 2/10 with contexts 2-10; InfLevel 32 frames at step 5 with contexts 4-20.

Results land in `<pretrain.folder>/intuitive_physics/<dataset>-<tag>/<write_tag>_r0.csv`;
summarise across models with `python tools/summarize_performance.py --root <folder> ...`.

Run one:  `CONFIG=evals/intuitive_physics/configs/grasp/student_salt.yaml bash scripts/slurm/submit_eval_dual.sh`
