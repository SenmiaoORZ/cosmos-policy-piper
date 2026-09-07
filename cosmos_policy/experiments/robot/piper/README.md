# Piper sponge Cosmos Policy overfit

This folder adapts the canonical Piper sponge dataset to the official Cosmos
Policy ALOHA training path. It intentionally favors a low-risk, compatible
overfit baseline over a new model architecture.

## Representation

- Observations: head RGB, left-wrist RGB, and 7-D Piper joint position.
- Action: next synchronized 7-D joint-position target.
- Compatibility padding: tensors are stored as 14-D `[piper_7, zeros_7]` and
  the left-wrist image is copied to the required `cam_right_wrist` slot.
- Deployment: execute **only** predicted dimensions `0:7`; reject or ignore
  dimensions `7:14`. Do not command hardware until the position-target
  convention, units, limits, and emergency-stop path have been verified.

## Prepare the dataset

```bash
python -m cosmos_policy.experiments.robot.piper.prepare_piper_dataset \
  --source /path/to/piper_sponge_canonical20_20260831 \
  --output /path/to/Piper-Sponge-Cosmos-Policy/preprocessed

python -m cosmos_policy.experiments.robot.piper.compute_piper_statistics \
  --data-dir /path/to/Piper-Sponge-Cosmos-Policy/preprocessed

export PIPER_DATASET_DIR=/path/to/Piper-Sponge-Cosmos-Policy/preprocessed
uv run --extra cu128 --group aloha --python 3.10 \
  -m cosmos_policy.datasets.save_aloha_t5_text_embeddings \
  --data_dir "$PIPER_DATASET_DIR"
```

## Train

Run from an NVIDIA CUDA environment after completing the upstream `SETUP.md`.
Use an initial smoke test before a long run:

```bash
export PIPER_DATASET_DIR=/path/to/Piper-Sponge-Cosmos-Policy/preprocessed
uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_piper_sponge_overfit \
  trainer.max_iter=10 dataloader_train.num_workers=0
```

For the overfit run, remove the final two overrides and select the actual GPU
count. The upstream implementation provides the training loop; this fork only
adds Piper conversion, task-string loading, and the overfit config.
