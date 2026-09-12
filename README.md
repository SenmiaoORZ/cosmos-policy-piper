# Piper Sponge Cosmos Policy

This repository contains the project code and reproduction workflow for
fine-tuning **Cosmos Policy** on a Piper sponge-manipulation task. It is a
focused adaptation of the upstream Cosmos Policy training stack: the project
adds the Piper data conversion, task-string handling, dataset statistics, and
an overfit-oriented training configuration while retaining the upstream model
and distributed training implementation.

> This project is an experimental research baseline. Do not connect a model to
> physical hardware until the action convention, units, joint limits, and
> emergency-stop procedure have been independently validated.

## What is in this repository

- Piper sponge dataset conversion and normalization utilities.
- A Cosmos Predict2 2B, 480p configuration for the Piper sponge overfit run.
- Training, checkpoint validation, and open-loop evaluation utilities.
- A released iteration-50,000 inference checkpoint on Hugging Face.

## Task representation

| Signal | Representation |
| --- | --- |
| Observations | Head RGB, left-wrist RGB, and 7-D Piper joint position |
| Action | Next synchronized 7-D joint-position target |
| Model compatibility format | 14-D `[piper_7, zeros_7]`; the left-wrist image is duplicated into the required `cam_right_wrist` input |
| Hardware execution | Only use predicted dimensions `0:7`; ignore dimensions `7:14` |

## Setup

Follow the CUDA/container environment instructions in [SETUP.md](SETUP.md).
The examples below assume the provided `uv` environment and CUDA 12.8 extra.

## Prepare the Piper sponge dataset

The canonical Piper sponge source dataset is not bundled in this repository.
After obtaining it, preprocess it and generate the statistics and text
embedding cache:

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

Run a single-GPU smoke test first:

```bash
export PIPER_DATASET_DIR=/path/to/Piper-Sponge-Cosmos-Policy/preprocessed
uv run --extra cu128 --group aloha --python 3.10 \
  torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment=cosmos_predict2_2b_480p_piper_sponge_overfit \
  trainer.max_iter=10 dataloader_train.num_workers=0
```

For the full overfit run, remove the final two overrides and set
`--nproc_per_node` to the available GPU count.

## Released checkpoint

The iteration-50,000 inference checkpoint is hosted in the private Hugging
Face repository [SourORZ/cosmos-policy-piper-50k](https://huggingface.co/SourORZ/cosmos-policy-piper-50k).
Request access from the repository owner, then download it with:

```bash
huggingface-cli download SourORZ/cosmos-policy-piper-50k \
  --repo-type model \
  --local-dir /path/to/checkpoints
```

The checkpoint uses the PyTorch Distributed Checkpoint (DCP) format. Preserve
both `checkpoint/model/.metadata` and `checkpoint/model/__0_0.distcp` exactly
as downloaded. Validate the checkpoint before using it:

```bash
python -m cosmos_policy.experiments.robot.piper.validate_checkpoint \
  --checkpoint /path/to/checkpoints/checkpoint/model
```

The release is intended for inference. Its optimizer shard is incomplete, so
it must not be used as a training-resume checkpoint.

## Deployment notes

The model output is padded to a 14-D compatibility representation. A Piper
controller must normalize the 7-D observed joint position with the released
dataset statistics, unnormalize only the first 7 action dimensions, enforce
robot-specific limits, and reject dimensions `7:14`. The checkpoint validation
tool verifies DCP loading only; it does not command a robot.

## Upstream project

This work is built on [Cosmos Policy](https://github.com/nvidia-cosmos/cosmos-policy).
Please follow the upstream license and cite the original Cosmos Policy paper
when using the underlying model and training stack.
