---
library_name: pytorch
tags:
  - cosmos
  - robotics
  - piper
  - diffusion-policy
---

# Cosmos Policy Piper Sponge Overfit

This is a task-specific fine-tune of
[`nvidia/Cosmos-Predict2-2B-Video2World`](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)
for the Piper sponge task. It is deliberately an overfit baseline, not a
general-purpose robot policy.

## Training data and setup

- 20 canonical Piper sponge episodes, 5,897 synchronized steps.
- Inputs: head RGB, left-wrist RGB, and 7-D Piper joint position.
- Target: next synchronized 7-D joint-position target.
- The upstream Cosmos Policy ALOHA interface requires 14-D proprio/action
  tensors, so this release stores `[piper_7, zeros_7]`. Only dimensions `0:7`
  are meaningful and may be executed.
- Training: 5,000 steps, batch size 4, one GPU, initialized from the public
  Cosmos Predict2 2B Video2World checkpoint.
- The release includes the pre- and post-normalization statistics used during
  training.

## Files

- `model/`: native PyTorch Distributed Checkpoint (DCP), including
  `.metadata` and `*.distcp` weight shards.
- `config.yaml`: resolved training configuration.
- `dataset_statistics.json` and `dataset_statistics_post_norm.json`:
  normalization statistics required by the Piper controller.

## Load and validate

Use the companion repository
[`SenmiaoORZ/cosmos-policy-piper`](https://github.com/SenmiaoORZ/cosmos-policy-piper).
Its `download_hf_checkpoint` helper recognizes this layout and returns the
`model/` DCP directory. To validate a downloaded snapshot:

```bash
python -m cosmos_policy.experiments.robot.piper.validate_checkpoint \
  --checkpoint /path/to/snapshot/model
```

## Safety and limits

This release has no held-out generalization claim and has not been validated on
physical hardware by this release process. Before commanding a robot, the
integrator must verify joint order, position-target convention, units, camera
calibration, timing, action clipping, joint limits, and the emergency-stop
path. Normalize the observed 7-D state with the included statistics;
unnormalize only predicted action dimensions `0:7`; reject dimensions `7:14`.

## License and attribution

The fine-tuned weights are derived from NVIDIA Cosmos Predict2 and are subject
to the applicable upstream NVIDIA model license and access terms. The companion
code retains its own upstream notices and Apache-2.0 licensing where provided.
