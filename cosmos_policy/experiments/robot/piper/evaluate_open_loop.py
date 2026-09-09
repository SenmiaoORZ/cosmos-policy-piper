#!/usr/bin/env python3
"""Offline open-loop action-chunk evaluation for a Piper Cosmos Policy checkpoint.

This script only reads a recorded Piper episode.  It never opens a robot
connection or sends commands.  For one logged observation it samples the
policy's 16-step action chunk and reports its error against the corresponding
recorded target chunk.  The first seven dimensions are the Piper joint targets;
dimensions 7:14 are the compatibility padding used by the ALOHA-based model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)


@dataclass
class OfflinePiperConfig:
    """Subset of deployment options consumed by ``get_action``."""

    suite: str = "aloha"  # Reuse the released three-camera ALOHA latent layout.
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 2
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    chunk_size: int = 16
    use_variance_scale: bool = False
    use_jpeg_compression: bool = False
    trained_with_image_aug: bool = False
    flip_images: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local DCP model directory containing .metadata")
    parser.add_argument("--data-dir", type=Path, required=True, help="Preprocessed Piper dataset root")
    parser.add_argument("--episode", default="episode0", help="Episode stem under data-dir/train (default: episode0)")
    parser.add_argument("--step", type=int, default=0, help="Recorded observation index")
    parser.add_argument("--seed", type=int, default=195, help="Deterministic diffusion seed")
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True, help="Where to write the JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_path = args.data_dir / "train" / f"{args.episode}.hdf5"
    if not episode_path.is_file():
        raise FileNotFoundError(episode_path)
    if not (args.checkpoint / ".metadata").is_file():
        raise FileNotFoundError(f"Expected DCP metadata at {args.checkpoint / '.metadata'}")

    cfg = OfflinePiperConfig()
    statistics = load_dataset_stats(str(args.data_dir / "dataset_statistics.json"))
    init_t5_text_embeddings_cache(str(args.data_dir / "t5_embeddings.pkl"))

    model, _ = load_model_from_checkpoint(
        experiment_name="cosmos_predict2_2b_480p_piper_sponge_overfit",
        s3_checkpoint_dir=str(args.checkpoint),
        config_file="cosmos_policy/config/config.py",
        enable_fsdp=False,
        load_ema_to_reg=False,
        instantiate_ema=False,
        to_device="cuda",
    )
    model.eval()

    with h5py.File(episode_path, "r") as episode:
        total_steps = len(episode["action"])
        if not 0 <= args.step < total_steps:
            raise ValueError(f"step {args.step} is outside [0, {total_steps})")
        observation = {
            "left_wrist_image": episode["observations/images/cam_left_wrist"][args.step],
            "right_wrist_image": episode["observations/images/cam_right_wrist"][args.step],
            "primary_image": episode["observations/images/cam_high"][args.step],
            "proprio": episode["observations/qpos"][args.step].astype(np.float32),
        }
        instruction = str(episode.attrs["task_description"])
        targets = episode["action"][args.step : args.step + cfg.chunk_size].astype(np.float32)

    # Match the dataset's end-of-episode target padding exactly.
    if len(targets) < cfg.chunk_size:
        targets = np.concatenate([targets, np.repeat(targets[-1:], cfg.chunk_size - len(targets), axis=0)])

    result = get_action(
        cfg,
        model,
        statistics,
        observation,
        instruction,
        seed=args.seed,
        num_denoising_steps_action=args.denoising_steps,
        generate_future_state_and_value_in_parallel=False,
    )
    predicted = np.asarray(result["actions"], dtype=np.float32)
    if predicted.shape != targets.shape:
        raise RuntimeError(f"Predicted {predicted.shape}, expected {targets.shape}")
    if not np.isfinite(predicted).all():
        raise RuntimeError("Non-finite action values were generated")

    error = predicted[:, :7] - targets[:, :7]
    report = {
        "episode": args.episode,
        "step": args.step,
        "instruction": instruction,
        "chunk_size": cfg.chunk_size,
        "action_dim": 7,
        "denoising_steps": args.denoising_steps,
        "seed": args.seed,
        "finite_actions": bool(np.isfinite(predicted).all()),
        "action_mae": float(np.mean(np.abs(error))),
        "action_rmse": float(np.sqrt(np.mean(error**2))),
        "per_timestep_mae": np.mean(np.abs(error), axis=1).tolist(),
        "predicted_actions": predicted[:, :7].tolist(),
        "target_actions": targets[:, :7].tolist(),
        "padding_max_abs": float(np.max(np.abs(predicted[:, 7:]))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("chunk_size", "finite_actions", "action_mae", "action_rmse", "padding_max_abs")}, indent=2))


if __name__ == "__main__":
    main()
