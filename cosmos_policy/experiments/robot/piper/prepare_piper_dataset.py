#!/usr/bin/env python3
"""Convert canonical Piper episodes to the released Cosmos Policy ALOHA layout.

This deliberately uses ALOHA-compatible 14-D tensors so no model internals
need to be rewritten.  Dimensions 0:7 hold the Piper joint-position state and
next-step target; dimensions 7:14 are zero padding.  The two available camera
streams populate `cam_high` and `cam_left_wrist`; `cam_right_wrist` duplicates
the left wrist feed solely to satisfy the released ALOHA model's 3-camera
layout.  The deployment adapter must discard output dimensions 7:14.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


DEFAULT_TASK = "Pick up the sponge and place into the basket"
CAMERAS = ("cam_high", "cam_left_wrist")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Canonical Piper dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Output preprocessed dataset root")
    parser.add_argument("--image-size", type=int, default=256, help="Square RGB image side length")
    parser.add_argument("--val-episodes", type=int, default=0, help="Reserve final N episodes for val")
    parser.add_argument("--overwrite", action="store_true", help="Replace output episode files")
    return parser.parse_args()


def episode_number(path: Path) -> int:
    return int(path.name.removeprefix("episode"))


def read_instruction(episode: Path) -> str:
    with (episode / "instructions.json").open(encoding="utf-8") as handle:
        value = json.load(handle)
    return str(value.get("instructions", [DEFAULT_TASK])[0])


def read_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS))


def source_paths(data: h5py.File, episode: Path, camera: str) -> list[Path]:
    values = data[f"camera/color/{camera}"][()]
    paths = [episode / value.decode("utf-8") for value in values]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing synchronized image: {missing[0]}")
    return paths


def make_padded_aloha_vectors(qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if qpos.ndim != 2 or qpos.shape[1] != 7:
        raise ValueError(f"Expected synchronized Piper qpos shaped (T, 7); got {qpos.shape}")
    future_qpos = np.concatenate([qpos[1:], qpos[-1:]], axis=0)
    padded_qpos = np.pad(qpos.astype(np.float32), ((0, 0), (0, 7)))
    padded_actions = np.pad(future_qpos.astype(np.float32), ((0, 0), (0, 7)))
    return padded_qpos, padded_actions


def write_episode(episode: Path, output: Path, image_size: int) -> None:
    with h5py.File(episode / "data.hdf5", "r") as source:
        camera_paths = {camera: source_paths(source, episode, camera) for camera in CAMERAS}
        qpos, action = make_padded_aloha_vectors(source["arm/jointStatePosition/puppetLeft"][:])
        qvel = np.pad(source["arm/jointStateVelocity/puppetLeft"][:].astype(np.float32), ((0, 0), (0, 7)))
        effort = np.pad(source["arm/jointStateEffort/puppetLeft"][:].astype(np.float32), ((0, 0), (0, 7)))
        timestamps = source["timestamp"][:].astype(np.float64)

    frame_count = len(qpos)
    if any(len(paths) != frame_count for paths in camera_paths.values()):
        raise ValueError(f"{episode.name}: camera and proprio frame counts are not synchronized")

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as target:
        target.attrs["sim"] = False
        target.attrs["task_description"] = read_instruction(episode)
        target.attrs["source_episode"] = episode.name
        target.attrs["piper_action_dim"] = 7
        target.attrs["piper_action_semantics"] = "next synchronized 7-D joint-position target"
        target.create_dataset("action", data=action, compression="gzip", compression_opts=4)
        observations = target.create_group("observations")
        observations.create_dataset("qpos", data=qpos, compression="gzip", compression_opts=4)
        observations.create_dataset("qvel", data=qvel, compression="gzip", compression_opts=4)
        observations.create_dataset("effort", data=effort, compression="gzip", compression_opts=4)
        observations.create_dataset("timestamp", data=timestamps)
        images = observations.create_group("images")
        image_shape = (frame_count, image_size, image_size, 3)
        datasets = {
            name: images.create_dataset(name, shape=image_shape, dtype=np.uint8, compression="gzip", compression_opts=4)
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        }
        for index, (high_path, wrist_path) in enumerate(zip(camera_paths["cam_high"], camera_paths["cam_left_wrist"], strict=True)):
            high = read_rgb(high_path, image_size)
            wrist = read_rgb(wrist_path, image_size)
            datasets["cam_high"][index] = high
            datasets["cam_left_wrist"][index] = wrist
            datasets["cam_right_wrist"][index] = wrist


def main() -> None:
    args = arguments()
    episodes = sorted((path for path in args.source.glob("episode*") if path.is_dir()), key=episode_number)
    if not episodes:
        raise FileNotFoundError(f"No episode directories found in {args.source}")
    if args.val_episodes < 0 or args.val_episodes >= len(episodes):
        raise ValueError("--val-episodes must be between 0 and number of episodes - 1")
    val_start = len(episodes) - args.val_episodes
    for index, episode in enumerate(episodes):
        split = "val" if args.val_episodes and index >= val_start else "train"
        output = args.output / split / f"{episode.name}.hdf5"
        if output.exists() and not args.overwrite:
            print(f"Keeping existing {output}")
            continue
        print(f"Converting {episode.name} -> {output}")
        write_episode(episode, output, args.image_size)


if __name__ == "__main__":
    main()
