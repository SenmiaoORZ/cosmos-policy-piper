#!/usr/bin/env python3
"""Precompute Cosmos Policy min/max normalization statistics for Piper data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy.datasets.dataset_utils import calculate_dataset_statistics, rescale_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Preprocessed Piper dataset root")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing statistics JSON files")
    return parser.parse_args()


def write_json(path: Path, values: dict[str, np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.write_text(json.dumps({key: value.tolist() for key, value in values.items()}, indent=4) + "\n")


def main() -> None:
    args = parse_args()
    files = sorted((args.data_dir / "train").glob("episode*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No train/episode*.hdf5 files under {args.data_dir}")

    data: dict[int, dict[str, np.ndarray]] = {}
    for index, path in enumerate(files):
        with h5py.File(path, "r") as episode:
            data[index] = {
                "actions": episode["action"][:].astype(np.float32),
                "proprio": episode["observations/qpos"][:].astype(np.float32),
            }

    statistics = calculate_dataset_statistics(data)
    normalized_actions = rescale_data(data, statistics, "actions")
    normalized = rescale_data(normalized_actions, statistics, "proprio")
    post_statistics = calculate_dataset_statistics(normalized)

    for key in ("actions", "proprio"):
        values = np.concatenate([episode[key] for episode in normalized.values()], axis=0)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values after {key} normalization")
        # Piper compatibility padding is intentionally all zero after scaling.
        if not np.array_equal(values[:, 7:], np.zeros_like(values[:, 7:])):
            raise ValueError(f"Piper padding dimensions were not preserved as zero for {key}")

    write_json(args.data_dir / "dataset_statistics.json", statistics, args.overwrite)
    write_json(args.data_dir / "dataset_statistics_post_norm.json", post_statistics, args.overwrite)
    total_steps = sum(entry["actions"].shape[0] for entry in data.values())
    print(f"Wrote statistics for {len(data)} episodes / {total_steps} synchronized steps to {args.data_dir}")


if __name__ == "__main__":
    main()
