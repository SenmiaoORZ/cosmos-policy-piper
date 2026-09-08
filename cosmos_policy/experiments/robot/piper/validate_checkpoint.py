"""Load a Piper Cosmos Policy DCP checkpoint as an inference model.

This is a deployment-readiness check: it exercises the same local DCP path
used after a Hugging Face snapshot download, without starting robot control.
"""

import argparse

import torch

from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint


DEFAULT_EXPERIMENT = "cosmos_predict2_2b_480p_piper_sponge_overfit"
DEFAULT_CONFIG = "cosmos_policy/config/config.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local DCP model directory (contains .metadata).")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        s3_checkpoint_dir=args.checkpoint,
        config_file=args.config_file,
        enable_fsdp=False,
        load_ema_to_reg=False,
        instantiate_ema=False,
        to_device=args.device,
    )
    model.eval()
    first_parameter = next(model.parameters())
    if first_parameter.device.type != torch.device(args.device).type:
        raise RuntimeError(f"Model was loaded on {first_parameter.device}, expected {args.device}.")
    print(f"DCP_INFERENCE_LOAD_OK model={type(model).__name__} device={first_parameter.device}")


if __name__ == "__main__":
    main()
