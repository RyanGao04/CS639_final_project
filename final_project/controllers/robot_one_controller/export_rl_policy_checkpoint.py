"""Export a trained 16->64->64->2 tanh actor to starter_controller.py arrays."""

import argparse
import json
from pathlib import Path

import torch


EXPECTED_SHAPES = {
    "w1": (64, 16),
    "b1": (64,),
    "w2": (64, 64),
    "b2": (64,),
    "w3": (2, 64),
    "b3": (2,),
}


def _to_serializable(array):
    return [[round(float(v), 8) for v in row] for row in array] if array.ndim == 2 else [round(float(v), 8) for v in array]


def _extract_arrays(state_dict):
    direct_keys = ("w1", "b1", "w2", "b2", "w3", "b3")
    if all(key in state_dict for key in direct_keys):
        arrays = {key: state_dict[key].detach().cpu().numpy() for key in direct_keys}
    else:
        candidate_sets = [
            {
                "w1": "actor.0.weight",
                "b1": "actor.0.bias",
                "w2": "actor.2.weight",
                "b2": "actor.2.bias",
                "w3": "actor.4.weight",
                "b3": "actor.4.bias",
            },
            {
                "w1": "net.0.weight",
                "b1": "net.0.bias",
                "w2": "net.2.weight",
                "b2": "net.2.bias",
                "w3": "net.4.weight",
                "b3": "net.4.bias",
            },
        ]
        arrays = None
        for candidate in candidate_sets:
            if all(key in state_dict for key in candidate.values()):
                arrays = {
                    target: state_dict[source].detach().cpu().numpy()
                    for target, source in candidate.items()
                }
                break
        if arrays is None:
            keys = sorted(state_dict.keys())
            raise KeyError(f"Unsupported checkpoint keys: {keys}")

    for key, expected_shape in EXPECTED_SHAPES.items():
        if tuple(arrays[key].shape) != expected_shape:
            raise ValueError(f"{key} has shape {arrays[key].shape}, expected {expected_shape}")
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a torch checkpoint or state_dict.")
    parser.add_argument(
        "--format",
        choices=("json", "python"),
        default="python",
        help="Output format for pasted weights.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    arrays = _extract_arrays(state_dict)
    serializable = {key: _to_serializable(value) for key, value in arrays.items()}

    if args.format == "json":
        print(json.dumps(serializable))
        return

    print("Replace EmbeddedActorPolicy._build_bootstrap_actor() with the arrays below:\n")
    for key in ("w1", "b1", "w2", "b2", "w3", "b3"):
        print(f"{key} = np.array({serializable[key]}, dtype=np.float32)")


if __name__ == "__main__":
    main()
