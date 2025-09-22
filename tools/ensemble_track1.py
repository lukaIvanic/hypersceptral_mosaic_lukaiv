import argparse
import json
import numpy as np
from pathlib import Path


def load_npz(dir_path: str, ids: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for sid in ids:
        p = Path(dir_path) / f"{sid}.npz"
        data = np.load(p)
        out[sid] = data["cube"]  # (H,W,C)
    return out


def main(args):
    with open(args.ids_json, "r") as f:
        ids = json.load(f)["ids"]
    pred_dirs = [Path(p) for p in args.pred_dirs]
    ws = np.array(args.weights, dtype=np.float32)
    ws = ws / ws.sum()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load all predictions per model directory
    preds = [load_npz(str(d), ids) for d in pred_dirs]

    for sid in ids:
        cubes = [preds[m][sid] for m in range(len(pred_dirs))]  # each (H,W,C)
        # Weighted blend
        acc = np.zeros_like(cubes[0], dtype=np.float32)
        for w, cube in zip(ws, cubes):
            acc += w * cube
        np.savez_compressed(Path(args.out_dir) / f"{sid}.npz", cube=acc)

    print(f"[OK] Wrote blended predictions to {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dirs", type=str, nargs="+", required=True)
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--ids_json", type=str, required=True, help='JSON file {"ids": ["id1", ...]}')
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()
    main(args)

