"""exp006: derive the benchmark/functional training configs from the official ink_9um recipe.

Reads the recipe JSON from the pinned villa checkout, applies ONLY the overrides of the experiment
config, writes one JSON per run and validates each with villa's own TrainingConfig.
Run inside the villa uv env (needs `vesuvius` importable):
  uv run --no-sync python exp006_make_configs.py --config ... --villa ... --data ... --out ...
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


def validate(mapping: dict) -> None:
    from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping
    TrainingConfig.from_mapping(resolve_training_mapping(mapping))


def base_config(recipe: dict, cfg: dict, data_root: Path) -> dict:
    c = copy.deepcopy(recipe)
    segs_path = data_root / "labels" / "aligned-scrollprizeorg-21slices"
    keys = [s["key"] for s in cfg["data"]["segments"]]
    c["datasets"] = [{
        "segments_path": str(segs_path),
        "segments": keys,
        "surface_volume_paths": {k: str(segs_path / k / "surface-volume.zarr") for k in keys},
        "volume_scale": 0,
    }]
    c.pop("fixed_scroll_prior", None)
    for k, v in cfg["overrides"].items():
        c[k] = v
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--villa", type=Path, required=True, help="villa checkout root")
    ap.add_argument("--data", type=Path, required=True, help="output of exp006_prepare_train_data")
    ap.add_argument("--out", type=Path, required=True, help="dir for configs and run outputs")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    recipe = json.loads((args.villa / cfg["upstream"]["recipe"]).read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    base = base_config(recipe, cfg, args.data)
    made = []

    b = cfg["benchmark_runs"]
    for run in b["runs"]:
        c = copy.deepcopy(base)
        c.update({"batch_size": run["batch_size"], "num_iterations": b["num_iterations"],
                  "best_checkpoint_metric": None, "val_every": 10**6, "save_every": b["num_iterations"],
                  "log_every": 25, "out_dir": str(args.out / "runs" / "bench_shared"),
                  "benchmark": {"enabled": True, "warmup_steps": b["warmup_steps"],
                                "output_path": str(args.out / "runs" / f"{run['id']}_benchmark.json")}})
        c["warmup_steps"] = min(c["warmup_steps"], b["warmup_steps"])
        validate(c)
        path = args.out / f"{run['id']}.json"
        path.write_text(json.dumps(c, indent=1))
        made.append({"id": run["id"], "gpus": run["gpus"], "config": str(path), "kind": "benchmark"})

    f = cfg["functional_run"]
    c = copy.deepcopy(base)
    c.update({"batch_size": f["batch_size"], "num_iterations": f["num_iterations"], "val_every": f["val_every"],
              "val_steps": f["val_steps"], "save_every": f["save_every"], "log_every": 25,
              "out_dir": str(args.out / "runs" / f["id"])})
    c["warmup_steps"] = min(c["warmup_steps"], 30)
    validate(c)
    path = args.out / f"{f['id']}.json"
    path.write_text(json.dumps(c, indent=1))
    made.append({"id": f["id"], "gpus": f["gpus"], "config": str(path), "kind": "functional"})

    (args.out / "runs.json").write_text(json.dumps(made, indent=1))
    print(json.dumps(made, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
