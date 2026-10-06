"""exp007: derive the end-to-end benchmark configs (one JSON per run) from the official recipe.

Uses the exp006 base config logic (same recipe, same overrides), then sets batch size, number of
data-loader workers and the surface-volume root (compressed / uncompressed variant) per run.
Writes runs.json with the per-run environment variables. Validates every config with villa's TrainingConfig.
Run in the villa uv env:
  uv run --no-sync python exp007_make_configs.py --config ... --villa ... --data ... --data-unc ... --out ...
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp006_make_configs import base_config, validate  # noqa: E402

ENV_PRESETS = {"threads1": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--villa", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--data-unc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    base6 = yaml.safe_load((args.config.parents[1] / cfg["base_config"]).read_text())
    recipe = json.loads((args.villa / base6["upstream"]["recipe"]).read_text())
    roots = {"compressed": args.data, "uncompressed": args.data_unc}
    b = cfg["benchmark_runs"]
    args.out.mkdir(parents=True, exist_ok=True)
    made = []
    for run in b["runs"]:
        c = base_config(recipe, base6, roots[run["volume"]])
        c.update({"batch_size": run["batch_size"], "num_iterations": b["num_iterations"], "dataloader_workers": run["workers"],
                  "best_checkpoint_metric": None, "val_every": 10**6, "save_every": b["num_iterations"], "log_every": 25,
                  "out_dir": str(args.out / "runs" / f"shared_{run['volume']}"),
                  "benchmark": {"enabled": True, "warmup_steps": b["warmup_steps"],
                                "output_path": str(args.out / "runs" / f"{run['id']}_benchmark.json")}})
        c["warmup_steps"] = min(c["warmup_steps"], b["warmup_steps"])
        validate(c)
        path = args.out / f"{run['id']}.json"
        path.write_text(json.dumps(c, indent=1))
        made.append({"id": run["id"], "gpus": run["gpus"], "workers": run["workers"], "volume": run["volume"],
                     "env": ENV_PRESETS.get(run.get("env", ""), {}), "env_name": run.get("env", "default"),
                     "config": str(path)})
    (args.out / "runs.json").write_text(json.dumps(made, indent=1))
    print(json.dumps([{k: r[k] for k in ("id", "gpus", "workers", "volume", "env_name")} for r in made], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
