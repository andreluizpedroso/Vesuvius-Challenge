"""exp009: derive the reproduction training config from the official ink_9um recipe.

Takes the recipe JSON of the pinned villa checkout, keeps its 'aligned' dataset entries (drops the native 0139
entry), points every entry at the mounted exp008 corpus of its scroll, applies the config overrides and validates
the result with villa's own TrainingConfig. Segments missing from the mounted corpus are dropped with a loud warning
(a failed corpus segment must not silently change the experiment: the list is printed and written to the output).
Run in the villa uv env:
  uv run --no-sync python exp009_make_train_config.py --config ... --villa ... --corpus-a ... --corpus-b ... \
      --out cfg.json --out-dir run_dir [--smoke]
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


def build(recipe: dict, cfg: dict, roots: dict[str, Path], out_dir: Path, smoke: bool) -> tuple[dict, dict]:
    c = copy.deepcopy(recipe)
    datasets, report = [], {"kept": {}, "missing": {}, "dropped_entries": []}
    for entry in recipe["datasets"]:
        if "aligned" not in entry["segments_path"]:
            report["dropped_entries"].append(entry["segments_path"].rsplit("/", 1)[-1])
            continue
        scroll = entry["sampling_scroll"]
        root = roots[cfg["scroll_group"][scroll]] / cfg["segments_dir"]
        present = [s for s in entry["segments"] if (root / s / "surface-volume.zarr").exists()
                   and (root / s / f"{s}_inklabels.zarr").exists()]
        report["kept"][scroll] = present
        report["missing"][scroll] = [s for s in entry["segments"] if s not in present]
        if not present:
            continue
        e = copy.deepcopy(entry)
        e["segments_path"] = str(root)
        e["segments"] = present
        e["surface_volume_paths"] = {s: str(root / s / "surface-volume.zarr") for s in present}
        for key in ("sampling_physical_segment_keys", "sampling_representation_keys"):
            if key in e:
                e[key] = {s: v for s, v in e[key].items() if s in present}
        datasets.append(e)
    c["datasets"] = datasets
    for k, v in cfg["overrides"].items():
        c[k] = v
    if smoke:
        sm = cfg["smoke"]
        c.update({"num_iterations": sm["num_iterations"], "save_every": sm["save_every"], "val_every": sm["val_every"],
                  "val_steps": 4})
    c["out_dir"] = str(out_dir)
    return c, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--villa", type=Path, required=True)
    ap.add_argument("--corpus-a", type=Path, required=True)
    ap.add_argument("--corpus-b", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    recipe = json.loads((args.villa / cfg["upstream"]["recipe"]).read_text())
    c, report = build(recipe, cfg, {"A": args.corpus_a, "B": args.corpus_b}, args.out_dir, args.smoke)
    validate(c)
    args.out.write_text(json.dumps(c, indent=1))
    n = sum(len(v) for v in report["kept"].values())
    print(json.dumps({"segments_kept": n, "per_scroll": {k: len(v) for k, v in report["kept"].items()},
                      "missing": {k: v for k, v in report["missing"].items() if v},
                      "dropped_entries": report["dropped_entries"], "num_iterations": c["num_iterations"],
                      "max_steps": c.get("max_steps"), "batch_size": c["batch_size"],
                      "sampling": c["sampling_strategy"], "prior": c.get("fixed_scroll_prior")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
