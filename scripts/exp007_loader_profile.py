"""exp007 A+B: profile the ink training data loader in isolation (no model, no GPU).

Builds the same InkDataset the villa trainer builds (from one generated training config JSON) and measures:
  - single-process samples/s with augmentation on / off
  - cProfile of dataset[i] (top functions by own time) with augmentation on / off
  - DataLoader throughput (examples/s) and CPU busy fraction vs number of workers (augmentation on),
    plus one augmentation-off reference at 4 workers
Each section is isolated: an exception is recorded under "errors" and the others still run.
Environment variables set before this script starts (e.g. OMP_NUM_THREADS=1) are inherited by the workers.
Run in the villa uv env:  uv run --no-sync python exp007_loader_profile.py --train-config cfg.json --params cfg.yaml --out out.json
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import random
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp007_cpu_monitor import read_cpu  # noqa: E402


def load_datasets(train_config: Path):
    from vesuvius.ink_detection.config import TrainingConfig, resolve_training_mapping
    from vesuvius.ink_detection.data.dataset import InkDataset
    from vesuvius.ink_detection.training.train import training_dataset_config

    cfg = TrainingConfig.from_mapping(resolve_training_mapping(json.loads(train_config.read_text())))
    dc = training_dataset_config(cfg)
    shared = InkDataset(dc, do_augmentations=False)
    patches, segments = shared.training_patches, shared.segments
    ds_aug = InkDataset(dc, do_augmentations=True, patches=patches, segments=segments)
    ds_plain = InkDataset(dc, do_augmentations=False, patches=patches, segments=segments)
    return ds_aug, ds_plain, len(patches), cfg


def sample_indices(n_items: int, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(n_items) for _ in range(n)]


def time_samples(ds, idx: list[int]) -> float:
    t0 = time.perf_counter()
    for i in idx:
        ds[i]
    return len(idx) / (time.perf_counter() - t0)


def profile_top(ds, idx: list[int], top: int = 14) -> dict:
    pr = cProfile.Profile()
    pr.enable()
    for i in idx:
        ds[i]
    pr.disable()
    st = pstats.Stats(pr)
    total = sum(v[2] for v in st.stats.values())
    rows = sorted(st.stats.items(), key=lambda kv: -kv[1][2])[:top]
    return {"total_own_time_s": round(total, 3),
            "top": [{"func": f"{'/'.join(Path(k[0]).parts[-2:])}:{k[2]}", "calls": v[1], "own_s": round(v[2], 3),
                     "own_pct": round(100 * v[2] / total, 1), "cum_s": round(v[3], 3)} for k, v in rows]}


def sweep_workers(ds, workers: int, batch_size: int, warm: int, timed: int) -> dict:
    from torch.utils.data import DataLoader
    kw = {}
    if workers > 0:
        kw = {"multiprocessing_context": "spawn", "persistent_workers": True, "prefetch_factor": 2}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False,
                    drop_last=True, **kw)
    it = iter(dl)
    for _ in range(warm):
        next(it)
    c0, t0 = read_cpu("/proc/stat"), time.perf_counter()
    for _ in range(timed):
        next(it)
    dt, c1 = time.perf_counter() - t0, read_cpu("/proc/stat")
    busy = 1 - (c1[1] - c0[1]) / max(c1[0] - c0[0], 1e-9)
    del it, dl
    return {"examples_per_s": round(timed * batch_size / dt, 1), "cpu_busy": round(busy, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-config", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True, help="exp007 yaml (loader_profile section)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = yaml.safe_load(args.params.read_text())["loader_profile"]
    res: dict = {"env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                 "cpus": os.cpu_count(), "errors": {}}

    def guarded(name, fn):
        try:
            res[name] = fn()
        except Exception:  # noqa: BLE001 - diagnostics must not stop the other sections
            res["errors"][name] = traceback.format_exc()[-1500:]
            print(f"[{name}] FAILED\n{res['errors'][name]}", flush=True)

    ds_aug, ds_plain, n_patches, _ = load_datasets(args.train_config)
    res["n_training_patches"] = n_patches
    idx = sample_indices(len(ds_aug), p["profile_samples"], p["seed"])
    ds_aug[idx[0]]                                              # warm caches (zarr opens, imports)
    guarded("single_process_examples_per_s", lambda: {"aug_on": round(time_samples(ds_aug, idx), 1),
                                                       "aug_off": round(time_samples(ds_plain, idx), 1)})
    guarded("profile_aug_on", lambda: profile_top(ds_aug, idx))
    guarded("profile_aug_off", lambda: profile_top(ds_plain, idx))
    res["sweep_aug_on"] = {}
    for w in p["worker_sweep"]:
        timed = 6 if w == 0 else p["sweep_timed_batches"]
        try:
            res["sweep_aug_on"][str(w)] = sweep_workers(ds_aug, w, p["sweep_batch_size"], p["sweep_warmup_batches"], timed)
            print(f"[sweep] workers={w}: {res['sweep_aug_on'][str(w)]}", flush=True)
        except Exception:  # noqa: BLE001
            res["errors"][f"sweep_w{w}"] = traceback.format_exc()[-1500:]
    guarded("sweep_aug_off_w4", lambda: sweep_workers(ds_plain, 4, p["sweep_batch_size"], p["sweep_warmup_batches"], p["sweep_timed_batches"]))
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
