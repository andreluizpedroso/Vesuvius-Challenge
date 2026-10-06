"""exp004 step 1: for each segment, crop the validation-mask bounding box, build the 21-slice input + GT.

Per segment <key> (config data.segments), writes <out>/<key>/{input.zarr, gt_ink.npy, gt_sup.npy, gt_val.npy}
and <out>/prep.json. Conventions (so exp004_eval.py can reuse exp002_eval.py):
  gt_ink = ink label (channel 10) > 0 ; gt_sup = validation mask > 0 (the official validation supervision);
  gt_val = ones. Crop = chunk-aligned bounding box of the non-empty validation-mask chunks, no margin.

Run in the villa uv env:  uv run --no-sync python exp004_prepare.py --config ... --out ...
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_exp001 as dz  # noqa: E402
from exp002_prepare import build_input, sha256  # noqa: E402
from exp003_prepare import CH, list_nonempty  # noqa: E402


def read_region(arr: "dz.ZArray", channel: int, nonempty: set[tuple[int, int]],
                y0: int, y1: int, x0: int, x1: int, workers: int = 8) -> np.ndarray:
    """(y1-y0, x1-x0) plane of one channel; only the listed non-empty chunks are fetched."""
    out = np.zeros((y1 - y0, x1 - x0), arr.dtype)
    jobs = sorted(j for j in nonempty if y0 // CH <= j[0] <= (y1 - 1) // CH and x0 // CH <= j[1] <= (x1 - 1) // CH)

    def one(j):
        iy, ix = j
        c = arr._chunk(0, iy, ix)[channel]
        ya, xa = max(iy * CH, y0), max(ix * CH, x0)
        yb, xb = min((iy + 1) * CH, y1), min((ix + 1) * CH, x1)
        out[ya - y0:yb - y0, xa - x0:xb - x0] = c[ya - iy * CH:yb - iy * CH, xa - ix * CH:xb - ix * CH]

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, jobs))
    return out


def prepare_segment(seg: dict, d: dict, out: Path) -> dict:
    key, ch = seg["key"], d["channel"]
    base, tree = f"{d['labels_root']}/{key}", f"{d['tree_root']}/{key}"
    names = {k: f"{key}_{n}.zarr" for k, n in (("ink", "inklabels"), ("sup", "supervision_mask"),
                                                ("val", "validation_mask"))}
    arrs = {k: dz.ZArray(n, "0", base=base) for k, n in names.items()}
    vol = dz.ZArray("", d["volume_level"], base=seg["volume_url"])
    H, W = vol.shape[1:]
    assert all(a.shape[1:] == (H, W) for a in arrs.values()), "label/volume grids differ"
    ne = {k: list_nonempty(tree, n) for k, n in names.items()}
    ys, xs = [a for a, _ in ne["val"]], [b for _, b in ne["val"]]
    y0, y1 = min(ys) * CH, min((max(ys) + 1) * CH, H)
    x0, x1 = min(xs) * CH, min((max(xs) + 1) * CH, W)
    print(f"[{key}] volume {vol.shape}; validation chunks {len(ne['val'])}; crop y[{y0}:{y1}] x[{x0}:{x1}]", flush=True)

    planes = {k: read_region(arrs[k], ch, ne[k], y0, y1, x0, x1) for k in names}
    gt_ink, val = (planes["ink"] > 0).astype(np.uint8), (planes["val"] > 0).astype(np.uint8)
    sup = (planes["sup"] > 0)

    block = vol.read_yx(y0, y1, x0, x1)                       # (109, h, w)
    assert block.shape[0] >= d["input"]["n_planes"], block.shape
    volume, z0 = build_input(block, d["input"]["n_planes"], d["input"]["z_pool"], d["input"]["xy_pool"])
    del block

    sd = out / key
    sd.mkdir(parents=True, exist_ok=True)
    import zarr
    if int(zarr.__version__.split(".")[0]) >= 3:
        z = zarr.create_array(store=str(sd / "input.zarr"), shape=volume.shape,
                              chunks=(volume.shape[0], 128, 128), dtype=volume.dtype)
    else:
        z = zarr.open(str(sd / "input.zarr"), mode="w-", shape=volume.shape,
                      chunks=(volume.shape[0], 128, 128), dtype=volume.dtype)
    z[...] = volume
    np.save(sd / "gt_ink.npy", gt_ink)
    np.save(sd / "gt_sup.npy", val)                            # evaluation mask = validation mask
    np.save(sd / "gt_val.npy", np.ones_like(val))
    nv = int(val.sum())
    return {"key": key, "crop": {"y0": y0, "y1": y1, "x0": x0, "x1": x1}, "volume_shape": list(vol.shape),
            "input_shape": list(volume.shape), "z_start_plane": z0, "input_sha256": sha256(volume),
            "input_stats": {"mean": float(volume.mean()), "std": float(volume.std())},
            "nonempty_chunks": {k: len(v) for k, v in ne.items()},
            "validation_px": nv, "validation_fraction_of_crop": float(val.mean()),
            "ink_prevalence_in_validation": float(gt_ink[val == 1].mean()) if nv else None,
            "validation_inside_supervision": float(sup[val == 1].mean()) if nv else None,
            "gt_sha256": {"ink": sha256(gt_ink), "val": sha256(val)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    rec = {"experiment": cfg["experiment"], "segments": []}
    for seg in cfg["data"]["segments"]:
        rec["segments"].append(prepare_segment(seg, cfg["data"], args.out))
        rec["fetch_stats"] = dict(dz._fetch_stats)
        (args.out / "prep.json").write_text(json.dumps(rec, indent=1))   # partial progress survives a crash
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
