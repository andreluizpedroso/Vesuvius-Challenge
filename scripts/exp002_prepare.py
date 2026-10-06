"""exp002 step 1: build the 21-slice ~9.36 um input and the 2D ground truth for 841/w00.

Reads the level-0 window (~270 MB of uncompressed chunks) and the z=32 label planes over HTTP,
reusing the zarr-v2 reader from diag_exp001.py. Writes:
  <out>/input.zarr     (21, H, W) uint8, bare 3D array for vesuvius infer
  <out>/gt_{ink,sup,val}.npy   (H, W) uint8
  <out>/prep.json      shapes, hashes, sanity checks

Run inside the villa/vesuvius uv env:  uv run --no-sync python exp002_prepare.py --config ... --out ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_exp001 as dz  # noqa: E402  (zarr-v2 HTTP reader; BASE/VOLUME/LABELS are module globals)


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def pool_xy(a: np.ndarray, f: int) -> np.ndarray:
    z, h, w = a.shape
    return a.astype(np.float32).reshape(z, h // f, f, w // f, f).mean(axis=(2, 4))


def build_input(vol: np.ndarray, n_planes: int, z_pool: int, xy_pool: int) -> tuple[np.ndarray, int]:
    """Centered `n_planes` -> (n_planes/z_pool, H/xy_pool, W/xy_pool) uint8, single rounding."""
    z0 = math.ceil((vol.shape[0] - n_planes) / 2)  # same rule as villa's centered_slice
    out = np.empty((n_planes // z_pool, vol.shape[1] // xy_pool, vol.shape[2] // xy_pool), np.uint8)
    for k in range(out.shape[0]):
        block = vol[z0 + k * z_pool: z0 + (k + 1) * z_pool].astype(np.float32)
        out[k] = np.rint(pool_xy(block, xy_pool).mean(axis=0)).astype(np.uint8)
    return out, z0


def pyramid_z_check(vol0: np.ndarray, y0: int, x0: int) -> dict:
    """level1[i] should equal xy-pooled level0[i] (z not pooled), not level0[2i]."""
    size = 256
    l1y, l1x = (y0 // 2) + 384, (x0 // 2) + 384          # level-1 coords, chunk aligned
    l1 = dz.ZArray(dz.VOLUME, "1").read_yx(l1y, l1y + size, l1x, l1x + size)  # (33, 256, 256)
    oy, ox = 2 * 384, 2 * 384
    sub = vol0[:, oy:oy + 2 * size, ox:ox + 2 * size]
    pooled = pool_xy(sub, 2)                                                  # (65, 256, 256)
    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    c_1to1 = [corr(l1[i], pooled[i]) for i in range(l1.shape[0])]
    c_2i = [corr(l1[i], pooled[2 * i]) for i in range(l1.shape[0]) if 2 * i < pooled.shape[0]]
    return {"corr_1to1_mean": float(np.mean(c_1to1)), "corr_1to1_min": float(np.min(c_1to1)),
            "corr_2i_mean": float(np.mean(c_2i)), "n_planes_l1": int(l1.shape[0])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    d, inp = cfg["data"], cfg["data"]["input"]
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    dz.BASE, dz.VOLUME, dz.LABELS = d["base_url"], d["volume"], dict(d["labels"])
    w = d["window_l0"]; y0, x0, S = w["y0"], w["x0"], w["size"]

    vol0 = dz.ZArray(dz.VOLUME, "0").read_yx(y0, y0 + S, x0, x0 + S)
    assert vol0.shape == (65, S, S), vol0.shape
    print("level-0 window read", vol0.shape, flush=True)

    check = pyramid_z_check(vol0, y0, x0)
    print("pyramid z check:", check, flush=True)
    ok = (check["corr_1to1_mean"] >= inp["sanity_min_corr"]
          and check["corr_1to1_mean"] > check["corr_2i_mean"] + 0.1)
    if not ok:
        (args.out / "prep.json").write_text(json.dumps({"FAILED_sanity": check}, indent=1))
        raise SystemExit(f"Sanity check failed (levels are not z-truncated xy-pools?): {check}")

    volume, z0 = build_input(vol0, inp["n_planes"], inp["z_pool"], inp["xy_pool"])
    del vol0
    print("input built", volume.shape, "planes start", z0, flush=True)

    import zarr
    path = args.out / "input.zarr"
    if int(zarr.__version__.split(".")[0]) >= 3:
        z = zarr.create_array(store=str(path), shape=volume.shape, chunks=(volume.shape[0], 128, 128),
                              dtype=volume.dtype)
    else:
        z = zarr.open(str(path), mode="w-", shape=volume.shape, chunks=(volume.shape[0], 128, 128),
                      dtype=volume.dtype)
    z[...] = volume

    rec = {"experiment": cfg["experiment"], "window_l0": w, "input_shape": list(volume.shape),
           "input_sha256": sha256(volume), "z_start_plane": z0,
           "input_stats": {"mean": float(volume.mean()), "std": float(volume.std())},
           "pyramid_z_check": check, "labels": {}}
    lp = inp["label_plane"]
    gt = cfg["data"]["gt"]
    for key, name in d["labels"].items():
        lab = dz.ZArray(name, "0").read_yx(y0, y0 + S, x0, x0 + S)
        plane = lab[lp] > 0
        other = int((lab > 0).any(axis=(1, 2)).sum())
        blocks = plane.reshape(S // 2, 2, S // 2, 2).mean(axis=(1, 3))
        arr = (blocks >= gt["ink_min_fraction"]) if key == "ink" else (blocks == 1.0)
        arr = arr.astype(np.uint8)
        np.save(args.out / f"gt_{'sup' if key == 'supervision' else 'val' if key == 'validation' else 'ink'}.npy", arr)
        rec["labels"][key] = {"nonzero_planes": other, "positive_fraction": float(arr.mean()),
                              "plane_positive_fraction_l0": float(plane.mean()), "sha256": sha256(arr)}
        del lab
    rec["fetch_stats"] = dict(dz._fetch_stats)
    (args.out / "prep.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
