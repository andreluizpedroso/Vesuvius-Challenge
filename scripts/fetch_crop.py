"""Fetch a small crop of an ink-labels segment (volume + labels) by lazy chunk reads.

Thin wrapper: remote Zarr access is delegated to
`vesuvius.ink_detection.volume_io.open_volume_root` (villa). Only the chunks that
intersect the selected window are downloaded.

Run inside the villa/vesuvius uv environment:
    uv run --no-sync python fetch_crop.py --config <exp.yaml> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import yaml
import zarr

from vesuvius.ink_detection.volume_io import ZARR_V3, open_volume_root


def open_level(base_url: str, name: str, level: str):
    root = open_volume_root(f"{base_url}/{name}")
    return root if hasattr(root, "shape") else root[level]


def as_2d(arr: np.ndarray) -> np.ndarray:
    """Label zarrs may be stored as (1, Y, X); squeeze to (Y, X)."""
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected a single-slice label, got {arr.shape}")
        arr = arr[0]
    return arr


def window_sums(mask: np.ndarray, win: int, stride: int) -> tuple[np.ndarray, list[int], list[int]]:
    """Sum of `mask` over every win x win window placed on a stride grid."""
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = list(range(0, mask.shape[0] - win + 1, stride))
    xs = list(range(0, mask.shape[1] - win + 1, stride))
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    sums = (
        integral[yy + win, xx + win] - integral[yy, xx + win]
        - integral[yy + win, xx] + integral[yy, xx]
    )
    return sums, ys, xs


def select_window(data_cfg: dict) -> dict:
    """Deterministic crop selection on a coarse label level (see exp config)."""
    crop = data_cfg["crop"]
    base, labels = data_cfg["base_url"], data_cfg["labels"]
    sel_level, lvl = crop["select_level"], data_cfg["label_level"]
    factor = 2 ** (int(sel_level) - int(lvl))  # coarse px -> label_level px

    val = as_2d(open_level(base, labels["validation"], sel_level)[...]) > 0
    sup = as_2d(open_level(base, labels["supervision"], sel_level)[...]) > 0
    ink = as_2d(open_level(base, labels["ink"], sel_level)[...]) > 0
    valid = val & sup

    win = crop["size_yx"][0] // factor
    if crop["size_yx"][0] != crop["size_yx"][1] or win * factor != crop["size_yx"][0]:
        raise ValueError("crop.size_yx must be square and divisible by the level factor")
    cov, ys, xs = window_sums(valid, win, crop["select_stride"])
    inkw, _, _ = window_sums(ink & valid, win, crop["select_stride"])
    cov = cov / win**2
    inkf = inkw / win**2

    ok = cov >= crop["min_coverage"]
    score = np.where(ok, inkf, -1.0) if ok.any() else cov
    iy, ix = np.unravel_index(np.argmax(score), score.shape)  # argmax => smallest (y, x) on ties
    y0, x0 = ys[iy] * factor, xs[ix] * factor
    return {
        "bbox_yx": [y0, x0, y0 + crop["size_yx"][0], x0 + crop["size_yx"][1]],
        "coarse": {"level": sel_level, "coverage": float(cov[iy, ix]), "ink_fraction": float(inkf[iy, ix]),
                   "met_min_coverage": bool(ok.any()), "n_candidates_ok": int(ok.sum())},
    }


def check_level_hypothesis(base: str, volume: str, level: str, bbox: list[int]) -> dict:
    """Compare level `level` against a 2x2x2 mean of level 0 on a small block."""
    if level != "1":
        return {"skipped": f"only implemented for level 1 (got {level})"}
    lv0, lv1 = open_level(base, volume, "0"), open_level(base, volume, "1")
    y0, x0 = bbox[0], bbox[1]
    a1 = np.asarray(lv1[0:16, y0:y0 + 64, x0:x0 + 64], dtype=np.float32)
    a0 = np.asarray(lv0[0:32, 2 * y0:2 * y0 + 128, 2 * x0:2 * x0 + 128], dtype=np.float32)
    pooled_zyx = a0.reshape(16, 2, 64, 2, 64, 2).mean(axis=(1, 3, 5))
    pooled_yx = a0[::2].reshape(16, 64, 2, 64, 2).mean(axis=(2, 4))  # alternative: xy-only + z stride
    def stats(ref):
        return {"mae": float(np.abs(ref - a1).mean()),
                "corr": float(np.corrcoef(ref.ravel(), a1.ravel())[0, 1])}
    return {"level0_shape": list(lv0.shape), "level1_shape": list(lv1.shape),
            "vs_mean_2x2x2": stats(pooled_zyx), "vs_xy_mean_z_stride2": stats(pooled_yx)}


def sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def write_volume(path: Path, arr: np.ndarray) -> None:
    chunks = (arr.shape[0], 128, 128)
    if ZARR_V3:
        z = zarr.create_array(store=str(path), shape=arr.shape, chunks=chunks, dtype=arr.dtype)
    else:
        z = zarr.open(str(path), mode="w-", shape=arr.shape, chunks=chunks, dtype=arr.dtype)
    z[...] = arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    data = cfg["data"]
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    sel = select_window(data) if data["crop"]["bbox_yx"] is None else {"bbox_yx": data["crop"]["bbox_yx"]}
    y0, x0, y1, x1 = sel["bbox_yx"]
    print(f"crop bbox (level {data['volume_level']}): y[{y0}:{y1}] x[{x0}:{x1}]", flush=True)

    vol_arr = open_level(data["base_url"], data["volume"], data["volume_level"])
    vol = np.asarray(vol_arr[:, y0:y1, x0:x1])
    write_volume(args.out / "volume.zarr", vol)

    report = {"experiment": cfg["experiment"], "selection": sel,
              "source": {"base_url": data["base_url"], "volume": data["volume"],
                         "volume_level": data["volume_level"], "volume_shape": list(vol_arr.shape)},
              "volume_crop": {"shape": list(vol.shape), "dtype": str(vol.dtype), "sha256": sha256(vol)},
              "labels": {}}
    for key, name in data["labels"].items():
        lab_arr = open_level(data["base_url"], name, data["label_level"])
        if tuple(lab_arr.shape[-2:]) != tuple(vol_arr.shape[-2:]):
            raise ValueError(f"{name} level {data['label_level']} shape {lab_arr.shape} "
                             f"!= volume {vol_arr.shape}")
        sl = (slice(None), slice(y0, y1), slice(x0, x1)) if lab_arr.ndim == 3 else (slice(y0, y1), slice(x0, x1))
        lab = (as_2d(lab_arr[sl]) > 0).astype(np.uint8)
        np.save(args.out / f"{key}.npy", lab)
        report["labels"][key] = {"source": name, "positive_fraction": float(lab.mean()), "sha256": sha256(lab)}

    valid = (np.load(args.out / "validation.npy") > 0) & (np.load(args.out / "supervision.npy") > 0)
    report["exact_coverage_valid"] = float(valid.mean())
    report["level_hypothesis"] = check_level_hypothesis(data["base_url"], data["volume"],
                                                        data["volume_level"], sel["bbox_yx"])
    report["env"] = {"python": platform.python_version(), "numpy": np.__version__, "zarr": zarr.__version__}
    (args.out / "crop.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
