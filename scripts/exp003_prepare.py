"""exp003 step 1: select a window of Paris 4 w00, build the official 21-slice input and the GT.

Volume: public S3 2.4 um surface volume, level 2 (uncompressed chunks of 109x128x128).
Labels: ink_9um training labels (21 slices, annotated at one channel), HF bucket.
Reuses the zarr-v2 HTTP reader (diag_exp001.ZArray) and build_input (exp002_prepare).
Writes <out>/input.zarr, gt_{ink,sup,val}.npy, prep.json (same contract as exp002_prepare,
so exp002_eval.py is reused unchanged; gt_val is all ones because Paris 4 has no validation mask).

Run in the villa uv env:  uv run --no-sync python exp003_prepare.py --config ... --out ...
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_exp001 as dz  # noqa: E402
from exp002_prepare import build_input, sha256  # noqa: E402

CH = 128


def list_nonempty(tree_url: str, name: str) -> set[tuple[int, int]]:
    """(iy, ix) of chunks whose stored size exceeds the all-zero chunk size (the modal size).

    Chunk keys are '<iz>.<iy>.<ix>' under '<name>/0/'. An all-zero chunk compresses to a tiny
    constant size (78 bytes here), so everything larger carries data; the rest decode to zeros.
    """
    url, entries = f"{tree_url}/{name}/0?recursive=true", []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            entries += json.load(r)
            link = r.headers.get("Link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    chunks = {}
    for e in entries:
        key = e["path"].rsplit("/", 1)[-1]
        if e["type"] == "file" and re.fullmatch(r"\d+\.\d+\.\d+", key):
            _, iy, ix = (int(t) for t in key.split("."))
            chunks[(iy, ix)] = e["size"]
    empty = collections.Counter(chunks.values()).most_common(1)[0][0]
    return {k for k, v in chunks.items() if v > empty}


def read_plane(arr: "dz.ZArray", channel: int, nonempty: set[tuple[int, int]], workers: int = 8) -> np.ndarray:
    """Whole (Y, X) plane of one channel, fetching only the chunks listed as non-empty."""
    H, W = arr.shape[1:]
    out = np.zeros((-(-H // CH) * CH, -(-W // CH) * CH), arr.dtype)

    def one(j):
        iy, ix = j
        out[iy * CH:(iy + 1) * CH, ix * CH:(ix + 1) * CH] = arr._chunk(0, iy, ix)[channel]

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, sorted(nonempty)))
    return out[:H, :W]


def select_window(ink: np.ndarray, sup: np.ndarray, size: int, stride: int, min_cov: float) -> dict:
    """Deterministic rule shared with exp002; windows are aligned to the 128-px chunk grid."""
    assert size % CH == 0 and stride % CH == 0
    ty, tx = ink.shape[0] // CH, ink.shape[1] // CH
    def tiles(a):
        return a[:ty * CH, :tx * CH].reshape(ty, CH, tx, CH).sum(axis=(1, 3), dtype=np.int64)
    t_sup, t_ink = tiles(sup > 0), tiles((ink > 0) & (sup > 0))
    n, step = size // CH, stride // CH
    def integral(t):
        return np.pad(t.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    I_s, I_i = integral(t_sup), integral(t_ink)
    ys, xs = list(range(0, ty - n + 1, step)), list(range(0, tx - n + 1, step))
    best, best_key, fallback = None, None, None
    for y in ys:
        for x in xs:
            s = I_s[y + n, x + n] - I_s[y, x + n] - I_s[y + n, x] + I_s[y, x]
            i = I_i[y + n, x + n] - I_i[y, x + n] - I_i[y + n, x] + I_i[y, x]
            cov, inkf = s / size**2, i / size**2
            if fallback is None or cov > fallback[0]:
                fallback = (cov, inkf, y, x)
            if cov >= min_cov and (best is None or inkf > best[1]):  # strict > keeps smallest (y, x) on ties
                best = (cov, inkf, y, x)
    chosen, met = (best, True) if best is not None else (fallback, False)
    cov, inkf, y, x = chosen
    return {"y0": int(y * CH), "x0": int(x * CH), "size": size, "coverage": float(cov),
            "ink_fraction": float(inkf), "met_min_coverage": met}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    d, inp, lab = cfg["data"], cfg["data"]["input"], cfg["data"]["labels"]
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    ink_arr = dz.ZArray(lab["ink"], "0", base=lab["base_url"])
    sup_arr = dz.ZArray(lab["supervision"], "0", base=lab["base_url"])
    vol = dz.ZArray("", d["volume"]["level"], base=d["volume"]["base_url"])
    print("shapes: ink", ink_arr.shape, "sup", sup_arr.shape, "volume", vol.shape, flush=True)
    assert ink_arr.shape[1:] == sup_arr.shape[1:] == vol.shape[1:], "label/volume grids differ"

    ne_ink = list_nonempty(lab["tree_url"], lab["ink"])
    ne_sup = list_nonempty(lab["tree_url"], lab["supervision"])
    print("non-empty label chunks: ink", len(ne_ink), "sup", len(ne_sup), flush=True)
    ink = read_plane(ink_arr, lab["channel"], ne_ink)
    sup = read_plane(sup_arr, lab["channel"], ne_sup)
    print("label planes read; ink px", int((ink > 0).sum()), "sup px", int((sup > 0).sum()), flush=True)
    w = select_window(ink, sup, d["window"]["size"], d["window"]["stride"], d["window"]["min_coverage"])
    print("window:", w, flush=True)
    y0, x0, S = w["y0"], w["x0"], w["size"]

    block = vol.read_yx(y0, y0 + S, x0, x0 + S)                    # (109, S, S) uint8
    assert block.shape[0] >= inp["n_planes"], block.shape
    volume, z0 = build_input(block, inp["n_planes"], inp["z_pool"], inp["xy_pool"])
    del block
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

    win = (slice(y0, y0 + S), slice(x0, x0 + S))
    gt_ink, gt_sup = (ink[win] > 0).astype(np.uint8), (sup[win] > 0).astype(np.uint8)
    np.save(args.out / "gt_ink.npy", gt_ink)
    np.save(args.out / "gt_sup.npy", gt_sup)
    np.save(args.out / "gt_val.npy", np.ones_like(gt_sup))        # no validation mask for Paris 4

    # Which channels of the label zarr carry data in this window (expect only the annotation channel)
    nz = {}
    for key, arr in (("ink", ink_arr), ("supervision", sup_arr)):
        sub = arr.read_yx(y0, y0 + S, x0, x0 + S)
        nz[key] = [int(c) for c in np.where((sub > 0).any(axis=(1, 2)))[0]]
    rec = {"experiment": cfg["experiment"], "window": w, "input_shape": list(volume.shape),
           "input_sha256": sha256(volume), "z_start_plane": z0,
           "input_stats": {"mean": float(volume.mean()), "std": float(volume.std())},
           "label_nonzero_channels": nz, "nonempty_label_chunks": {"ink": len(ne_ink), "sup": len(ne_sup)}, "annotation_channel": lab["channel"],
           "labels": {"ink": {"positive_fraction": float(gt_ink.mean()), "sha256": sha256(gt_ink)},
                      "supervision": {"positive_fraction": float(gt_sup.mean()), "sha256": sha256(gt_sup)}},
           "fetch_stats": dict(dz._fetch_stats)}
    (args.out / "prep.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
