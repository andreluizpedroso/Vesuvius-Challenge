"""exp006 step 1: build a SPARSE copy of the ink_9um training layout for a few segments.

For each segment <key> writes, under <out>/labels/aligned-scrollprizeorg-21slices/<key>/:
  <key>_{inklabels,supervision_mask,validation_mask}.zarr   byte-exact copies of the non-empty label chunks
                                                           (empty chunks are simply absent = zeros)
  surface-volume.zarr                                       (21, H, W) uint8, zarr v2, 21 slices of 9.6 um
                                                           (84 centered planes of the 2.4 um level-2 volume,
                                                           mean of 4), only for the chunks the trainer can touch:
                                                           non-empty label chunks dilated by `chunk_margin`.
Everything else in the volume is absent (zeros) - fine because patches are only drawn on supervised
areas. Reuses diag_exp001.ZArray, exp003_prepare.list_nonempty and exp002_prepare.build_input.

Run in the villa uv env:  uv run --no-sync python exp006_prepare_train_data.py --config ... --out ...
"""

from __future__ import annotations

import argparse
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
from exp002_prepare import build_input  # noqa: E402
from exp003_prepare import CH, list_nonempty  # noqa: E402

KINDS = ("inklabels", "supervision_mask", "validation_mask")


def copy_label_zarr(base: str, tree: str, key: str, kind: str, dest: Path) -> set[tuple[int, int]]:
    """Copy .zgroup/.zattrs/0/.zarray (+optional 0/.zattrs) and every non-empty chunk, byte for byte."""
    name = f"{key}_{kind}.zarr"
    root = dest / key / name
    (root / "0").mkdir(parents=True, exist_ok=True)
    for rel, optional in ((".zgroup", False), (".zattrs", False), ("0/.zarray", False), ("0/.zattrs", True)):
        raw = dz.fetch(f"{base}/{key}/{name}/{rel}", allow_missing=optional)
        if raw is not None:
            (root / rel).write_bytes(raw)
    meta = json.loads((root / "0/.zarray").read_text())
    sep = meta.get("dimension_separator", ".")
    assert sep == ".", f"unexpected dimension_separator {sep!r} in {name}"
    nonempty = list_nonempty(tree, name)

    def one(j):
        iy, ix = j
        raw = dz.fetch(f"{base}/{key}/{name}/0/0.{iy}.{ix}")
        (root / "0" / f"0.{iy}.{ix}").write_bytes(raw)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, sorted(nonempty)))
    return nonempty


def dilate(chunks: set[tuple[int, int]], margin: int, ny: int, nx: int) -> set[tuple[int, int]]:
    out = set()
    for iy, ix in chunks:
        for dy in range(-margin, margin + 1):
            for dx in range(-margin, margin + 1):
                if 0 <= iy + dy < ny and 0 <= ix + dx < nx:
                    out.add((iy + dy, ix + dx))
    return out


def prepare_segment(seg: dict, d: dict, out: Path) -> dict:
    key = seg["key"]
    dest = out / "labels" / "aligned-scrollprizeorg-21slices"
    labels = {k: copy_label_zarr(d["labels_root"], d["tree_root"], key, k, dest) for k in KINDS}
    vol = dz.ZArray("", d["volume_level"], base=seg["volume_url"])
    H, W = vol.shape[1:]
    meta = json.loads((dest / key / f"{key}_inklabels.zarr" / "0/.zarray").read_text())
    assert tuple(meta["shape"][1:]) == (H, W), (meta["shape"], vol.shape)
    ny, nx = -(-H // CH), -(-W // CH)
    need = dilate(set().union(*labels.values()), d["chunk_margin"], ny, nx)
    print(f"[{key}] volume {vol.shape}; label chunks " + str({k: len(v) for k, v in labels.items()}) +
          f"; volume chunks to fetch {len(need)} (~{len(need) * 1.78 / 1000:.2f} GB)", flush=True)

    import zarr
    path = dest / key / "surface-volume.zarr"
    arr = zarr.create_array(store=str(path), shape=(d["input"]["n_planes"] // d["input"]["z_pool"], H, W),
                            chunks=(d["input"]["n_planes"] // d["input"]["z_pool"], CH, CH),
                            dtype="uint8", fill_value=0, zarr_format=2)

    def one(j):
        iy, ix = j
        block = vol._chunk(0, iy, ix)                             # (109, 128, 128) uint8
        pooled, _ = build_input(block, d["input"]["n_planes"], d["input"]["z_pool"], 1)
        return iy, ix, pooled

    nonzero = 0
    with ThreadPoolExecutor(8) as pool:
        for iy, ix, pooled in pool.map(one, sorted(need)):
            y0, x0 = iy * CH, ix * CH
            y1, x1 = min(y0 + CH, H), min(x0 + CH, W)
            arr[:, y0:y1, x0:x1] = pooled[:, : y1 - y0, : x1 - x0]
            nonzero += int(pooled.any())
    return {"key": key, "shape": [21, H, W], "label_nonempty_chunks": {k: len(v) for k, v in labels.items()},
            "volume_chunks_fetched": len(need), "volume_chunks_nonzero": nonzero}


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
    (args.out / "train_data.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
