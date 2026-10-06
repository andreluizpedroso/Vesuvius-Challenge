"""exp001 diagnostics for ink-labels 841/w00 (CPU only, a few tens of MB of reads).

Questions answered (each result is written to diag.json):
  A. Metadata: pyramid shapes/chunks/compressors, volume vs label grids.
  B. Physical pixel size of the surface-volume canvas, from tifxyz geometry.
  C. How pyramid levels relate (z/xy pooling mode, offsets), by correlation.
  D. Where along z the label masks live (per-z positive fraction).
  E. Ink contrast (Cohen's d) per z at pyramid levels 1 and 2.

Only numeric summaries are produced; no images are written.
Dependencies: numpy, pyyaml, numcodecs (blosc), tifffile.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = "https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/841/w00"
LABELS = {"ink": "w00_inklabels_v2.zarr", "supervision": "w00_supervision_mask_v2.zarr",
          "validation": "w00_validation_mask_v2.zarr"}
VOLUME = "w00.zarr"
# Window used by exp000 (level-1 pixel coordinates): y 5504:6528, x 5248:6272.
WIN_L1 = (5504, 5248)
CHUNK = 128

_fetch_stats = {"requests": 0, "bytes": 0, "missing": 0}


def fetch(url: str, allow_missing: bool = False) -> bytes | None:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            _fetch_stats["requests"] += 1
            _fetch_stats["bytes"] += len(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404 and allow_missing:
                _fetch_stats["missing"] += 1
                return None
            if attempt == 3:
                raise
        except Exception:
            if attempt == 3:
                raise
    return None


def fetch_json(rel: str, base: str | None = None) -> dict:
    return json.loads(fetch(f"{base or BASE}/{rel}"))


def decode_chunk(raw: bytes, compressor: dict | None) -> bytes:
    if compressor is None:
        return raw
    if compressor.get("id") != "blosc":
        raise NotImplementedError(compressor)
    from numcodecs import Blosc
    return bytes(Blosc().decode(raw))


class ZArray:
    """Minimal zarr-v2 reader (uncompressed or blosc), enough for chunk-aligned windows."""

    def __init__(self, name: str, level: str, base: str | None = None):
        self.base = base or BASE          # default: module-level BASE (exp001 behaviour)
        self.path = f"{name}/{level}" if name else str(level)
        self.meta = fetch_json(f"{self.path}/.zarray", self.base)
        self.shape = tuple(self.meta["shape"])
        self.chunks = tuple(self.meta["chunks"])
        self.sep = self.meta.get("dimension_separator", ".")
        self.comp = self.meta["compressor"]
        self.dtype = np.dtype(self.meta["dtype"])

    def _chunk(self, iz: int, iy: int, ix: int) -> np.ndarray:
        key = self.sep.join(str(i) for i in (iz, iy, ix))
        raw = fetch(f"{self.base}/{self.path}/{key}", allow_missing=True)
        if raw is None:
            return np.zeros(self.chunks, self.dtype)
        buf = decode_chunk(raw, self.comp)
        return np.frombuffer(buf, self.dtype).reshape(self.chunks)

    def read_yx(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """All z, window [y0:y1, x0:x1]; z must fit in one chunk (true for this dataset)."""
        assert self.chunks[0] >= self.shape[0], "multi-chunk z not supported"
        cy, cx = self.chunks[1], self.chunks[2]
        iys = range(y0 // cy, (y1 - 1) // cy + 1)
        ixs = range(x0 // cx, (x1 - 1) // cx + 1)
        jobs = [(iy, ix) for iy in iys for ix in ixs]
        with ThreadPoolExecutor(8) as pool:
            blocks = list(pool.map(lambda j: self._chunk(0, j[0], j[1]), jobs))
        out = np.zeros((self.shape[0], len(iys) * cy, len(ixs) * cx), self.dtype)
        for (iy, ix), blk in zip(jobs, blocks):
            out[:, (iy - iys[0]) * cy:(iy - iys[0] + 1) * cy,
                (ix - ixs[0]) * cx:(ix - ixs[0] + 1) * cx] = blk[:self.shape[0]]
        oy, ox = y0 - iys[0] * cy, x0 - ixs[0] * cx
        return out[:, oy:oy + (y1 - y0), ox:ox + (x1 - x0)]


# ----------------------------------------------------------------------------- A
def part_a() -> dict:
    zattrs = fetch_json(f"{VOLUME}/.zattrs")
    res = {"volume_zattrs_extra": {k: v for k, v in zattrs.items() if k != "multiscales"},
           "volume_levels": {}, "label_levels": {}, "grid_match": {}}
    for lv in range(6):
        v = ZArray(VOLUME, str(lv))
        res["volume_levels"][str(lv)] = {"shape": v.shape, "chunks": v.chunks, "compressor": bool(v.comp)}
    for lv in (0, 1, 2):
        for key, name in LABELS.items():
            a = ZArray(name, str(lv))
            res["label_levels"][f"{key}/{lv}"] = {"shape": a.shape, "chunks": a.chunks,
                                                  "compressor": (a.comp or {}).get("cname")}
        res["grid_match"][str(lv)] = {
            "volume_yx": res["volume_levels"][str(lv)]["shape"][1:],
            "ink_yx": res["label_levels"][f"ink/{lv}"]["shape"][1:]}
    res["meta_json"] = fetch_json("meta.json")
    return res


# ----------------------------------------------------------------------------- B
def part_b(meta: dict) -> dict:
    import tifffile
    arrs = {}
    for c in "xyz":
        arrs[c] = tifffile.imread(io.BytesIO(fetch(f"{BASE}/{c}.tif"))).astype(np.float64)
    X, Y, Z = arrs["x"], arrs["y"], arrs["z"]
    invalid = (X == -1) & (Y == -1) & (Z == -1)
    P = np.stack([X, Y, Z], -1)
    sx, sy = meta["scale"]
    res = {"tif_shape": X.shape, "valid_fraction": float((~invalid).mean()),
           "scale_meta": [sx, sy], "tracing_volume_meta": meta.get("volume")}
    for name, a, b, s in (("along_cols", P[:, 1:], P[:, :-1], sx), ("along_rows", P[1:], P[:-1], sy)):
        ok = ~(invalid[:, 1:] | invalid[:, :-1]) if name == "along_cols" else ~(invalid[1:] | invalid[:-1])
        d = np.linalg.norm(a - b, axis=-1)[ok]
        p = float(np.median(d) * s)  # canvas-pixel pitch in tracing-volume voxels
        res[name] = {"median_grid_step_voxels": float(np.median(d)), "p10": float(np.percentile(d, 10)),
                     "p90": float(np.percentile(d, 90)), "pixel_pitch_voxels": p,
                     "pixel_um_if_tracing_2.403": p * 2.403, "pixel_um_if_tracing_4.681": p * 4.681,
                     "pixel_um_if_tracing_9.366": p * 9.366}
    return res


# ----------------------------------------------------------------------------- C
def corr_mae(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan"), float(np.abs(a - b).mean())
    return float(np.corrcoef(a, b)[0, 1]), float(np.abs(a - b).mean())


def reduce_xy(S: np.ndarray, mode: str) -> np.ndarray:
    S = S.astype(np.float32)
    if mode == "mean2x2":
        return S.reshape(S.shape[0], S.shape[1] // 2, 2, S.shape[2] // 2, 2).mean(axis=(2, 4))
    oy, ox = (int(c) for c in mode[-2:])  # "stride01" -> offsets (0, 1)
    return S[:, oy::2, ox::2]


def zsel(R: np.ndarray, i: np.ndarray, mode: str, off: int) -> np.ndarray | None:
    src = 2 * i + off
    if mode == "pick":
        idx = [src]
    elif mode == "mean2":
        idx = [src, src + 1]
    else:  # mean3
        idx = [src - 1, src, src + 1]
    if min(int(j.min()) for j in idx) < 0 or max(int(j.max()) for j in idx) >= R.shape[0]:
        return None
    return np.mean([R[j] for j in idx], axis=0)


def best_pyramid_match(src: np.ndarray, tgt: np.ndarray, zr: range) -> dict:
    """Find how `tgt` (next level) was produced from `src` (finer level)."""
    m = 4
    i = np.array(list(zr))
    rows = []
    for xy in ["mean2x2", "stride00", "stride01", "stride10", "stride11"]:
        R = reduce_xy(src, xy)
        for zm in ("pick", "mean2", "mean3"):
            for off in (-1, 0, 1):
                Rz = zsel(R, i, zm, off)
                if Rz is None:
                    continue
                for sy in (-1, 0, 1):
                    for sx in (-1, 0, 1):
                        t = tgt[i][:, m + sy:tgt.shape[1] - m + sy, m + sx:tgt.shape[2] - m + sx]
                        r = Rz[:, m:R.shape[1] - m, m:R.shape[2] - m]
                        c, mae = corr_mae(r, t)
                        rows.append({"xy": xy, "z": zm, "z_off": off, "shift_yx": [sy, sx],
                                     "corr": c, "mae": mae})
    rows = [r for r in rows if not np.isnan(r["corr"])]
    rows.sort(key=lambda r: -r["corr"])
    return {"best": rows[:5], "worst_corr": rows[-1]["corr"] if rows else None, "n_tested": len(rows)}


def part_c(vol: list[ZArray]) -> dict:
    y1, x1 = WIN_L1
    s0 = vol[0].read_yx(2 * y1, 2 * y1 + 256, 2 * x1, 2 * x1 + 256)           # (65,256,256)
    s1 = vol[1].read_yx(y1, y1 + 128, x1, x1 + 128)                           # (33,128,128)
    s2 = vol[2].read_yx(y1 // 2, y1 // 2 + 64, x1 // 2, x1 // 2 + 64)         # (17,64,64)
    res = {"shapes": {"l0": s0.shape, "l1": s1.shape, "l2": s2.shape}, "z_profile_mean": {
        "l0": [round(float(v), 2) for v in s0.mean(axis=(1, 2))],
        "l1": [round(float(v), 2) for v in s1.mean(axis=(1, 2))],
        "l2": [round(float(v), 2) for v in s2.mean(axis=(1, 2))]}}
    res["l0_to_l1"] = best_pyramid_match(s0, s1, range(6, 27))
    res["l1_to_l2"] = best_pyramid_match(s1, s2, range(3, 14))
    return res


# ----------------------------------------------------------------------------- D / E
def part_d_e(vol: list[ZArray], labs: dict[str, list[ZArray]]) -> dict:
    out = {"label_z_fraction": {}, "contrast": {}}
    y1, x1 = WIN_L1
    wins = {1: (y1, y1 + 512, x1, x1 + 512), 2: (y1 // 2, y1 // 2 + 256, x1 // 2, x1 // 2 + 256)}
    for lv, (ya, yb, xa, xb) in wins.items():
        L = {k: (labs[k][lv].read_yx(ya, yb, xa, xb) > 0) for k in LABELS}
        V = vol[lv].read_yx(ya, yb, xa, xb).astype(np.float32)
        out["label_z_fraction"][f"level{lv}"] = {
            k: {"nonzero_z": [int(z) for z in np.where(a.any(axis=(1, 2)))[0]],
                "positive_fraction_by_z_nonzero": {int(z): round(float(a[z].mean()), 4)
                                                   for z in np.where(a.any(axis=(1, 2)))[0]}}
            for k, a in L.items()}
        ink = L["ink"].max(axis=0)
        sup = L["supervision"].max(axis=0)
        pos, neg = ink & sup, (~ink) & sup
        d = []
        for z in range(V.shape[0]):
            a, b = V[z][pos], V[z][neg]
            sd = np.sqrt((a.var() + b.var()) / 2) if a.size and b.size else np.nan
            d.append(round(float((a.mean() - b.mean()) / sd), 3) if sd and sd > 0 else None)
        out["contrast"][f"level{lv}"] = {"n_ink_px": int(pos.sum()), "n_bg_px": int(neg.sum()),
                                         "cohen_d_by_z": d}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-geometry", action="store_true", help="skip part B (needs tifffile)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import yaml
    global BASE, LABELS, VOLUME, WIN_L1
    cfg = yaml.safe_load(args.config.read_text())["data"]
    BASE, VOLUME, LABELS = cfg["base_url"], cfg["volume"], dict(cfg["labels"])
    WIN_L1 = tuple(cfg["window_l1_yx"])

    result: dict = {}
    result["A_metadata"] = part_a()
    print("A done", flush=True)
    if not args.skip_geometry:
        result["B_geometry"] = part_b(result["A_metadata"]["meta_json"])
        print("B done", flush=True)
    vol = [ZArray(VOLUME, str(l)) for l in range(3)]
    labs = {k: [ZArray(n, str(l)) for l in range(3)] for k, n in LABELS.items()}
    result["C_pyramid"] = part_c(vol)
    print("C done", flush=True)
    result["DE_labels_contrast"] = part_d_e(vol, labs)
    result["fetch_stats"] = dict(_fetch_stats)
    (args.out / "diag.json").write_text(json.dumps(result, indent=1, default=lambda o: list(o)))
    print(json.dumps(result, indent=1, default=lambda o: list(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
