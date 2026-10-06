"""exp007: build the 'uncompressed' variant of the exp006 sparse training layout.

Copies the label zarrs byte for byte and rewrites every segment's surface-volume.zarr WITHOUT
compression (same shape, chunks, zarr v2), chunk by chunk, only for chunks that exist.
Also prints the compressor of the original volume. Run in the villa uv env:
  uv run --no-sync python exp007_make_variants.py --src <train_data> --dst <train_data_unc>
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

SEG_ROOT = Path("labels") / "aligned-scrollprizeorg-21slices"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    args = ap.parse_args()
    import zarr

    if args.dst.exists() and any(args.dst.iterdir()):
        raise FileExistsError(f"{args.dst} is not empty")
    report = {}
    for seg_dir in sorted((args.src / SEG_ROOT).iterdir()):
        if not seg_dir.is_dir():
            continue
        out_dir = args.dst / SEG_ROOT / seg_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in seg_dir.iterdir():                       # labels: byte-exact copy
            if item.name != "surface-volume.zarr":
                shutil.copytree(item, out_dir / item.name)
        src = zarr.open(str(seg_dir / "surface-volume.zarr"), mode="r")
        dst = zarr.create_array(store=str(out_dir / "surface-volume.zarr"), shape=src.shape, chunks=src.chunks,
                                dtype=src.dtype, fill_value=0, zarr_format=2, compressors=None, overwrite=True)
        zarray = json.loads((seg_dir / "surface-volume.zarr" / ".zarray").read_text())
        cy, cx = src.chunks[1], src.chunks[2]
        n = 0
        for f in sorted((seg_dir / "surface-volume.zarr").iterdir()):
            m = re.fullmatch(r"0\.(\d+)\.(\d+)", f.name)
            if not m:
                continue
            iy, ix = int(m.group(1)), int(m.group(2))
            y0, x0 = iy * cy, ix * cx
            y1, x1 = min(y0 + cy, src.shape[1]), min(x0 + cx, src.shape[2])
            dst[:, y0:y1, x0:x1] = np.asarray(src[:, y0:y1, x0:x1])
            n += 1
        new = json.loads((out_dir / "surface-volume.zarr" / ".zarray").read_text())
        report[seg_dir.name] = {"chunks_copied": n, "src_compressor": zarray.get("compressor"),
                                "dst_compressor": new.get("compressor")}
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
