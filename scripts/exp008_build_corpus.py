"""exp008: build one group of the ink_9um 'aligned' training corpus (sparse, 21 slices of 9.6 um).

Loops over the segments of a group (config `groups`), reusing exp006_prepare_train_data.prepare_segment
(labels byte-exact, volume pooled with the official recipe). A failure in one segment is recorded and the
others continue; corpus.json is rewritten after every segment. Needs only numpy, pyyaml, zarr, numcodecs:
  python exp008_build_corpus.py --config configs/exp008_corpus.yaml --group A --out /kaggle/working/corpus
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_exp001 as dz  # noqa: E402
from exp006_prepare_train_data import prepare_segment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--group", required=True, choices=["A", "B"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    scrolls = set(cfg["groups"][args.group])
    segs = [s for s in cfg["segments"] if s["key"].split("-")[0] in scrolls]
    args.out.mkdir(parents=True, exist_ok=True)
    rec = {"experiment": cfg["experiment"], "group": args.group, "segments": [], "failed": []}
    t_all = time.time()
    for i, seg in enumerate(segs, 1):
        t0 = time.time()
        print(f"[{i}/{len(segs)}] {seg['key']} ...", flush=True)
        try:
            r = prepare_segment(seg, cfg["data"], args.out)
            r["seconds"] = round(time.time() - t0, 1)
            rec["segments"].append(r)
            print(f"    done in {r['seconds']} s; volume chunks fetched {r['volume_chunks_fetched']} "
                  f"(nonzero {r['volume_chunks_nonzero']}); fetched so far {dz._fetch_stats['bytes'] / 1e9:.2f} GB", flush=True)
        except Exception:  # noqa: BLE001 - keep going; the failure is recorded
            rec["failed"].append({"key": seg["key"], "error": traceback.format_exc()[-1200:]})
            print(f"    FAILED {seg['key']}:\n{rec['failed'][-1]['error']}", flush=True)
        rec["fetch_stats"] = dict(dz._fetch_stats)
        rec["elapsed_s"] = round(time.time() - t_all, 1)
        (args.out / "corpus.json").write_text(json.dumps(rec, indent=1))
    print(f"group {args.group}: {len(rec['segments'])} ok, {len(rec['failed'])} failed, {rec['elapsed_s']} s, "
          f"{dz._fetch_stats['bytes'] / 1e9:.2f} GB fetched", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
