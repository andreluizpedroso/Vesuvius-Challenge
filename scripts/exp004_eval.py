"""exp004 step 3: per-segment and pooled metrics on the validation regions (forward runs).

Reuses the exact metric code of exp002_eval.py. Pooled = sum of the uint8 histograms of all
segments (pixel-weighted) with a tile bootstrap over the tiles of all segments together.
Prints a compact table; writes metrics.json and report.md (numbers only, no images).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp002_eval as e2  # noqa: E402


def tile_counts(pred, gt, mask, t, tile):
    """(tp, fp, fn) per tile at threshold t (v >= t), tiles of tile x tile pixels."""
    H, W = pred.shape
    ty, tx = H // tile, W // tile
    pb = pred >= t
    out = np.zeros((3, ty * tx))
    for i in range(ty):
        for j in range(tx):
            sl = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
            g, p, m = gt[sl] == 1, pb[sl], mask[sl]
            out[:, i * tx + j] = ((p & g & m).sum(), (p & ~g & m).sum(), (~p & g & m).sum())
    return out


def boot_ci(counts, beta, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, counts.shape[1], size=(n, counts.shape[1]))
    tp, fp, fn = (counts[i][idx].sum(1) for i in range(3))
    v = e2.fbeta(tp, fp, fn, beta)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summarize(pos, neg, counts, ev):
    t = ev["threshold_u8"]
    tp, fp, fn = e2.curves(pos, neg)
    best = int(np.argmax(e2.fbeta(tp, fp, fn, ev["beta"])))
    return {"fixed_0.5": e2.metrics_at(pos, neg, t, ev["beta"]), "auprc": e2.average_precision(pos, neg),
            "oracle_threshold (OPTIMISTIC)": e2.metrics_at(pos, neg, best, ev["beta"]),
            "f_beta_ci95_tile_bootstrap": boot_ci(counts, ev["beta"], ev["bootstrap"]["n"], ev["bootstrap"]["seed"]),
            "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dir from exp004_prepare (one subdir per segment)")
    ap.add_argument("--preds", type=Path, required=True, help="dir with <key>/pred_off<k>.tif")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    import tifffile

    cfg = yaml.safe_load(args.config.read_text())
    ev, tile = cfg["evaluation"], cfg["evaluation"]["bootstrap"]["tile"]
    offsets, segs = cfg["inference"]["depth_offsets"], cfg["data"]["segments"]
    args.out.mkdir(parents=True, exist_ok=True)

    per, acc = {}, {k: {"pos": 0, "neg": 0, "counts": []} for k in offsets}
    seg_info, tot_pos, tot_n = {}, 0.0, 0.0
    for seg in segs:
        key = seg["key"]
        gt = np.load(args.data / key / "gt_ink.npy")
        mask = (np.load(args.data / key / "gt_sup.npy") == 1) & (np.load(args.data / key / "gt_val.npy") == 1)
        prev = float(gt[mask].mean())
        seg_info[key] = {"eval_pixels": int(mask.sum()), "ink_prevalence": prev,
                         "all_positive_f_beta": float(e2.fbeta(mask.sum() * prev, mask.sum() * (1 - prev), 0.0, ev["beta"])),
                         "contamination": e2.contamination(args.repo_root / ev["contamination_registry"],
                                                           cfg["checkpoint"]["registry_key"], seg["scroll"],
                                                           seg["segment"], ev.get("region"))}
        tot_pos += mask.sum() * prev; tot_n += mask.sum()
        for k in offsets:
            f = args.preds / key / f"pred_off{k}.tif"
            pred = tifffile.imread(str(f))
            assert pred.shape == gt.shape and pred.dtype == np.uint8, (f, pred.shape, pred.dtype)
            r = e2.evaluate_run(pred, gt, mask, ev)
            r["pred_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
            per[f"{key}|offset{k:+d}_forward"] = r
            pos, neg = e2.hists(pred, gt, mask)
            acc[k]["pos"] = acc[k]["pos"] + pos; acc[k]["neg"] = acc[k]["neg"] + neg
            acc[k]["counts"].append(tile_counts(pred, gt, mask, ev["threshold_u8"], tile))

    pooled = {f"offset{k:+d}_forward": summarize(acc[k]["pos"], acc[k]["neg"],
                                                  np.concatenate(acc[k]["counts"], axis=1), ev) for k in offsets}
    pprev = tot_pos / tot_n
    out = {"experiment": cfg["experiment"], "segments": seg_info,
           "pooled_eval_pixels": int(tot_n), "pooled_ink_prevalence": float(pprev),
           "pooled_all_positive_f_beta": float(e2.fbeta(tot_n * pprev, tot_n * (1 - pprev), 0.0, ev["beta"])),
           "primary": ev["primary"], "per_segment": per, "pooled": pooled}
    (args.out / "metrics.json").write_text(json.dumps(out, indent=1))

    pk = f"offset{ev['primary']['offset']:+d}_forward"
    L = [f"# {cfg['experiment']} - metrics (forward only)", "",
         "Region: validation masks of training segments (same-scroll held-out; optimistic vs a new scroll).", ""]
    for key, i in seg_info.items():
        L.append(f"- {key}: {i['eval_pixels']} px, ink prevalence {i['ink_prevalence']:.3f}, "
                 f"all-positive F0.5 {i['all_positive_f_beta']:.3f} | {i['contamination']}")
    L += [f"- POOLED: {out['pooled_eval_pixels']} px, ink prevalence {pprev:.3f}, all-positive F0.5 "
          f"{out['pooled_all_positive_f_beta']:.3f}", "",
          "| segment | run | F0.5@0.5 | CI95 (tiles) | P | R | Dice | AUPRC | oracle thr | oracle F0.5 (optimistic) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    def row(seg, name, r):
        m, o, c = r["fixed_0.5"], r["oracle_threshold (OPTIMISTIC)"], r["f_beta_ci95_tile_bootstrap"]
        star = " **(primary)**" if name == pk else ""
        return (f"| {seg} | {name}{star} | {m['f_beta']:.3f} | [{c[0]:.3f}, {c[1]:.3f}] | {m['precision']:.3f} | "
                f"{m['recall']:.3f} | {m['dice_f1']:.3f} | {r['auprc']:.3f} | {o['threshold_u8']} | {o['f_beta']:.3f} |")
    for name in pooled:
        L.append(row("POOLED", name, pooled[name]))
    for kk, r in per.items():
        seg, name = kk.split("|")
        L.append(row(seg, name, r))
    L += ["", "Pooled = pixel-weighted sum of the 3 segments (histograms) with a tile bootstrap over all tiles.",
          "Primary run was fixed in the config before inference; other offsets are sensitivity analysis."]
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("metrics.json sha256:", hashlib.sha256((args.out / "metrics.json").read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
