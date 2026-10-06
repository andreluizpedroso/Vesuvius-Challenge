"""exp002 step 3: metrics of ink_9um predictions vs the 841/w00 ground truth.

Exact threshold-free/threshold metrics from uint8 histograms (no sklearn): F0.5, F1 (Dice),
precision/recall at a fixed threshold, average precision (AUPRC), the F0.5-oracle threshold
(optimistic), a tile bootstrap CI, an all-positive baseline and the contamination label.
Prints a compact table; writes metrics.json and report.md (numbers only, no images).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def fbeta(tp, fp, fn, beta):
    b2 = beta * beta
    den = (1 + b2) * tp + b2 * fn + fp
    return np.where(den > 0, (1 + b2) * tp / np.maximum(den, 1e-12), 0.0)


def hists(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    pos = np.bincount(pred[mask & (gt == 1)], minlength=256).astype(np.float64)
    neg = np.bincount(pred[mask & (gt == 0)], minlength=256).astype(np.float64)
    return pos, neg


def curves(pos, neg):
    """Counts when predicting positive for v >= t, for t = 0..255."""
    tp = np.cumsum(pos[::-1])[::-1]
    fp = np.cumsum(neg[::-1])[::-1]
    fn = pos.sum() - tp
    return tp, fp, fn


def average_precision(pos, neg) -> float:
    tp, fp, fn = curves(pos, neg)
    P = pos.sum()
    if P == 0:
        return float("nan")
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)
    rec = tp / P
    # thresholds descending in t => ascending recall; step-wise AP like sklearn
    t = np.arange(255, -1, -1)
    r, p = rec[t], prec[t]
    r_prev = np.concatenate([[0.0], r[:-1]])
    return float(np.sum((r - r_prev) * p))


def metrics_at(pos, neg, t: int, beta: float) -> dict:
    tp, fp, fn = curves(pos, neg)
    TP, FP, FN = tp[t], fp[t], fn[t]
    return {"threshold_u8": int(t), "f_beta": float(fbeta(TP, FP, FN, beta)),
            "dice_f1": float(fbeta(TP, FP, FN, 1.0)),
            "precision": float(TP / (TP + FP)) if TP + FP > 0 else 0.0,
            "recall": float(TP / (TP + FN)) if TP + FN > 0 else 0.0}


def tile_bootstrap(pred, gt, mask, t, beta, tile, n, seed):
    H, W = pred.shape
    ty, tx = H // tile, W // tile
    pb = (pred >= t) & mask
    tp = np.zeros((ty, tx)); fp = np.zeros((ty, tx)); fn = np.zeros((ty, tx))
    for i in range(ty):
        for j in range(tx):
            sl = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
            g, p, m = gt[sl] == 1, pb[sl], mask[sl]
            tp[i, j] = (p & g & m).sum(); fp[i, j] = (p & ~g & m).sum(); fn[i, j] = (~p & g & m).sum()
    tp, fp, fn = tp.ravel(), fp.ravel(), fn.ravel()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, tp.size, size=(n, tp.size))
    vals = fbeta(tp[idx].sum(1), fp[idx].sum(1), fn[idx].sum(1), beta)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def contamination(registry_path: Path, key: str, scroll: str, segment: str, region: str | None = None) -> str:
    reg = yaml.safe_load(registry_path.read_text())[key]
    if scroll not in reg["train_scrolls"]:
        return "cross-scroll (scroll not in training data)"
    if region == "held_out_validation" and segment in reg.get("held_out_regions", {}).get(scroll, []):
        return ("SAME-SCROLL held-out region (validation mask; adjacent to training areas; "
                "optimistic vs a new scroll)")
    segs = reg["train_segments"]
    listed = any(segment in v for grp in segs.values() for s, v in grp.items() if s == scroll)
    return ("IN-TRAIN segment (not a generalization measure)" if listed
            else "same scroll, segment not listed in training (in-scroll, not cross-scroll)")


def evaluate_run(pred_u8, gt, mask, ev) -> dict:
    pos, neg = hists(pred_u8, gt, mask)
    t = ev["threshold_u8"]
    res = {"fixed_0.5": metrics_at(pos, neg, t, ev["beta"])}
    # raw p >= 0.5 and (p-lo)/(hi-lo) >= 0.5 are the same decision when lo/hi are symmetric about 0.5
    lo, hi = ev["rescale_check"]["lo"], ev["rescale_check"]["hi"]
    v = np.arange(256) / 255.0
    same = bool(np.array_equal(v >= 0.5, np.clip((v - lo) / (hi - lo), 0, 1) >= 0.5))
    res["rescaled_0.5_identical_to_raw_0.5"] = same
    tp, fp, fn = curves(pos, neg)
    fb = fbeta(tp, fp, fn, ev["beta"])
    best = int(np.argmax(fb))
    res["oracle_threshold (OPTIMISTIC)"] = metrics_at(pos, neg, best, ev["beta"])
    res["auprc"] = average_precision(pos, neg)
    res["n_pos"], res["n_neg"] = int(pos.sum()), int(neg.sum())
    res["pred_mean_u8"] = float(pred_u8[mask].mean())
    res["f_beta_ci95_tile_bootstrap"] = tile_bootstrap(
        pred_u8, gt, mask, t, ev["beta"], ev["bootstrap"]["tile"], ev["bootstrap"]["n"], ev["bootstrap"]["seed"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dir with gt_*.npy from exp002_prepare")
    ap.add_argument("--preds", type=Path, required=True, help="dir with pred_off<k>[_reverse].tif")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    import tifffile

    cfg = yaml.safe_load(args.config.read_text())
    ev = cfg["evaluation"]
    gt = np.load(args.data / "gt_ink.npy")
    mask = (np.load(args.data / "gt_sup.npy") == 1) & (np.load(args.data / "gt_val.npy") == 1)
    args.out.mkdir(parents=True, exist_ok=True)

    prev = float(gt[mask].mean())
    tp, fp, fn = float(mask.sum() * prev), float(mask.sum() * (1 - prev)), 0.0
    out = {"experiment": cfg["experiment"], "eval_pixels": int(mask.sum()),
           "eval_mask_fraction": float(mask.mean()), "ink_prevalence": prev,
           "baseline_all_positive": {"f_beta": float(fbeta(tp, fp, fn, ev["beta"])),
                                     "precision": prev, "recall": 1.0, "auprc": prev},
           "contamination": contamination(args.repo_root / ev["contamination_registry"],
                                          cfg["checkpoint"]["registry_key"], cfg["data"]["scroll"],
                                          cfg["data"]["segment"]),
           "primary": ev["primary"], "runs": {}}
    for k in cfg["inference"]["depth_offsets"]:
        for direction in ("forward", "reverse"):
            f = args.preds / (f"pred_off{k}.tif" if direction == "forward" else f"pred_off{k}_reverse.tif")
            pred = tifffile.imread(str(f))
            assert pred.shape == gt.shape and pred.dtype == np.uint8, (f, pred.shape, pred.dtype)
            r = evaluate_run(pred, gt, mask, ev)
            r["pred_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
            out["runs"][f"offset{k:+d}_{direction}"] = r

    (args.out / "metrics.json").write_text(json.dumps(out, indent=1))
    pk = f"offset{ev['primary']['offset']:+d}_{ev['primary']['direction']}"
    lines = [f"# {cfg['experiment']} - metrics", "",
             f"Contamination: {out['contamination']}", "",
             f"Eval pixels {out['eval_pixels']} ({out['eval_mask_fraction']:.3f} of crop); "
             f"ink prevalence {prev:.3f}; all-positive baseline F0.5 = {out['baseline_all_positive']['f_beta']:.3f}, "
             f"AUPRC baseline = {prev:.3f}", "",
             "| run | F0.5@0.5 | CI95 (tiles) | P | R | Dice | AUPRC | oracle thr | oracle F0.5 (optimistic) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in out["runs"].items():
        m, o = r["fixed_0.5"], r["oracle_threshold (OPTIMISTIC)"]
        star = " **(primary)**" if name == pk else ""
        lines.append(f"| {name}{star} | {m['f_beta']:.3f} | [{r['f_beta_ci95_tile_bootstrap'][0]:.3f}, "
                     f"{r['f_beta_ci95_tile_bootstrap'][1]:.3f}] | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['dice_f1']:.3f} | {r['auprc']:.3f} | {o['threshold_u8']} | {o['f_beta']:.3f} |")
    same = all(r["rescaled_0.5_identical_to_raw_0.5"] for r in out["runs"].values())
    lines += ["", f"Raw p>=0.5 and rescaled (p-0.25)/0.5>=0.5 are the same decision: {same} "
              "(thresholded metrics and AUPRC identical by construction).",
              "Primary run was fixed in the config before inference; other rows are sensitivity analysis."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("metrics.json sha256:", hashlib.sha256((args.out / "metrics.json").read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
