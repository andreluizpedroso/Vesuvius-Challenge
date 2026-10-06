"""exp005 step 3: metrics of every checkpoint on the three conditions (paris4, valregions, e841).

Reuses exp002_eval (metrics) and exp004_eval (tile counts / pooled summary). Layout expected:
  <data>/<condition>/...           from the original exp002/003/004 prepare scripts
  <preds>/<ckpt_id>/<condition>[/<segment>]/pred_off0[_reverse].tif
Prints a compact table and the run-to-run spread; writes metrics.json and report.md (numbers only).
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
import exp004_eval as e4  # noqa: E402


def pred_name(direction: str) -> str:
    return "pred_off0.tif" if direction == "forward" else "pred_off0_reverse.tif"


def load_gt(d: Path):
    gt = np.load(d / "gt_ink.npy")
    mask = (np.load(d / "gt_sup.npy") == 1) & (np.load(d / "gt_val.npy") == 1)
    return gt, mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    import tifffile

    cfg = yaml.safe_load(args.config.read_text())
    ev, root = cfg["evaluation"], args.repo_root
    reg = root / ev["contamination_registry"]
    key = cfg["checkpoint_repo"]["registry_key"]
    val_cfg = yaml.safe_load((root / cfg["conditions"]["valregions"]["segments_from"]).read_text())
    segs = val_cfg["data"]["segments"]

    out = {"experiment": cfg["experiment"], "conditions": {}, "runs": {}}
    cond = cfg["conditions"]
    out["conditions"]["paris4"] = {"contamination": e2.contamination(reg, key, cond["paris4"]["scroll"], cond["paris4"]["segment"])}
    out["conditions"]["e841"] = {"contamination": e2.contamination(reg, key, cond["e841"]["scroll"], cond["e841"]["segment"])}
    out["conditions"]["valregions"] = {"contamination": {s["key"]: e2.contamination(reg, key, s["scroll"], s["segment"], cond["valregions"]["region"]) for s in segs}}

    cols = {}                                    # column name -> {ckpt_id: (auprc, f_beta)}
    for ck in cfg["checkpoint_repo"]["checkpoints"]:
        cid, base = ck["id"], args.preds / ck["id"]
        runs = {}
        for name in ("paris4", "e841"):
            gt, mask = load_gt(args.data / name)
            for d in cond[name]["directions"]:
                f = base / name / pred_name(d)
                pred = tifffile.imread(str(f))
                assert pred.shape == gt.shape and pred.dtype == np.uint8, (f, pred.shape)
                r = e2.evaluate_run(pred, gt, mask, ev)
                r["pred_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
                runs[f"{name}_{d}"] = r
                cols.setdefault(f"{name}_{d}", {})[cid] = (r["auprc"], r["fixed_0.5"]["f_beta"])
        pos = neg = 0; counts = []; per_seg = {}
        for s in segs:
            gt, mask = load_gt(args.data / "valregions" / s["key"])
            f = base / "valregions" / s["key"] / pred_name("forward")
            pred = tifffile.imread(str(f))
            assert pred.shape == gt.shape and pred.dtype == np.uint8, (f, pred.shape)
            r = e2.evaluate_run(pred, gt, mask, ev)
            r["pred_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
            per_seg[s["key"]] = r
            p, n = e2.hists(pred, gt, mask); pos, neg = pos + p, neg + n
            counts.append(e4.tile_counts(pred, gt, mask, ev["threshold_u8"], ev["bootstrap"]["tile"]))
        pooled = e4.summarize(pos, neg, np.concatenate(counts, axis=1), ev)
        runs["valregions_pooled_forward"] = pooled
        runs["valregions_per_segment_forward"] = per_seg
        cols.setdefault("valregions_pooled_forward", {})[cid] = (pooled["auprc"], pooled["fixed_0.5"]["f_beta"])
        out["runs"][cid] = runs

    # ---- reproduction check, spread, seed gap
    ref, tol = ev["reference_s42_75000"], ev["reproduction_tolerance_auprc"]
    refmap = {"paris4_forward": ref["paris4"], "valregions_pooled_forward": ref["valregions_pooled"],
              "e841_forward": ref["e841_forward"], "e841_reverse": ref["e841_reverse"]}
    repro = {}
    for col, r in refmap.items():
        got = cols[col]["s42_75000"]
        repro[col] = {"auprc": got[0], "ref_auprc": r["auprc"], "f_beta": got[1], "ref_f_beta": r["f_beta"],
                      "reproduced": bool(abs(got[0] - r["auprc"]) <= tol)}
    spread = {col: {"auprc_min": min(v[0] for v in d.values()), "auprc_max": max(v[0] for v in d.values()),
                    "f_beta_min": min(v[1] for v in d.values()), "f_beta_max": max(v[1] for v in d.values())}
              for col, d in cols.items()}
    seed_gap = {col: {step: abs(d[f"s42_{step}"][0] - d[f"s43_{step}"][0]) for step in (20000, 50000, 75000)}
                for col, d in cols.items()}
    out.update({"reproduction": repro, "spread": spread, "seed_gap_auprc": seed_gap})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(out, indent=1))

    order = ["paris4_forward", "valregions_pooled_forward", "e841_forward", "e841_reverse"]
    L = [f"# {cfg['experiment']} - metrics", "",
         f"- paris4: {out['conditions']['paris4']['contamination']}",
         f"- valregions: same-scroll held-out region (see per-segment labels in metrics.json)",
         f"- e841: {out['conditions']['e841']['contamination']}", "",
         "Cells are AUPRC / F0.5@0.5 (forward unless noted; e841 reverse = post-hoc orientation).", "",
         "| checkpoint | Paris 4 (in-train) | val regions (pooled) | 841 forward | 841 reverse |", "|---|---|---|---|---|"]
    for ck in cfg["checkpoint_repo"]["checkpoints"]:
        c = ck["id"]
        L.append(f"| {c} | " + " | ".join(f"{cols[o][c][0]:.3f} / {cols[o][c][1]:.3f}" for o in order) + " |")
    L += ["", "Spread across the 6 checkpoints (AUPRC min-max): " +
          "; ".join(f"{o}: {spread[o]['auprc_min']:.3f}-{spread[o]['auprc_max']:.3f}" for o in order),
          "Seed gap |s42-s43| in AUPRC (steps 20000/50000/75000): " +
          "; ".join(f"{o}: " + "/".join(f"{seed_gap[o][s]:.3f}" for s in (20000, 50000, 75000)) for o in order),
          "", "Reproduction of s42_75000 vs exp002-004 (tolerance |dAUPRC| <= %.3f): " % tol +
          "; ".join(f"{o}: {repro[o]['auprc']:.3f} vs {repro[o]['ref_auprc']:.3f} -> {'OK' if repro[o]['reproduced'] else 'NOT REPRODUCED'}" for o in order)]
    (args.out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("metrics.json sha256:", hashlib.sha256((args.out / "metrics.json").read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
