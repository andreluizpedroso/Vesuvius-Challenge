"""exp007: summarize the end-to-end variant benchmarks and the loader-only profiles.

Reads <out>/runs.json, <out>/runs/<id>_benchmark.json, <out>/logs/cpu_<id>.jsonl,
<out>/loader_profile_default.json and <out>/loader_profile_thr1.json. Prints compact tables and writes
summary.json (numbers only) plus its sha256. CPU busy for a run = mean over the middle of the CPU samples
(first 30 % = start-up / patch discovery and last 10 % dropped), an approximation of the timed phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def mid_busy(path: Path) -> float | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(rows) < 6:
        return None
    mid = rows[int(0.3 * len(rows)): int(0.9 * len(rows))]
    return sum(r["busy"] for r in mid) / len(mid)


def log_reason(path: Path) -> str:
    text = path.read_text(errors="replace") if path.exists() else ""
    for pat in ("OutOfMemoryError", "out of memory", "Error", "Traceback"):
        m = [l for l in text.splitlines() if pat in l]
        if m:
            return m[-1].strip()[:160]
    return "no benchmark json and no recognizable error"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    runs = json.loads((args.out / "runs.json").read_text())
    summary: dict = {"runs": {}, "loader_profile": {}}
    base_eps = None
    rows = []
    for r in runs:
        f = args.out / "runs" / f"{r['id']}_benchmark.json"
        if f.exists():
            s = json.loads(f.read_text())
            eps = s["examples_per_second"]
            base_eps = eps if r["id"] == "base_w4" else base_eps
            e = {"ok": True, "gpus": r["gpus"], "workers": r["workers"], "volume": r["volume"], "env": r["env_name"],
                 "examples_per_s": eps, "steps_per_s": s["steps_per_second"],
                 "data_wait_per_step_s": s["data_wait_seconds_per_step"], "peak_alloc_gb": s["peak_allocated_bytes"] / 1e9,
                 "cpu_busy_mid": mid_busy(args.out / "logs" / f"cpu_{r['id']}.jsonl"),
                 "hours_20k_steps_batch64": 20000 * 64 / eps / 3600}
        else:
            e = {"ok": False, "reason": log_reason(args.out / "logs" / f"{r['id']}.log")}
        summary["runs"][r["id"]] = e
    for r in runs:
        e = summary["runs"][r["id"]]
        if e["ok"]:
            e["speedup_vs_base_w4"] = e["examples_per_s"] / base_eps if base_eps else None
            busy = "n/a" if e["cpu_busy_mid"] is None else f"{100 * e['cpu_busy_mid']:.0f}%"
            sp = "n/a" if e["speedup_vs_base_w4"] is None else f"{e['speedup_vs_base_w4']:.2f}x"
            rows.append(f"| {r['id']} | {e['volume']} | {e['env']} | {e['workers']} | {e['gpus']} | {e['examples_per_s']:.1f} | {sp} | "
                        f"{e['data_wait_per_step_s']:.3f} | {busy} | {e['hours_20k_steps_batch64']:.1f} |")
        else:
            rows.append(f"| {r['id']} | FAILED: {e['reason']} | | | | | | | | |")
    print("| run | volume | threads env | workers/proc | GPUs | examples/s | vs base_w4 | data wait s/step | CPU busy (mid-run) | h for 20k steps @ batch 64 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    for name in ("default", "thr1"):
        f = args.out / f"loader_profile_{name}.json"
        if not f.exists():
            summary["loader_profile"][name] = {"missing": True}
            continue
        d = json.loads(f.read_text())
        summary["loader_profile"][name] = d
        print(f"\nloader-only [{name}] cpus={d.get('cpus')} patches={d.get('n_training_patches')} single-process ex/s={d.get('single_process_examples_per_s')}")
        print("  worker sweep (aug on):", {w: (v["examples_per_s"], v["cpu_busy"]) for w, v in d.get("sweep_aug_on", {}).items()},
              "| aug-off w4:", d.get("sweep_aug_off_w4"))
        for key in ("profile_aug_on", "profile_aug_off"):
            if key in d:
                print(f"  {key}: top own-time:", [(t["func"], t["own_pct"]) for t in d[key]["top"][:6]])
        if d.get("errors"):
            print("  errors:", list(d["errors"].keys()))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nsummary.json sha256:", hashlib.sha256((args.out / "summary.json").read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
