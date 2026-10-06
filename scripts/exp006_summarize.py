"""exp006: summarize the benchmark runs (speed, memory, projections) and the functional run (loss trend).

Reads <out>/runs.json, <out>/runs/<id>_benchmark.json, <out>/logs/<id>.log and (functional run)
<out>/runs/<functional id>/. Prints a compact table, writes summary.json (numbers only) and prints its sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

OFFICIAL_STEPS, OFFICIAL_EFFECTIVE_BATCH = 78125, 64


def log_tail_reason(log: str) -> str:
    for pat in ("OutOfMemoryError", "out of memory", "CUDA error", "NCCL", "Traceback", "Error"):
        m = [l for l in log.splitlines() if pat in l]
        if m:
            return m[-1].strip()[:200]
    return "no benchmark_summary.json and no recognizable error (see log)"


def losses(log: str) -> list[float]:
    return [float(x) for x in re.findall(r"loss=([0-9]+\.[0-9]+)", log)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quota-hours", type=float, default=30.0)
    args = ap.parse_args()
    runs = json.loads((args.out / "runs.json").read_text())
    summary = {"benchmarks": {}, "functional": {}}
    rows = []
    for r in runs:
        log = (args.out / "logs" / f"{r['id']}.log")
        text = log.read_text(errors="replace") if log.exists() else ""
        if r["kind"] == "benchmark":
            f = args.out / "runs" / f"{r['id']}_benchmark.json"
            if f.exists():
                s = json.loads(f.read_text())
                eff_batch = s["batch_size_per_process"] * s["world_size"]
                hours_official = OFFICIAL_STEPS * OFFICIAL_EFFECTIVE_BATCH / s["examples_per_second"] / 3600
                entry = {"ok": True, "gpus": r["gpus"], "effective_batch": eff_batch,
                         "steps_per_second": s["steps_per_second"], "examples_per_second": s["examples_per_second"],
                         "data_wait_per_step_s": s["data_wait_seconds_per_step"],
                         "peak_allocated_gb": s["peak_allocated_bytes"] / 1e9, "peak_reserved_gb": s["peak_reserved_bytes"] / 1e9,
                         "hours_for_78125_steps_at_batch64": hours_official,
                         "hours_for_20000_steps_at_batch64": hours_official * 20000 / OFFICIAL_STEPS,
                         "share_of_weekly_quota_official": hours_official / args.quota_hours}
            else:
                entry = {"ok": False, "gpus": r["gpus"], "reason": log_tail_reason(text)}
            summary["benchmarks"][r["id"]] = entry
            if entry["ok"]:
                rows.append(f"| {r['id']} | {entry['effective_batch']} | {entry['steps_per_second']:.2f} | {entry['examples_per_second']:.1f} | "
                            f"{entry['data_wait_per_step_s']:.3f} | {entry['peak_allocated_gb']:.1f} | {entry['hours_for_20000_steps_at_batch64']:.1f} | "
                            f"{entry['hours_for_78125_steps_at_batch64']:.1f} | {100 * entry['share_of_weekly_quota_official']:.0f}% |")
            else:
                rows.append(f"| {r['id']} | FAILED: {entry['reason']} | | | | | | | |")
        else:
            ls = losses(text)
            vm = sorted((args.out / "runs" / r["id"]).glob("*valid*")) + sorted((args.out / "runs" / r["id"]).glob("*val*.json*"))
            ckpts = sorted(p.name for p in (args.out / "runs" / r["id"]).glob("*.pth"))
            summary["functional"] = {"id": r["id"], "n_loss_points": len(ls),
                                     "loss_first": ls[:3], "loss_last": ls[-3:],
                                     "checkpoints": ckpts, "validation_files": [p.name for p in vm],
                                     "log_has_traceback": "Traceback" in text}
    table = ["| run | eff. batch | steps/s | examples/s | data wait s/step | peak alloc GB | h for 20k steps | h for 78,125 steps | % of 30 h quota (official) |",
             "|---|---|---|---|---|---|---|---|---|"] + rows
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(table))
    print("\nfunctional run:", json.dumps(summary["functional"]))
    print("summary.json sha256:", hashlib.sha256((args.out / "summary.json").read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
