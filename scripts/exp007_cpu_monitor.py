"""exp007: sample whole-machine CPU utilisation from /proc/stat until killed (Linux only).

Appends one JSON line per interval: {"t": seconds, "busy": 0-1, "iowait": 0-1}. No dependencies.
Usage: python exp007_cpu_monitor.py --out cpu.jsonl [--interval 2] [--proc-stat /proc/stat] [--max-samples N]
"""

from __future__ import annotations

import argparse
import json
import time


def read_cpu(path: str) -> tuple[float, float, float]:
    """(total, idle, iowait) jiffies of the aggregate 'cpu' line."""
    with open(path) as f:
        parts = f.readline().split()
    vals = [float(x) for x in parts[1:]]
    total = sum(vals[:8]) if len(vals) >= 8 else sum(vals)          # user nice system idle iowait irq softirq steal
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)
    return total, idle, (vals[4] if len(vals) > 4 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--proc-stat", default="/proc/stat")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = run until killed")
    args = ap.parse_args()
    t0 = time.time()
    prev = read_cpu(args.proc_stat)
    n = 0
    with open(args.out, "a") as out:
        while args.max_samples == 0 or n < args.max_samples:
            time.sleep(args.interval)
            cur = read_cpu(args.proc_stat)
            dt, di, dw = cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2]
            if dt > 0:
                out.write(json.dumps({"t": round(time.time() - t0, 1), "busy": round(1 - di / dt, 4),
                                      "iowait": round(dw / dt, 4)}) + "\n")
                out.flush()
            prev = cur
            n += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
