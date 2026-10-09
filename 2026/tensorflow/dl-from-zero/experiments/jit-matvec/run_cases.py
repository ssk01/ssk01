#!/usr/bin/env python3
"""Independent processes, sequential runs, rotated order; keep every sample."""
import argparse
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("--out", type=Path, default=Path("results"))
p.add_argument("--repeats", type=int, default=5)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "VECLIB_MAXIMUM_THREADS": "1"}
binary = str(args.binary.resolve())
subprocess.run([binary, "verify"], check=True, env=env)
shapes = [(4, 8), (4, 16), (8, 16), (16, 16), (8, 32), (16, 64), (8, 128)]
jobs = [(f"exec-{m}-{k}", ["exec", str(m), str(k)]) for m, k in shapes]
jobs += [("mixed-off", ["mixed", "0", "100"]),
         ("mixed-small", ["mixed", "2", "100"]),
         ("mixed-large", ["mixed", "8", "100"]),
         ("fixed", ["fixed", "2", "100"]),
         ("lifecycle", ["lifecycle", "8", "16", "800"])]
samples = {name: [] for name, _ in jobs}
for repeat in range(args.repeats):
    order = jobs[repeat:] + jobs[:repeat]
    if repeat % 2:
        order = list(reversed(order))
    for name, command in order:
        extra = [str(repeat)] if command[0] == "exec" else []
        result = subprocess.run([binary, *command, *extra], check=True,
                                capture_output=True, text=True, env=env)
        (args.out / f"{name}-{repeat}.log").write_text(result.stdout + result.stderr)
        value = json.loads(result.stdout)
        samples[name].append(value)
        print(repeat, name, flush=True)
for name in ("mixed-off", "mixed-small", "mixed-large", "fixed"):
    for r in samples[name]:
        assert r["hits"] + r["misses"] == r["requests"] == 800
        assert r["misses"] == (800 if name in ("mixed-off", "mixed-small") else 0)
assert len({r["checksum"] for name in ("mixed-off", "mixed-small", "mixed-large")
            for r in samples[name]}) == 1

def medians(rows):
    return {k: medians([r[k] for r in rows]) if isinstance(v, dict)
            else statistics.median(r[k] for r in rows) if isinstance(v, (int, float))
            else v for k, v in rows[0].items()}

result = {"environment": {"platform": platform.platform(), "machine": platform.machine(),
          "compiler": subprocess.check_output(["clang++", "--version"], text=True),
          "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
          "VECLIB_MAXIMUM_THREADS": "1", "repeats": args.repeats},
          "medians": {name: medians(rows) for name, rows in samples.items()}, "samples": samples}
(args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["medians"], indent=2))
