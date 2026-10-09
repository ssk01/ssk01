#!/usr/bin/env python3
"""Measure the working ASIMD candidates; keep verbose diagnostics separate."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument("binary", type=Path)
p.add_argument("--out", type=Path, default=Path("results/arm"))
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
report = {}
for op in ("shuffle", "reorder"):
    report[op] = {str(cap): {"runs": []} for cap in (2, 256)}
    for repeat in range(5):
        for cap in ((2, 256) if repeat % 2 == 0 else (256, 2)):
            proc = subprocess.run([str(a.binary.resolve()), op, str(cap)],
                env=dict(os.environ, ONEDNN_VERBOSE="0", DNNL_VERBOSE="0"),
                capture_output=True, text=True, timeout=30, check=True)
            (a.out / f"{op}-{cap}-{repeat}.log").write_text(proc.stdout + proc.stderr)
            line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT "))
            fields = dict(item.split("=", 1) for item in line[7:].split())
            report[op][str(cap)]["runs"].append({k: float(v) for k, v in fields.items() if k.endswith("_ms")})
    for cap in (2, 256):
        data = report[op][str(cap)]
        data["median"] = {k: statistics.median(r[k] for r in data["runs"]) for k in data["runs"][0]}
        proc = subprocess.run([str(a.binary.resolve()), op, str(cap)],
            env=dict(os.environ, ONEDNN_VERBOSE="2", DNNL_VERBOSE="2"),
            capture_output=True, text=True, timeout=30, check=True)
        (a.out / f"{op}-{cap}-diagnostic.log").write_text(proc.stdout + proc.stderr)
        region = proc.stdout.split("MEASURE_BEGIN\n")[1].split("MEASURE_END\n")[0]
        fields = [line.split(",") for line in region.splitlines()]
        events = [f[3] for f in fields if len(f)>6 and f[:3]==["onednn_verbose","v1","primitive"] and f[5]==op]
        data["hits"] = events.count("create:cache_hit")
        data["misses"] = events.count("create:cache_miss")
        data["executions"] = events.count("exec")
        if data["hits"] + data["misses"] != 160 or data["executions"] != 160:
            raise RuntimeError("incomplete/unrecognized diagnostic events")
(a.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({op: {cap: {k: v for k, v in data.items() if k!="runs"} for cap, data in caps.items()} for op, caps in report.items()}, indent=2))
