#!/usr/bin/env python3
"""Run isolated performance cases; collect cache events in separate diagnostic runs."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("binary", type=Path)
parser.add_argument("--out", type=Path, default=Path("results"))
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--rounds", type=int, default=25)
parser.add_argument("--k", type=int, default=256)
parser.add_argument("--n", type=int, default=256)
parser.add_argument("--dtype", choices=["f32", "bf16", "bf16-f32"], default="f32")
args = parser.parse_args()
if args.repeats < 1 or args.rounds < 1:
    parser.error("repeats and rounds must be positive")
args.out.mkdir(parents=True, exist_ok=True)
cases = [("fixed", "fixed", 2), ("mixed-small", "mixed", 2),
         ("mixed-large", "mixed", 256), ("mixed-off", "mixed", 0)]


def run(case, verbose, suffix):
    name, mode, capacity = case
    # Same request count even though fixed has just one distinct shape.
    rounds = args.rounds * (8 if mode == "fixed" else 1)
    env = dict(os.environ, ONEDNN_VERBOSE=str(verbose), DNNL_VERBOSE=str(verbose),
               OMP_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    command = [str(args.binary.resolve()), mode, str(capacity), str(rounds), str(args.k), str(args.n), args.dtype]
    proc = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=120)
    (args.out / f"{name}-{suffix}.log").write_text(proc.stdout)
    proc.check_returncode()
    result_lines = [x for x in proc.stdout.splitlines() if x.startswith("RESULT ")]
    if len(result_lines) != 1:
        raise RuntimeError("missing/ambiguous result")
    result = json.loads(result_lines[0][7:])
    result["config"] = next(x for x in proc.stdout.splitlines() if x.startswith("CONFIG "))
    result["implementations"] = sorted({x.split(" impl=", 1)[1].split(" prepare_us=", 1)[0]
                                        for x in proc.stdout.splitlines() if x.startswith("WARM ")})
    if verbose:
        measured = proc.stdout.split("MEASURE_BEGIN\n", 1)[1].split("MEASURE_END\n", 1)[0]
        hit = miss = executions = 0
        create_ms = exec_ms = 0.0
        for line in measured.splitlines():
            parts = line.split(",")
            # Verified against oneDNN 3.9.1 primitive verbose v1 format.
            if len(parts) < 7 or parts[:3] != ["onednn_verbose", "v1", "primitive"]:
                continue
            event, kind = parts[3], parts[5]
            if kind != "matmul":
                continue
            if event == "create:cache_hit":
                hit += 1; create_ms += float(parts[-1])
            elif event == "create:cache_miss":
                miss += 1; create_ms += float(parts[-1])
            elif event == "exec":
                executions += 1; exec_ms += float(parts[-1])
        if hit + miss != result["requests"] or executions != result["requests"]:
            raise RuntimeError("unrecognized/incomplete verbose events; inspect diagnostic log")
        result["diagnostic"] = dict(hits=hit, misses=miss, executions=executions,
                                    logged_create_ms=create_ms, logged_exec_ms=exec_ms)
    return result


report = {"platform": platform.platform(), "k": args.k, "n": args.n, "dtype": args.dtype,
          "repeats": args.repeats, "rounds": args.rounds,
          "performance": {c[0]: [] for c in cases}, "diagnostics": {}}
for rep in range(args.repeats):
    for case in (cases if rep % 2 == 0 else list(reversed(cases))):
        report["performance"][case[0]].append(run(case, 0, f"perf-{rep}"))
for case in cases:
    report["diagnostics"][case[0]] = run(case, 2, "diagnostic")
mixed = [report["performance"][name] for name in ("mixed-small", "mixed-large", "mixed-off")]
if len({r["checksum"] for group in mixed for r in group}) != 1:
    raise RuntimeError("mixed workloads have different checksums")
if len({tuple(r["implementations"]) for group in mixed for r in group}) != 1:
    raise RuntimeError("mixed cases selected different implementations")
report["median"] = {}
for name, values in report["performance"].items():
    report["median"][name] = {key: statistics.median(v[key] for v in values)
                              for key in values[0] if isinstance(values[0][key], (int, float))}
(args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"median": report["median"],
                  "cache_events": {k: v["diagnostic"] for k, v in report["diagnostics"].items()}}, indent=2))
