#!/usr/bin/env python3
"""Real HN sample -> two ten-bucket selections -> benchmarks -> explanatory plots."""
import sys
sys.dont_write_bytecode = True
import argparse
import platform
import collections
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
import struct
import subprocess
import statistics

ROOT = Path(__file__).resolve().parent
from select_keys import select

def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

def sample():
    data = ROOT / 'data'
    manifest = json.loads((data / 'manifest.json').read_text())
    missing=[name for name in ('comments.bin.gz','lengths.bin.gz') if not (data/name).exists()]
    if missing:
        raise SystemExit('Benchmark sample is local only. Place comments.bin.gz and lengths.bin.gz in data/ (see data/ATTRIBUTION.md), or use --trace-only / --plots-only.')
    blob = gzip.decompress((data / 'comments.bin.gz').read_bytes())
    lengths = gzip.decompress((data / 'lengths.bin.gz').read_bytes())
    assert hashlib.sha256(blob).hexdigest() == manifest['sha256_text']
    assert hashlib.sha256(lengths).hexdigest() == manifest['sha256_lengths']
    rows, off = [], 0
    for n, in struct.iter_unpack('<I', lengths):
        rows.append(blob[off:off+n])
        off += n
    assert off == len(blob) == manifest['text_bytes'] and len(rows) == manifest['rows']
    build = ROOT / 'build'
    build.mkdir(exist_ok=True)
    (build / 'comments.bin').write_bytes(blob)
    (build / 'lengths.bin').write_bytes(lengths)
    return rows, blob

def census(rows, blob):
    words, bytefreq, rowfirst = [collections.Counter() for _ in range(3)]
    for row in rows:
        # Word statistics use visible text. Matching statistics below use ORIGINAL bytes.
        visible = html.unescape(re.sub(r'<[^>]*>', ' ', row.decode('utf-8')))
        tokens = re.findall(r'[A-Za-z]+', visible)
        words.update(tokens)
        bytefreq.update(row)
        if row:
            rowfirst[row[0]] += 1
    out = ROOT / 'results'
    save_csv(out / 'word-frequency.csv', [dict(word=k, count=v) for k,v in words.most_common()])
    (out / 'first-word-frequency.csv').unlink(missing_ok=True)
    save_csv(out / 'byte-frequency.csv', [dict(byte=k, count=bytefreq[k], row_first_count=rowfirst[k]) for k in range(256)])
    keys, selections = select(words, bytefreq, sum(words.values()), len(blob))
    save_csv(out / 'bucket-selection.csv', selections)
    stats = []
    for key in keys:
        n = key.encode()
        hits = sum(n in r for r in rows)
        # Prefix of each row needed to reach first full match, or full row on a miss.
        span = sum((p + len(n)) if (p := r.find(n)) >= 0 else len(r) for r in rows)
        stats.append(dict(needle=key, needle_bytes=len(n), word_count=words[key],
                          anchor_byte=chr(n[0]),
                          anchor_count=bytefreq[n[0]], anchor_pct=100*bytefreq[n[0]]/len(blob),
                          row_first_byte_count=rowfirst[n[0]], row_first_byte_pct=100*rowfirst[n[0]]/len(rows),
                          prefix_hits=sum(r.startswith(n) for r in rows),
                          suffix_hits=sum(r.endswith(n) for r in rows), contains_hits=hits,
                          contains_hit_pct=100*hits/len(rows), logical_search_span_pct=100*span/len(blob)))
    save_csv(out / 'key-statistics.csv', stats)
    return stats

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', choices=['arm64', 'x86_64'], default='arm64' if platform.system() == 'Darwin' else None, help='explicit macOS compiler target')
    p.add_argument('--runs', type=int, default=3, help='independent program invocations per compiler configuration')
    p.add_argument('--trace-only', action='store_true')
    p.add_argument('--plots-only', action='store_true')
    a = p.parse_args()
    if a.runs < 1:
        p.error('--runs must be positive')
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    if a.plots_only:
        import plot
        plot.draw()
        import report
        report.update()
        return
    if a.trace_only:
        subprocess.run(['make', 'all'] + (['ARCH='+a.arch] if a.arch else []), cwd=ROOT, check=True)
        trace=json.loads((ROOT/'data/trace.json').read_text())
        subprocess.run(['./build/demo','trace',trace['text'],trace['needle']],cwd=ROOT,check=True)
        return
    rows, blob = sample()
    stats = census(rows, blob)
    keys = [r['needle'] for r in stats]
    target = a.arch or 'native'
    stamp = ROOT / 'build/target.txt'
    if not stamp.exists() or stamp.read_text().strip() != target:
        for name in ['demo', 'demo-auto', 'test']:
            (ROOT / 'build' / name).unlink(missing_ok=True)
        stamp.write_text(target+'\n')
    subprocess.run(['make', 'all', 'test'] + (['ARCH='+a.arch] if a.arch else []), cwd=ROOT, check=True)
    trace = json.loads((ROOT / 'data/trace.json').read_text())
    actual = rows[trace['sample_row_index']][trace['byte_start']:trace['byte_start']+trace['byte_length']]
    assert actual == trace['text'].encode()
    result = subprocess.run(['./build/demo', 'trace', trace['text'], trace['needle']], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    (out / 'trace.txt').write_text(result.stdout)
    if a.trace_only:
        print(result.stdout)
        return
    compiler = subprocess.run(['clang++'] + (['-arch', a.arch] if a.arch and platform.system() == 'Darwin' else []) + ['--version'], capture_output=True, text=True, check=True).stdout
    (out / 'environment.txt').write_text('\n'.join(compiler.splitlines()[:2])+'\n')
    all_runs = []
    # Interleave configs by invocation; no two timing processes run concurrently.
    for rep in range(1, a.runs+1):
        configs = [('manual', 'demo'), ('auto', 'demo-auto')]
        if rep % 2 == 0:
            configs.reverse()
        for config, binary in configs:
            args = ['./build/'+binary, 'blob', 'build/comments.bin', 'build/lengths.bin', *keys]
            run = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
            (out / f'{config}-{rep}.csv').write_text(run.stdout)
            parsed = list(csv.DictReader(line for line in run.stdout.splitlines() if not line.startswith('#')))
            for row in parsed:
                row.update(config=config, run=rep)
            all_runs.extend(parsed)
            print(config, rep, 'OK', flush=True)
    grouped = collections.defaultdict(list)
    for row in all_runs:
        grouped[(row['config'],row['needle'],row['shape'],row['impl'])].append(row)
    summary=[]
    for (config,needle,shape,impl), rs in grouped.items():
        times=[float(r['ms']) for r in rs]
        assert len({r['hits'] for r in rs}) == 1
        r=rs[0]
        summary.append(dict(config=config,needle=needle,shape=shape,impl=impl,hits=int(r['hits']),
                            median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
                            blocks=int(r['blocks']),active_blocks=int(r['active_blocks']),
                            candidates=int(r['candidates']),tail_starts=int(r['tail_starts']),scalar_starts=int(r['scalar_starts']),
                            scalar_compares=int(r['scalar_compares'])))
    save_csv(out / 'summary.csv', summary)
    # Independently check counts against Python's byte-string operations.
    lookup = {r['needle']:r for r in census(rows,blob)}
    for r in summary:
        assert r['hits'] == lookup[r['needle']][r['shape']+'_hits']
    import plot
    plot.draw()
    import report
    report.update()
    print('All benchmark counts also match independent Python checks; plots and README updated.')

if __name__ == '__main__':
    main()
