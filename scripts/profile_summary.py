#!/usr/bin/env python3
"""Summarize a fresh DSV4F_PROFILE server log and its sequential benchmark."""
import argparse
import json
from pathlib import Path
import re
import statistics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--log', type=Path, required=True)
p.add_argument('--benchmark', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
rows = [{k: float(v) for k, v in re.findall(r'(\w+)=([\d.]+)', line)}
        for line in args.log.read_text().splitlines() if line.startswith('[dsv4f-profile]')]
bench = json.loads(args.benchmark.read_text())
# A fresh server and one benchmark only: no other requests before/between runs.
# The last prompt token is evaluated separately on the single-token graph. Native
# decode timing excludes that first sample, just as it excludes the first output token.
start = bench['warmup_tokens']
reports = []
for i, run in enumerate(bench['runs']):
    n = run['timings']['predicted_n']
    interval = rows[start + 1:start + n]
    if len(interval) != n - 1:
        raise ValueError('Log does not contain the expected benchmark samples')
    result = {k: statistics.mean(r[k] for r in interval)
              for k in interval[0] if k not in ('sample', 'moe_calls')}
    result.update(run=i + 1, samples=len(interval),
                  native_ms_per_token=1000 / run['timings']['predicted_per_second'])
    result['outside_moe_ms_approx'] = result['native_ms_per_token'] - result['moe_ms']
    reports.append(result)
    start += n
args.output.write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
