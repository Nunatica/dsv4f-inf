#!/usr/bin/env python3
"""Summarize the trailing native-decode interval of a rocprofv3 CSV trace."""
import argparse
import collections
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--trace-dir', type=Path, required=True)
p.add_argument('--benchmark', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
b = json.loads(args.benchmark.read_text())['runs'][-1]['timings']
kernel_path = args.trace_dir / 'dsv4f_kernel_trace.csv'
with kernel_path.open() as f:
    end = max(int(r['End_Timestamp']) for r in csv.DictReader(f))
start = end - int(b['predicted_ms'] * 1e6)
n = b['predicted_n'] - 1

def summarize(path, name_key):
    sums = collections.defaultdict(lambda: [0, 0])
    with path.open() as f:
        for r in csv.DictReader(f):
            t0, t1 = int(r['Start_Timestamp']), int(r['End_Timestamp'])
            if t0 < start or t1 > end:
                continue
            s = sums[r[name_key]]
            s[0] += 1
            s[1] += t1 - t0
    return [{'name': name, 'calls_per_token': value[0]/n,
             'sum_ms_per_token': value[1]/1e6/n}
            for name, value in sorted(sums.items(), key=lambda item: -item[1][1])]

kernels = summarize(kernel_path, 'Kernel_Name')
api = summarize(args.trace_dir / 'dsv4f_hip_api_trace.csv', 'Function')
report = {'window_ns': [start, end], 'tokens': n,
          'profiled_native_ms_per_token': b['predicted_ms']/n,
          'kernels_per_token': sum(r['calls_per_token'] for r in kernels),
          'kernel_sum_ms_per_token': sum(r['sum_ms_per_token'] for r in kernels),
          'kernels': kernels, 'api': api,
          'limits': 'Trailing window approximates decode boundaries using the last GPU event. '
                    'Profiler adds overhead. API waits, kernels, copies and CPU work overlap; '
                    'their durations must not be added to obtain end-to-end time.'}
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in ('kernels', 'api')}, indent=2))
