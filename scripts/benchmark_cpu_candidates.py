#!/usr/bin/env python3
"""Sequential CPU-only expert probes; never run beside model measurements."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

CONFIGS = {
    'baseline': ('none', 'dynamic', 32, 1),
    'physical': ('physical', 'dynamic', 32, 1),
    'static32': ('none', 'static', 32, 1),
    'static8': ('none', 'static', 8, 1),
    'chunk64': ('none', 'dynamic', 64, 1),
    'chunk128': ('none', 'dynamic', 128, 1),
    'physical-static8': ('physical', 'static', 8, 1),
    'rows2': ('none', 'dynamic', 32, 2),
    'rows4': ('none', 'dynamic', 32, 4),
    'physical-rows2': ('physical', 'dynamic', 32, 2),
    'physical-rows4': ('physical', 'dynamic', 32, 4),
    'physical-static-rows2': ('physical', 'static', 8, 2),
    'physical-static-rows4': ('physical', 'static', 8, 4),
    'avx512-rows2': ('none', 'dynamic', 32, 2, 'avx512'),
    'avx512-rows4': ('none', 'dynamic', 32, 4, 'avx512'),
    'avx512-physical': ('physical', 'dynamic', 32, 4, 'avx512'),
    'avx512-chunk128': ('none', 'dynamic', 128, 4, 'avx512'),
    'avx512-static': ('none', 'static', 8, 4, 'avx512'),
}
NAMES = ['DSV4F_CPU_AFFINITY', 'DSV4F_CPU_SCHEDULE', 'DSV4F_CPU_ROW_CHUNK', 'DSV4F_CPU_MXFP4_ROWS', 'DSV4F_CPU_MXFP4_ISA']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--experts', type=int, default=2)
    p.add_argument('--names', default=','.join(CONFIGS))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.rounds < 1 or not 1 <= args.experts <= 6:
        p.error('rounds >= 1 and 1 <= experts <= 6 required')
    selected = args.names.split(',')
    if any(name not in CONFIGS for name in selected):
        p.error('unknown configuration')
    root = Path(__file__).resolve().parents[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.with_suffix('.log').open('w') as log:
        for round_no in range(args.rounds):
            order = selected if round_no % 2 == 0 else list(reversed(selected))
            for name in order:
                env = dict(os.environ)
                env.update(DSV4F_CPU_THREADS='12', DSV4F_CPU_SPIN_US='300', DSV4F_CPU_FAST_SYNC='1')
                for key in ['DSV4F_CPU_PROFILE', 'DSV4F_PROBE_FALLBACK', 'DSV4F_PROFILE']:
                    env.pop(key, None)
                values = CONFIGS[name] if len(CONFIGS[name]) == 5 else (*CONFIGS[name], 'avx2')
                settings = dict(zip(NAMES, map(str, values)))
                env.update(settings)
                run = subprocess.run([str(root/'build/bin/dsv4f-cpu-probe'), str(args.experts)],
                                     env=env, text=True, capture_output=True, check=True)
                log.write(f'round={round_no+1} name={name} settings={settings}\n'+run.stderr+run.stdout)
                log.flush()
                measurements = []
                for line in run.stdout.splitlines():
                    pairs = dict(re.findall(r'(\w+)=([^ ]+)', line))
                    if 'mean_ms' not in pairs:
                        continue
                    row = {k: (v if k in ['path', 'type'] else float(v)) for k, v in pairs.items()}
                    measurements.append(row)
                if len(measurements) != 2:
                    raise RuntimeError('Missing MXFP4 or Q4_K probe measurements')
                results.append({'round': round_no+1, 'name': name, 'settings': settings, 'measurements': measurements})
                summary = {}
                for candidate in selected:
                    summary[candidate] = {}
                    for kind in ['mxfp4', 'q4_K']:
                        values = [m['mean_ms'] for r in results if r['name'] == candidate for m in r['measurements'] if m['type'] == kind]
                        if values:
                            summary[candidate][kind] = {'median_mean_ms': statistics.median(values), 'invocation_means_ms': values}
                args.output.write_text(json.dumps({'threads': 12, 'experts_per_call': args.experts,
                    'working_set_note': '816 MiB MXFP4 / 864 MiB Q4_K; effective_weight_GBps is nominal bytes/time, not measured DRAM bandwidth.',
                    'results': results, 'summary': summary}, indent=2))
                description = ', '.join('{} {:.4f} ms'.format(m['type'], m['mean_ms']) for m in measurements)
                print(f'round {round_no+1} {name}: '+description, flush=True)

if __name__ == '__main__':
    main()
