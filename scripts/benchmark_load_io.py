#!/usr/bin/env python3
"""Compare buffered/DIO resident loading with memory-pressure and FD telemetry."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from benchmark import generate

ROOT = Path(__file__).resolve().parents[1]


def fields(path, byte_values=False):
    result = {}
    for line in Path(path).read_text().splitlines():
        parts = line.replace(':', '').split()
        if len(parts) >= 2 and parts[1].isdigit():
            result[parts[0]] = int(parts[1]) * (1024 if byte_values else 1)
    return result


def pressure(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        result[parts[0]] = int(dict(p.split('=') for p in parts[1:])['total'])
    return result


def system_sample():
    memory = fields('/proc/meminfo', True)
    counters = fields('/proc/vmstat')
    return {
        'memory': {k: memory[k] for k in ('MemFree', 'MemAvailable', 'Cached', 'Shmem', 'Buffers', 'SReclaimable', 'Dirty', 'Writeback')},
        'reclaim': {k: v for k, v in counters.items() if k.startswith(('pgscan', 'pgsteal', 'allocstall'))},
        'pressure_us': pressure('/proc/pressure/memory'),
    }


def find_server(parent):
    pending = [parent]
    while pending:
        pid = pending.pop()
        try:
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if args and Path(os.fsdecode(args[0])).name == 'dsv4f-server':
                return pid
            pending.extend(int(p) for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            pass
    return None


def server_sample(pid):
    base = Path(f'/proc/{pid}')
    sample = {'memory': {k: v for k, v in fields(base / 'status', True).items() if k in ('VmRSS', 'VmHWM', 'VmSwap')},
              'io': fields(base / 'io'), 'gguf_fds': []}
    for fd in (base / 'fd').iterdir():
        try:
            target = os.readlink(fd)
            if not target.endswith('.gguf'):
                continue
            flag_line = next(line for line in (base / 'fdinfo' / fd.name).read_text().splitlines() if line.startswith('flags:'))
            flags = int(flag_line.split()[1], 8)
            sample['gguf_fds'].append({'path': target, 'direct': bool(flags & os.O_DIRECT), 'flags_octal': oct(flags)})
        except (FileNotFoundError, ProcessLookupError):
            pass
    group = next(line[3:] for line in (base / 'cgroup').read_text().splitlines() if line.startswith('0::'))
    cg = Path('/sys/fs/cgroup') / group.lstrip('/')
    sample['cgroup'] = str(cg)
    sample['cgroup_stat'] = fields(cg / 'memory.stat')
    sample['cgroup_pressure_us'] = pressure(cg / 'memory.pressure')
    sample['cgroup_bytes'] = {name: int((cg / name).read_text()) for name in ('memory.current', 'memory.peak', 'memory.swap.current') if (cg / name).exists()}
    return sample


def delta(before, after):
    return {k: after[k] - before.get(k, after[k]) for k in after}


def summarize(samples):
    result = {'system_reclaim_delta': delta(samples[0]['system']['reclaim'], samples[-1]['system']['reclaim']),
              'system_pressure_delta_us': delta(samples[0]['system']['pressure_us'], samples[-1]['system']['pressure_us']),
              'system_cached_start': samples[0]['system']['memory']['Cached'],
              'system_cached_peak': max(s['system']['memory']['Cached'] for s in samples),
              'system_cached_end': samples[-1]['system']['memory']['Cached'],
              'system_available_min': min(s['system']['memory']['MemAvailable'] for s in samples)}
    process = [s['server'] for s in samples if 'server' in s]
    if process:
        result.update(cgroup_stat_delta=delta(process[0]['cgroup_stat'], process[-1]['cgroup_stat']),
                      cgroup_pressure_delta_us=delta(process[0]['cgroup_pressure_us'], process[-1]['cgroup_pressure_us']),
                      cgroup_file_peak=max(s['cgroup_stat'].get('file', 0) for s in process),
                      cgroup_file_cache_peak=max(max(0, s['cgroup_stat'].get('file', 0) - s['cgroup_stat'].get('shmem', 0)) for s in process),
                      cgroup_file_cache_end=max(0, process[-1]['cgroup_stat'].get('file', 0) - process[-1]['cgroup_stat'].get('shmem', 0)),
                      process_rss_peak=max(s['memory'].get('VmRSS', 0) for s in process),
                      process_swap_peak=max(s['memory'].get('VmSwap', 0) for s in process),
                      direct_fd_sample_count=sum(fd['direct'] for s in process for fd in s['gguf_fds']),
                      buffered_fd_sample_count=sum(not fd['direct'] for s in process for fd in s['gguf_fds']),
                      direct_files=sorted({fd['path'] for s in process for fd in s['gguf_fds'] if fd['direct']}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', choices=('none', 'dio'), default=['none', 'dio', 'dio', 'none'])
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/load-io-comparison.json')
    parser.add_argument('--port', type=int, default=18089)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    manifest = ROOT / 'state/server.json'
    if manifest.exists():
        raise RuntimeError('An existing server manifest is present; leaving it alone')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    results = []
    report = {'date_utc': stamp, 'modes': args.modes, 'cache_flushed': False, 'pipeline_mib': 1024,
              'mtp': False, 'context': 4096, 'batch': 512, 'ubatch': 512, 'runs': results}
    for index, mode in enumerate(args.modes, 1):
        env = dict(os.environ, DSV4F_MTP='0', DSV4F_LOAD_PIPELINE_MIB='1024',
                   DSV4F_LOAD_MODE=mode, HOST='127.0.0.1', PORT=str(args.port),
                   CTX_SIZE='4096', BATCH_SIZE='512', UBATCH_SIZE='512')
        log_path = ROOT / f'logs/load-io-{stamp}-{index}-{mode}.log'
        print(f'Starting run {index}: load-mode={mode}, pipeline=1024 MiB, MTP=0', flush=True)
        samples = []
        state = None
        server_pid = None
        with log_path.open('w') as output:
            child = subprocess.Popen(['./run-server.sh', '--verbosity', '3'],
                                     cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                start = time.monotonic()
                while time.monotonic() - start < args.timeout:
                    if child.poll() is not None:
                        raise RuntimeError(f'Run {index} exited with {child.returncode}; see {log_path}')
                    sample = {'seconds': time.monotonic() - start, 'system': system_sample()}
                    server_pid = server_pid or find_server(child.pid)
                    if server_pid:
                        sample['server'] = server_sample(server_pid)
                    samples.append(sample)
                    if manifest.exists():
                        candidate = json.loads(manifest.read_text())
                        if candidate['pid'] == server_pid and candidate['url'] == f'http://127.0.0.1:{args.port}':
                            state = candidate
                            break
                    time.sleep(0.25)
                if state is None:
                    raise RuntimeError(f'Run {index} timed out')
                summary = summarize(samples)
                print(f'Run {index} ready: {state["load_seconds"]:.3f}s; direct FD samples={summary.get("direct_fd_sample_count", 0)}', flush=True)
                generated = generate(state['url'], 32)
                log_text = log_path.read_text(errors='replace')
                fallback = [line for line in log_text.splitlines() if 'falling back to buffered' in line.lower()]
                result = {'run': index, 'load_mode': mode, 'load_seconds': state['load_seconds'],
                          'memory': state['memory'], 'summary': summary, 'buffered_fallback_messages': fallback,
                          'native_log': state['log'], 'launch_log': str(log_path),
                          'timings': generated['timings'], 'content': generated['content'], 'samples': samples}
                results.append(result)
                args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
                print(json.dumps({k: result[k] for k in ('run', 'load_mode', 'load_seconds', 'memory', 'buffered_fallback_messages')}), flush=True)
            finally:
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if server_pid:
                        try:
                            os.kill(server_pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                if manifest.exists() and server_pid:
                    if json.loads(manifest.read_text())['pid'] == server_pid:
                        manifest.unlink()
        time.sleep(2)
    print(f'Results saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
