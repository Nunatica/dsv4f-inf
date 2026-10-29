#!/usr/bin/env python3
"""Start the native server and verify that all routed weights are resident."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
from model_info import inspect, swap_protected


def proc_memory(pid):
    fields = {}
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmSwap:', 'VmLck:')):
            k, v, _ = line.split()
            fields[k.rstrip(':')] = int(v)*1024
    return fields


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    opts = p.parse_args()
    cmd = opts.command[1:] if opts.command[:1] == ['--'] else opts.command
    info = inspect(os.environ['MODEL'])
    if not swap_protected():
        raise RuntimeError('Swap protection is absent; refusing to start')
    mem = {k: int(v.split()[0])*1024 for k, v in (s.split(':', 1) for s in Path('/proc/meminfo').read_text().splitlines())}
    if mem['MemAvailable'] < info['model_bytes'] + 2*(1<<30):
        raise RuntimeError(f"Need at least {(info['model_bytes']/(1<<30)+2):.1f} GiB available RAM for resident loading")
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logfile = opts.log_dir/f'server-{stamp}.log'
    manifest = opts.state_dir/'server.json'
    host = os.environ.get('HOST', '127.0.0.1')
    if host in ('0.0.0.0', '::'):
        host = '127.0.0.1' if host == '0.0.0.0' else '[::1]'
    url = f"http://{host}:{os.environ.get('PORT', '8080')}"
    print(f"[dsv4f] {info['name']}: {info['model_bytes']/(1<<30):.2f} GiB in {len(info['shards'])} shards", flush=True)
    print(f'[dsv4f] swap forbidden; resident loading (no mmap/lazy); log: {logfile}', flush=True)
    # Verbose GGUF tokenizer metadata can contain non-UTF-8 token bytes.
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace', bufsize=1)
    def forward(sig, _):
        if child.poll() is None:
            child.send_signal(sig)
    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    def copy_output():
        with logfile.open('w') as log:
            for line in child.stdout:
                log.write(line)
                log.flush()
                try:
                    print(line, end='', flush=True)
                except BrokenPipeError:
                    pass
    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    verified = False
    started = time.monotonic()
    try:
        while child.poll() is None:
            status = proc_memory(child.pid)
            if status.get('VmSwap', 0):
                raise RuntimeError('Server has swapped pages; stopping')
            if not verified:
                try:
                    with urlopen(url+'/health', timeout=1) as response:
                        health = json.load(response)
                    ready = health.get('status') == 'ok'
                except (URLError, TimeoutError, ValueError):
                    ready = False
                if ready:
                    if status.get('VmRSS', 0) < info['expert_bytes']*0.97:
                        raise RuntimeError(f"Resident RSS {status.get('VmRSS',0)/(1<<30):.2f} GiB is smaller than routed weights {info['expert_bytes']/(1<<30):.2f} GiB")
                    verified = True
                    state = {**info, 'pid': child.pid, 'supervisor_pid': os.getpid(), 'url': url,
                             'log': str(logfile), 'command': cmd, 'memory': status,
                             'engine_options': {k: os.environ[k] for k in (
                                 'DSV4F_CPU_THREADS', 'DSV4F_VRAM_HEADROOM_MIB', 'DSV4F_VRAM_GUARD_MIB', 'DSV4F_EXPERT_CACHE_MIB',
                                 'DSV4F_WHOLE_EXPERT', 'DSV4F_BATCH_TRANSFERS', 'DSV4F_GPU_JOIN',
                                 'DSV4F_PINNED_COMPUTE', 'DSV4F_PINNED_WEIGHTS', 'DSV4F_LOAD_MODE', 'DSV4F_LOAD_PIPELINE_MIB', 'DSV4F_EXPERT_GRAPHS', 'DSV4F_FUSED_HC_MIX',
                                 'DSV4F_KV_RESIDENT', 'DSV4F_MTP', 'MTP_MODEL', 'DSV4F_DRAFT_N_MAX', 'DSV4F_DRAFT_P_MIN',
                                 'DSV4F_GPU_SERVICE', 'DSV4F_GPU_WAIT', 'DSV4F_SKIP_NO_MISS', 'DSV4F_ROUTER_FUSED', 'DSV4F_ROUTER_DEBUG', 'DSV4F_CPU_SPIN_US', 'DSV4F_HC_WARP', 'DSV4F_HC_FULL', 'DSV4F_MXFP4_FUSED',
                                 'DSV4F_CPU_FAST_SYNC', 'DSV4F_CPU_PROFILE',
                                 'DSV4F_CPU_AFFINITY', 'DSV4F_CPU_SCHEDULE', 'DSV4F_CPU_ROW_CHUNK', 'DSV4F_CPU_MXFP4_ROWS',
                                 'DSV4F_CPU_MXFP4_ISA',
                                 'DSV4F_HC_POST_FULL',
                                 'DSV4F_PREFILL_STREAM', 'DSV4F_PREFILL_GROUP', 'DSV4F_PREFILL_CACHE', 'DSV4F_PREFILL_CACHE_MIN_TOKENS', 'DSV4F_PREFILL_PROFILE',
                                 'DSV4F_PHASE_WORKSPACE', 'DSV4F_DECODE_BATCH', 'DSV4F_DECODE_UBATCH',
                                 'DSV4F_CACHE_POLICY', 'DSV4F_CACHE_DECAY_TOKENS', 'DSV4F_CACHE_REPLACE_RATIO', 'DSV4F_CACHE_TRACE',
                                 'DSV4F_PROFILE', 'DSV4F_SCHED_SUMMARY', 'GGML_CUDA_DISABLE_GRAPHS',
                                 'GGML_CUDA_GRAPH_OPT', 'GGML_CUDA_NO_PINNED',
                                 'GGML_HIP_SPARSE_FA', 'GGML_HIP_SPARSE_FA_DEBUG',
                                 'GGML_HIP_PREFILL_FA', 'GGML_HIP_PREFILL_FA_DEBUG',
                                 'GGML_HIP_INDEXER_WMMA', 'GGML_HIP_CSA_TOPK') if k in os.environ},
                             'swap_protected': True, 'load_seconds': time.monotonic()-started}
                    temporary = manifest.with_suffix('.tmp')
                    temporary.write_text(json.dumps(state, indent=2))
                    temporary.replace(manifest)
                    print(f"[dsv4f] READY {url}; RSS={status['VmRSS']/(1<<30):.2f} GiB, swap=0", flush=True)
                elif time.monotonic()-started > float(os.environ.get('LOAD_TIMEOUT', '1800')):
                    raise RuntimeError('Model loading timed out')
            time.sleep(1 if not verified else 5)
        reader.join(timeout=2)
        return child.returncode
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if manifest.exists():
            try:
                if json.loads(manifest.read_text()).get('pid') == child.pid:
                    manifest.unlink()
            except (OSError, ValueError):
                pass


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f'[dsv4f] {error}', file=sys.stderr)
        sys.exit(1)
