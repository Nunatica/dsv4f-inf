#!/usr/bin/env python3
"""Process-tree RAM budget including anonymous, pinned and charged file memory."""
import json
import os
from pathlib import Path
import sys
from model_info import swap_protected

MIB = 1 << 20


def budget_bytes(env=None):
    env = os.environ if env is None else env
    value = env.get('DSV4F_RAM_BUDGET_MIB', '0')
    if not value.isascii() or not value.isdecimal() or int(value) > (1 << 44)-1:
        raise ValueError('DSV4F_RAM_BUDGET_MIB must be a nonnegative integer in MiB')
    return int(value)*MIB


def cgroup_paths(proc=Path('/proc/self/cgroup'), root=Path('/sys/fs/cgroup')):
    for line in proc.read_text().splitlines():
        if line.startswith('0::'):
            path = root / line[3:].lstrip('/')
            while True:
                yield path
                if path == root:
                    return
                path = path.parent


def memory_state(paths=None):
    paths = list(cgroup_paths() if paths is None else paths)
    limits = []
    for path in paths:
        try:
            value = (path/'memory.max').read_text().strip()
            if value != 'max':
                limits.append((int(value), path))
        except FileNotFoundError:
            continue
    limit, limited_path = min(limits, key=lambda x: x[0]) if limits else (None, None)
    state = {'limit_bytes': limit, 'limit_path': str(limited_path) if limited_path else None}
    if paths:
        state['path'] = str(paths[0])
        for name in ('memory.current', 'memory.peak', 'memory.swap.current'):
            try:
                state[name] = int((paths[0]/name).read_text())
            except FileNotFoundError:
                pass
    return state


def protected(budget, state=None):
    if budget == 0:
        return True
    state = memory_state() if state is None else state
    return state['limit_bytes'] is not None and state['limit_bytes'] <= budget


def effective_budget(budget, state):
    """Pass an inherited tighter ceiling to the native cache allocator too."""
    if not protected(budget, state):
        raise RuntimeError('Requested process RAM budget is not enforced by cgroup memory.max')
    if not budget:
        return 0
    value = min(budget, state['limit_bytes']) // MIB * MIB
    if not value:
        raise RuntimeError('The inherited cgroup RAM limit is too small')
    return value


if __name__ == '__main__':
    try:
        budget = budget_bytes()
        if sys.argv[1:] == ['--check']:
            sys.exit(0 if protected(budget) and swap_protected() else 1)
        print(json.dumps({'budget_bytes': budget, **memory_state()}, indent=2))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
