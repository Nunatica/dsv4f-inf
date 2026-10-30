"""Exercise response-boundary routing persistence using the tiny V4 fixture."""
import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

server, model = (Path(arg).resolve() for arg in sys.argv[1:])
environment = {key: value for key, value in os.environ.items()
               if not key.startswith(('DSV4F_', 'GGML_', 'MTP_'))}
environment.update(DSV4F_MTP='0', DSV4F_CPU_THREADS='4', DSV4F_RAM_BUDGET_MIB='0',
                   DSV4F_GPU_SERVICE='0', DSV4F_WHOLE_EXPERT='1',
                   DSV4F_ENGRAM_DIR=str(model.with_suffix('.engram')))


@contextlib.contextmanager
def running(directory, enabled, pruning=0):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    with (directory / 'server.log').open('w+') as log:
        process = subprocess.Popen([
            str(server), '--model', str(model), '--ctx-size', '128', '--parallel', '1',
            '--n-gpu-layers', '0', '--fit', 'off', '--load-mode', 'none',
            '--threads', '4', '--threads-batch', '4', '--batch-size', '64', '--ubatch-size', '32',
            '--flash-attn', 'on', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
            '--host', '127.0.0.1', '--port', str(port),
        ], cwd=directory, env={**environment, 'DSV4F_ADAPTIVE_ROUTING': str(enabled),
                              'DSV4F_ADAPTIVE_PRUNING': str(pruning)},
            stdout=log, stderr=log)
        try:
            for _ in range(200):
                if process.poll() is not None:
                    raise RuntimeError('server exited during startup')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=.2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(.05)
            else:
                raise TimeoutError('server startup')
            yield url
        except Exception:
            log.flush()
            print((directory / 'server.log').read_text(), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def complete(url, stream=False):
    # The synthetic tokenizer contains numeric fixture tokens only.
    body = dict(prompt=[2, 3, 4], n_predict=4, ignore_eos=True, temperature=0,
                cache_prompt=False, stream=stream)
    request = urllib.request.Request(url + '/completion', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


with tempfile.TemporaryDirectory(prefix='dsv4f-routing-server-') as temporary:
    directory = Path(temporary)
    path = directory / 'models' / (model.stem + '.routing.json')
    def total():
        data = json.loads(path.read_text())
        assert data['model'] == model.stem
        return sum(sum(row.values()) for row in data['experts'].values())
    with running(directory, 1) as url:
        assert not path.exists(), 'warmup must not save a routing table'
        first = json.loads(complete(url))
        assert first['tokens_predicted'] == 4
        previous = total()
        assert previous > 0, 'completed response must persist decode routes'
        stream = complete(url, True)
        assert b'"stop":true' in stream.replace(b' ', b'')
        assert total() == 2 * previous, 'streaming completion must save every route exactly once'
    with running(directory, 1) as url:
        complete(url)
        assert total() == 3 * previous, 'restart must retain cumulative counts and exclude warmup'
    contents = path.read_bytes()
    with running(directory, 0) as url:
        disabled = json.loads(complete(url))
        assert disabled['content'] == first['content'], 'cache history must preserve generated output'
        assert path.read_bytes() == contents, 'OFF must leave routing history untouched'
    with running(directory, 0, 25) as url:
        pruned = json.loads(complete(url))
        assert pruned['tokens_predicted'] == 4
        assert path.read_bytes() == contents, 'pruning must leave history untouched'
    # These errors must stop the binary before any model loading or serving.
    for routing, pruning, message in [(1, '25', 'cannot both be enabled'),
                                      (0, '51', 'integer in 0..50'),
                                      (0, '-1', 'integer in 0..50')]:
        result = subprocess.run([str(server), '--model', str(model)], cwd=directory,
            env={**environment, 'DSV4F_ADAPTIVE_ROUTING': str(routing),
                 'DSV4F_ADAPTIVE_PRUNING': pruning}, capture_output=True, timeout=15)
        assert result.returncode != 0 and message.encode() in result.stderr, result.stderr
print('Routing persistence, streaming, restart, OFF, pruning and startup rejection PASS')
