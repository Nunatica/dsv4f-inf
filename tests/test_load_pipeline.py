import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def write_fixture(model):
    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    model.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
                      + string('general.architecture') + struct.pack('<I', 8) + string('deepseek4'))


class LoadPipelineLauncherTest(unittest.TestCase):
    def test_resident_pruned_load_uses_dio(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('DSV4F_')}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        model = Path(temporary.name) / 'V4.gguf'
        write_fixture(model)
        env.update(MODEL=str(model), DSV4F_MTP='0', DSV4F_RAM_BUDGET_MIB='0',
                   DSV4F_ADAPTIVE_PRUNING='50')
        for preload in ('0', '1'):
            for pipeline in ('0', '1'):
                with self.subTest(preload=preload, pipeline=pipeline):
                    result = subprocess.run([str(ROOT / 'run-server.sh'), '--dry-run'],
                        env={**env, 'DSV4F_RAM_PRELOAD': preload, 'DSV4F_LOAD_PIPELINE_MIB': pipeline},
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('RAM budget: 0 MiB; SSD payload I/O: O_DIRECT', result.stdout)
                    self.assertIn('--load-mode dio', result.stdout)

    def test_batch_size_validation(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('DSV4F_')}
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / 'V4.gguf'
            write_fixture(model)
            env.update(MODEL=str(model), DSV4F_MTP='0')
            for value in ('0', '1', '1024', '65536'):
                result = subprocess.run([str(ROOT / 'run-server.sh'), '--dry-run'],
                    env={**env, 'DSV4F_LOAD_PIPELINE_MIB': value}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'Allocation/read pipeline: {value} MiB', result.stdout)
            for value in ('-1', '1.5', '65537', '100000', 'bad'):
                result = subprocess.run([str(ROOT / 'run-server.sh'), '--dry-run'],
                    env={**env, 'DSV4F_LOAD_PIPELINE_MIB': value}, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('DSV4F_LOAD_PIPELINE_MIB must be an integer in 0..65536', result.stderr)
