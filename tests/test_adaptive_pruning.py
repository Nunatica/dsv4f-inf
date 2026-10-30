import os
import random
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import model_info


class PruningMemoryEstimateTest(unittest.TestCase):
    def test_mixed_formats_and_sparse_history(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'Pruning.gguf'
            tensors = [('blk.0.ffn_exps_gu_type', [6], 26, 0)]
            offset = 32
            planes = {'q3_k': 28160, 'q2_0': 18432, 'ptq1_0': 14336}
            for suffix, typ in [('q3_k', 11), ('q2_0', 42), ('ptq1_0', 143)]:
                for role in ('gate', 'up', 'down'):
                    tensors.append((f'blk.0.ffn_{role}_exps.{suffix}.weight', [256, 256, 2], typ, offset))
                    offset += 2 * planes[suffix]
            with path.open('wb') as f:
                f.write(struct.pack('<6i', 11, 42, 143, 11, 42, 143))
                f.truncate(offset)
            history = {'version': 1, 'model': 'Pruning', 'experts': {
                'blk.0.ffn_exps.q3_k.weight': {'0': 1, '1': 2},
                'blk.0.ffn_exps.q2_0.weight': {'0': 11, '1': 12},
                'blk.0.ffn_exps.ptq1_0.weight': {'0': 101, '1': 102}}}
            table = Path(temp) / 'Pruning.routing.json'
            table.write_text(json.dumps(history))
            info = {'model': str(path), 'shards': [str(path)], 'layers': 1}
            with patch.object(model_info, 'header', return_value=({'deepseek4.expert_used_count': 2}, tensors, 0)):
                self.assertEqual(model_info.resident_pruned_bytes(info, 0, temp), 0)
                self.assertEqual(model_info.resident_pruned_bytes(info, 1, temp), 3 * planes['q3_k'])
                self.assertEqual(model_info.resident_pruned_bytes(info, 25, temp), 6 * planes['q3_k'])
                self.assertEqual(model_info.resident_pruned_bytes(info, 50, temp), 6 * planes['q3_k'] + 3 * planes['q2_0'])
                del history['experts']['blk.0.ffn_exps.ptq1_0.weight']
                table.write_text(json.dumps(history))
                self.assertEqual(model_info.resident_pruned_bytes(info, 25, temp), 6 * planes['ptq1_0'] + 3 * planes['q3_k'])


class DifferentialPruningTest(unittest.TestCase):
    def test_normalized_usage_and_byte_cost(self):
        layers = [
            {'layer': 0, 'counts': [10, 20, 30, 40], 'bytes': [10]*4, 'top_k': 1},
            {'layer': 1, 'counts': [0, 1, 1, 998], 'bytes': [10]*4, 'top_k': 1}]
        self.assertEqual(model_info.select_pruning(layers, 25), [set(), {0, 1}])
        # Multiplying one layer's sample size must not change its ranking.
        layers[0]['counts'] = [c*1000 for c in layers[0]['counts']]
        self.assertEqual(model_info.select_pruning(layers, 25), [set(), {0, 1}])
        # A larger plane can be cheaper per byte despite a higher count.
        layers = [{'layer': 0, 'counts': [1, 2, 97], 'bytes': [10, 100, 100], 'top_k': 1}]
        self.assertEqual(model_info.select_pruning(layers, 25), [{1}])
        # Even at zero historical cost, top-k experts must survive per layer.
        layers = [{'layer': l, 'counts': [0, 0, 0, 100], 'bytes': [10]*4, 'top_k': 3} for l in range(2)]
        self.assertEqual(model_info.select_pruning(layers, 25), [{0}, {0}])
        with self.assertRaises(ValueError):
            model_info.select_pruning(layers, 50)
        # Small cheap planes must not consume all removable slots when a
        # larger plane is needed to reach an otherwise feasible byte target.
        layers = [{'layer': 0, 'counts': [0, 0, 100], 'bytes': [1, 1, 100], 'top_k': 1}]
        self.assertEqual(model_info.select_pruning(layers, 50), [{0, 2}])

    def test_native_python_selection_parity(self):
        binary = ROOT / 'build/bin/dsv4f-adaptive-pruning-test'
        if not binary.exists():
            self.skipTest('Build dsv4f-adaptive-pruning-test for native planner parity')
        rng = random.Random(814)
        cases = []
        for case in range(100):
            layers = []
            for l in range(rng.randint(1, 5)):
                n = rng.randint(4, 20)
                counts = [rng.choice([0, 0, 1, 10, 100, 2**62]) for _ in range(n)]
                counts[-1] = max(1, counts[-1])
                layers.append({'layer': l*2, 'counts': counts,
                    'bytes': [rng.choice([10, 17, 100, 2**30]) for _ in range(n)],
                    'top_k': rng.randint(1, n)})
            rng.shuffle(layers)
            cases.append({'percentage': rng.choice([0, 1, 20, 25, 30, 50]), 'layers': layers})
        expected = []
        for case in cases:
            try:
                expected.append([sorted(ids) for ids in model_info.select_pruning(case['layers'], case['percentage'])])
            except ValueError:
                expected.append(None)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'cases.json'
            source.write_text(json.dumps(cases))
            result = subprocess.run([str(binary), str(source)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), expected)


class AdaptivePruningLauncherTest(unittest.TestCase):
    def test_pruning_range_and_mutual_exclusion(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith('DSV4F_')}
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / 'DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf'
            def string(value):
                value = value.encode()
                return struct.pack('<Q', len(value)) + value
            model.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 1)
                              + string('general.architecture') + struct.pack('<I', 8)
                              + string('deepseek4'))
            env.update(MODEL=str(model), DSV4F_ADAPTIVE_ROUTING='0', DSV4F_MTP='0')
            for value in ('0', '1', '25', '50'):
                with self.subTest(value=value):
                    result = subprocess.run([str(ROOT / 'run-server.sh'), '--dry-run'],
                        env={**env, 'DSV4F_ADAPTIVE_PRUNING': value}, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f'Adaptive expert pruning: {value}%', result.stdout)
            for value in ('-1', '51', '100', '1.5', 'abc', '01', '999999999999999999999'):
                with self.subTest(value=value):
                    result = subprocess.run([str(ROOT / 'run-server.sh'), '--dry-run'],
                        env={**env, 'DSV4F_ADAPTIVE_PRUNING': value}, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('integer in 0..50', result.stderr)
            result = subprocess.run([str(ROOT / 'run-server.sh'), '--dry-run'],
                env={**env, 'DSV4F_ADAPTIVE_ROUTING': '1', 'DSV4F_ADAPTIVE_PRUNING': '25'},
                capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('cannot both be enabled', result.stderr)
