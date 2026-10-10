"""Regression coverage for Python 3.12 app-scoped Inferno initialization."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
import numpy as np
from engine import inferno_adapter as adapter
from engine.inferno_runtime import RUNTIME_VERSION, normalize_probabilities


class InfernoRuntime(unittest.TestCase):
    def test_tiny_sigmoid_probabilities_keep_their_relative_evidence(self):
        output = normalize_probabilities([[1e-30, 2e-30], [0., 0.]])
        self.assertTrue(np.allclose(output[0], [1 / 3, 2 / 3]))
        self.assertEqual(output[1].sum(), 0.)

    def test_target_directory_processes_package_startup_hooks(self):
        # Reproduce the important --target installation property without loading
        # TensorFlow: a dependency .pth hook must execute even if bootstrap.py
        # already inserted its directory on sys.path.
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as temporary:
            app = pathlib.Path(temporary) / 'app'
            packages = app / '.deps'
            packages.mkdir(parents=True)
            (packages / 'dependency_hook.pth').write_text(
                "import os; os.environ['ARCFORGE_TEST_SITE_HOOK'] = 'initialized'\n", 'utf-8')
            server_packages = app / '.server-deps'
            server_packages.mkdir()
            (server_packages / 'arcforge_server_dependency.py').write_text('available = True\n', 'utf-8')
            code = ("import pathlib,sys,os; "
                    f"sys.path.insert(0, {str(ROOT)!r}); import bootstrap; "
                    "from engine.inferno_runtime import initialize_runtime; "
                    f"sys.path.insert(0, {str(packages)!r}); "
                    f"initialize_runtime(pathlib.Path({str(app)!r})); "
                    "import arcforge_server_dependency; assert arcforge_server_dependency.available; "
                    "assert os.environ.get('ARCFORGE_TEST_SITE_HOOK') == 'initialized'; "
                    f"assert sys.path[0] == {str(packages)!r}")
            result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_old_or_corrupt_runtime_record_is_not_verified(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as temporary:
            root = pathlib.Path(temporary)
            (root / 'models').mkdir()
            record = root / 'models/inferno-runtime-status.json'
            with patch.object(adapter, 'ROOT', root), patch.object(adapter.importlib.util, 'find_spec', return_value=object()):
                record.write_text('{"success": true}', 'utf-8')
                self.assertFalse(adapter.status()['inferenceVerified'])
                record.write_text('{bad json', 'utf-8')
                self.assertIsNone(adapter.status()['lastTest'])
                record.write_text(json.dumps({'success': True, 'metadata': {
                    'runtimeVersion': RUNTIME_VERSION, 'pythonVersion': list(sys.version_info[:2])}}), 'utf-8')
                self.assertTrue(adapter.status()['inferenceVerified'])

    def test_missing_python312_compatibility_package_is_reported(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as temporary:
            with patch.object(adapter, 'ROOT', pathlib.Path(temporary)), \
                 patch.object(adapter.importlib.util, 'find_spec', side_effect=lambda name: None if name == 'setuptools' else object()):
                state = adapter.status()
                self.assertFalse(state['runtimeAvailable'])
                self.assertEqual(state['missingRuntimePackages'], ['setuptools'])
                with self.assertRaisesRegex(RuntimeError, 'setuptools.*Setup ArcForge.ps1 -Inferno'):
                    adapter.predict('unused.wav', [0.])

    def test_worker_failure_invalidates_old_success_record(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as temporary:
            root = pathlib.Path(temporary)
            audio = root / 'fixture.wav'
            audio.write_bytes(b'test-audio-fingerprint')
            failure = subprocess.CompletedProcess([], 1, stdout='', stderr='RuntimeError: unavailable runtime')
            with patch.object(adapter, 'ROOT', root), \
                 patch.object(adapter, 'status', return_value={'weightsAvailable': True}), \
                 patch.object(adapter.subprocess, 'run', return_value=failure):
                with self.assertRaisesRegex(RuntimeError, 'unavailable runtime'):
                    adapter.predict(audio, [0., .5])
            record = json.loads((root / 'models/inferno-runtime-status.json').read_text('utf-8'))
            self.assertFalse(record['success'])
            self.assertEqual(record['runtimeVersion'], RUNTIME_VERSION)

    def test_nonfinite_unsorted_or_empty_times_fail_before_worker(self):
        with patch.object(adapter, 'status', return_value={'weightsAvailable': True}), \
             patch.object(adapter.subprocess, 'run') as worker:
            for times in [[], [float('nan')], [-.1], [.5, .1], [[.2, .3]]]:
                with self.subTest(times=times), self.assertRaisesRegex(ValueError, 'event times'):
                    adapter.predict('unused.wav', times)
            worker.assert_not_called()

    def test_model_result_has_valid_hand_position_probabilities(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as temporary:
            root = pathlib.Path(temporary)
            audio = root / 'fixture.wav'
            audio.write_bytes(b'test-audio-fingerprint')

            def worker(command, **kwargs):
                request = json.loads(pathlib.Path(command[-1]).read_text('utf-8'))
                np.savez_compressed(request['output'], probabilities=np.full((2, 2, 108), 1 / 108),
                                    metadata=json.dumps({'runtimeVersion': RUNTIME_VERSION}))
                return subprocess.CompletedProcess(command, 0, stdout='', stderr='')

            with patch.object(adapter, 'ROOT', root), \
                 patch.object(adapter, 'status', return_value={'weightsAvailable': True}), \
                 patch.object(adapter.subprocess, 'run', side_effect=worker) as called:
                probabilities, metadata = adapter.predict(audio, [0., .5])
                self.assertEqual(probabilities.shape, (2, 2, 108))
                self.assertEqual(metadata['runtimeVersion'], RUNTIME_VERSION)
                self.assertTrue(np.allclose(probabilities.sum(-1), 1))
                adapter.predict(audio, [0., .5])
                self.assertEqual(called.call_count, 1)


if __name__ == '__main__':
    unittest.main()
