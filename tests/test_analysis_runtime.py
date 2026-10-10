"""Actual compute-policy and denser-window contracts; no accuracy percentages."""
import contextlib, json, pathlib, sys, tempfile, threading, unittest
from unittest.mock import patch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
import numpy as np
from engine.analysis_runtime import normalize_compute, compute_plan, initial_batch_size, analysis_thread_scope
from engine.recognition.yamnet import log_mel_patches


class FakeCuda:
    class OutOfMemoryError(RuntimeError): pass
    @staticmethod
    def mem_get_info(): return 3 * 1024 ** 3, 4 * 1024 ** 3
    @staticmethod
    def synchronize(): pass
    @staticmethod
    def reset_peak_memory_stats(): pass
    @staticmethod
    def max_memory_allocated(): return 1234
    @staticmethod
    def max_memory_reserved(): return 2048
    @staticmethod
    def empty_cache(): pass
    @staticmethod
    def is_available(): return False


class FakeTorch:
    cuda = FakeCuda()
    def __init__(self): self.threads = 4
    def get_num_threads(self): return self.threads
    def set_num_threads(self, count): self.threads = count
    @staticmethod
    def inference_mode(): return contextlib.nullcontext()
    @staticmethod
    def from_numpy(value): return FakeTensor(value)


class FakeTensor:
    def __init__(self, value): self.value = value
    def to(self, _): return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


class FakeModel:
    def __init__(self, maximum=10000): self.maximum = maximum; self.attempts = []
    def __call__(self, value):
        self.attempts.append(len(value.value))
        if len(value.value) > self.maximum: raise FakeCuda.OutOfMemoryError('controlled memory failure')
        values = np.full((len(value.value), 521), .1, np.float32)
        return FakeTensor(values), None


class AnalysisComputePolicy(unittest.TestCase):
    def test_default_preserves_balanced_and_old_device_alias(self):
        self.assertEqual(normalize_compute(), {'quality': 'balanced', 'device': 'auto', 'cpuThreads': 0})
        self.assertEqual(normalize_compute({'recognitionDevice': 'cpu'})['device'], 'cpu')

    def test_rejects_nonfunctional_unknown_modes_and_threads(self):
        for value in [None, 'fake', 3]:
            with self.subTest(quality=value), self.assertRaises(ValueError): normalize_compute({'quality': value})
        for value in ['gpu', 'random']:
            with self.subTest(device=value), self.assertRaises(ValueError): normalize_compute({'device': value})
        for value in [True, 1.5, float('nan'), float('inf'), -1, 'all', (sys.maxsize,)]:
            with self.subTest(threads=value), self.assertRaises(ValueError): normalize_compute({'cpuThreads': value})

    @patch('engine.analysis_runtime.os.cpu_count', return_value=12)
    def test_deep_all_cpu_threads_and_four_temporal_views(self, _):
        torch = FakeTorch()
        deep = compute_plan({'quality': 'deep'}, torch)
        balanced = compute_plan(None, torch)
        self.assertEqual((deep['threads'], deep['fftWorkers'], deep['hopFrames']), (12, 12, 12))
        self.assertEqual((balanced['threads'], balanced['fftWorkers'], balanced['hopFrames']), (4, 1, 48))

    def test_cuda_batches_use_headroom_and_cpu_batches_are_bounded(self):
        torch = FakeTorch(); plan = compute_plan({'quality': 'deep', 'cpuThreads': 2}, torch)
        count, memory = initial_batch_size(torch, 'cuda', plan)
        self.assertLessEqual(count * memory['estimatedBytesPerPatch'], memory['activationBudgetBytes'])
        self.assertLess(memory['activationBudgetBytes'], memory['freeBytesAtStart'])
        self.assertLessEqual(initial_batch_size(torch, 'cpu', plan)[0], 128)
        self.assertEqual(initial_batch_size(torch, 'cpu', compute_plan(None, torch)), (32, None))

    @patch('engine.analysis_runtime.os.cpu_count', return_value=12)
    def test_thread_scope_restores_policy_even_when_inference_fails(self, _):
        torch = FakeTorch(); plan = compute_plan({'quality': 'deep'}, torch)
        with self.assertRaises(RuntimeError):
            with analysis_thread_scope(torch, plan):
                self.assertEqual(torch.threads, 12)
                raise RuntimeError('failed inference')
        self.assertEqual(torch.threads, 4)


class TemporalViews(unittest.TestCase):
    def test_default_features_equal_every_fourth_deep_window(self):
        samples = np.random.default_rng(712).normal(0, .1, 12 * 16000).astype(np.float32)
        balanced = log_mel_patches(samples)
        deep = log_mel_patches(samples, hop_frames=12)
        self.assertGreater(len(deep), len(balanced) * 3)
        self.assertEqual(deep.shape[1:], (96, 64))
        # Tail padding differs by the finer hop; compare shared non-padded views.
        np.testing.assert_allclose(deep[::4][:len(balanced)-1], balanced[:-1], atol=1e-7)

    def test_parallel_fft_keeps_model_feature_convention(self):
        samples = np.random.default_rng(713).normal(0, .1, 32000).astype(np.float32)
        np.testing.assert_allclose(log_mel_patches(samples, hop_frames=12, workers=2),
                                   log_mel_patches(samples, hop_frames=12), atol=2e-6)

    def test_invalid_window_hops_cannot_silently_change_model_shape(self):
        for value in [0, -1, 97, True, .12]:
            with self.subTest(hop=value), self.assertRaises(ValueError):
                log_mel_patches(np.zeros(16000, np.float32), hop_frames=value)


class RecognitionComputeIntegration(unittest.TestCase):
    def test_cuda_oom_retries_same_windows_and_restores_threads(self):
        from engine import recognition
        torch = FakeTorch(); model = FakeModel(maximum=4)
        with patch.object(recognition, 'status', return_value={'ready': True}), \
             patch.object(recognition, '_load_model', return_value=(model, torch, 'cuda')):
            result = recognition.classify_samples(np.zeros(3 * 16000, np.float32),
                settings={'quality': 'deep', 'device': 'cuda'})
        self.assertGreater(result['oomRetries'], 0)
        self.assertEqual(result['batchSize'], 4)
        self.assertEqual(len(result['scores']), result['patchCount'])
        self.assertEqual(len(result['times']), result['patchCount'])
        np.testing.assert_allclose(np.diff(result['times']), .12)
        self.assertEqual(result['gpuMemory']['peakAllocatedBytes'], 1234)
        self.assertEqual(torch.threads, 4)
        self.assertEqual(result['requestedCompute']['device'], 'cuda')

    def test_cpu_device_settings_and_existing_positional_device_both_work(self):
        from engine import recognition
        torch = FakeTorch(); model = FakeModel()
        with patch.object(recognition, 'status', return_value={'ready': True}), \
             patch.object(recognition, '_load_model', return_value=(model, torch, 'cpu')) as load:
            old = recognition.classify_samples(np.zeros(3 * 16000, np.float32), 'cpu')
            deep = recognition.classify_samples(np.zeros(3 * 16000, np.float32),
                settings={'quality': 'deep', 'device': 'cpu', 'cpuThreads': 1})
        self.assertEqual(load.call_args_list[0].kwargs['device'], 'cpu')
        self.assertEqual(old['hopSeconds'], .48)
        self.assertEqual(old['batchSize'], 32)
        self.assertEqual(deep['cpuThreads'], 1)
        self.assertGreater(deep['patchCount'], old['patchCount'])

    def test_explicit_unavailable_cuda_cannot_return_a_cache_hit(self):
        from engine import recognition
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temporary:
            audio = pathlib.Path(temporary) / 'fixture.ogg'; audio.write_bytes(b'fixture')
            with patch.object(recognition, 'status', return_value={'ready': True}), \
                 patch.object(recognition, '_torch', return_value=FakeTorch()):
                with self.assertRaisesRegex(ValueError, 'no available CUDA'):
                    recognition.recognize({'audioPath': str(audio)}, settings={'device': 'cuda'})

    def test_caches_isolate_compute_requests_and_report_origin(self):
        from engine import recognition
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temporary:
            root = pathlib.Path(temporary)
            audio = root / 'fixture.ogg'; audio.write_bytes(b'fixture')
            manifest = root / 'manifest.json'; manifest.write_text(json.dumps({'sha256': 'fixture'}), 'utf-8')
            result = {'scores': np.zeros((2, 521), np.float32), 'times': np.array([.4875, .9675]),
                      'device': 'cpu', 'quality': 'balanced', 'inferenceSeconds': 1., 'patchCount': 2, 'hopSeconds': .48}
            with patch.object(recognition, 'ROOT', root), patch.object(recognition, 'MANIFEST', manifest), \
                 patch.object(recognition, 'status', return_value={'ready': True}), \
                 patch.object(recognition, '_torch', return_value=FakeTorch()), \
                 patch('engine.features.decode_audio', return_value=np.zeros(16000, np.float32)), \
                 patch.object(recognition, 'classify_samples', return_value=result) as classify, \
                 patch.object(recognition, 'evidence_from_scores', side_effect=lambda *_: {'frames': [], 'events': []}):
                analysis = {'audioPath': str(audio), 'duration': 1.}
                cold = recognition.recognize(analysis, settings={'quality': 'balanced', 'device': 'cpu'})
                warm = recognition.recognize(analysis, settings={'quality': 'balanced', 'device': 'cpu'})
                deep = recognition.recognize(analysis, settings={'quality': 'deep', 'device': 'cpu'})
                auto = recognition.recognize(analysis, settings={'quality': 'deep', 'device': 'auto'})
            self.assertEqual(classify.call_count, 3)
            self.assertFalse(cold['performance']['cacheHit'])
            self.assertTrue(warm['performance']['cacheHit'])
            self.assertFalse(warm['performance']['inferenceExecuted'])
            self.assertEqual(warm['performance']['cacheOriginDevice'], 'cpu')
            self.assertFalse(deep['performance']['cacheHit'])
            self.assertFalse(auto['performance']['cacheHit'])

    def test_deep_cuda_overlaps_one_source_worker_and_reuses_its_evidence(self):
        from engine import recognition
        cpu_started = threading.Event(); gpu_started = threading.Event()
        baseline = {'frames': ['measured'], 'events': []}
        inference = {'scores': np.zeros((2, 521)), 'times': np.array([0., .12])}
        def sources(*_):
            cpu_started.set()
            self.assertTrue(gpu_started.wait(2.), 'CPU features were serialized behind GPU inference')
            return baseline, .5
        def classifier(*_, **__):
            gpu_started.set()
            self.assertTrue(cpu_started.wait(2.), 'The independent CPU feature worker did not start')
            return inference
        torch = FakeTorch()
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(recognition, '_torch', return_value=torch), \
             patch.object(recognition, '_source_baseline', side_effect=sources), \
             patch.object(recognition, 'classify_samples', side_effect=classifier), \
             patch.object(recognition, 'evidence_from_scores', return_value={'frames': [], 'events': []}) as fused:
            _, _, stages = recognition._uncached_recognition({}, np.zeros(16000),
                {'quality': 'deep', 'device': 'auto'}, None, None)
        self.assertIs(fused.call_args.args[3], baseline)
        self.assertTrue(stages['parallelSourceAnalysis'])
        self.assertEqual(stages['sourceFeatureWorkers'], 1)
        self.assertFalse(any(t.name.startswith('ArcAudio-source-features') for t in threading.enumerate()))

    def test_cpu_or_balanced_does_not_start_a_parallel_source_worker(self):
        from engine import recognition
        inference = {'scores': np.zeros((2, 521)), 'times': np.array([0., .48])}
        for settings in [{'quality': 'balanced', 'device': 'auto'}, {'quality': 'deep', 'device': 'cpu'}]:
            with self.subTest(settings=settings), \
                 patch.object(recognition, '_torch', side_effect=AssertionError('Unnecessary GPU probe')), \
                 patch.object(recognition, '_source_baseline', side_effect=AssertionError('Unexpected parallel source task')), \
                 patch.object(recognition, 'classify_samples', return_value=inference), \
                 patch.object(recognition, 'evidence_from_scores', return_value={'frames': [], 'events': []}) as fused:
                _, _, stages = recognition._uncached_recognition({}, np.zeros(16000), settings, None, None)
            self.assertFalse(stages['parallelSourceAnalysis'])
            self.assertEqual(len(fused.call_args.args), 3)

    def test_parallel_source_failure_propagates_and_joins_worker(self):
        from engine import recognition
        inference = {'scores': np.zeros((2, 521)), 'times': np.array([0., .12])}
        torch = FakeTorch()
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(recognition, '_torch', return_value=torch), \
             patch.object(recognition, '_source_baseline', side_effect=ValueError('HPSS decoding failed')), \
             patch.object(recognition, 'classify_samples', return_value=inference):
            with self.assertRaisesRegex(ValueError, 'HPSS decoding failed'):
                recognition._uncached_recognition({}, np.zeros(16000), {'quality': 'deep', 'device': 'cuda'}, None, None)
        self.assertFalse(any(t.name.startswith('ArcAudio-source-features') for t in threading.enumerate()))

    def test_inference_cancellation_joins_started_cpu_worker(self):
        from engine import recognition
        from training_runtime import TrainingCancelled
        cpu_started = threading.Event(); inference_failed = threading.Event(); cpu_finished = threading.Event()
        def sources(*_):
            cpu_started.set()
            self.assertTrue(inference_failed.wait(2.))
            cpu_finished.set()
            return {'frames': [], 'events': []}, .1
        def classifier(*_, **__):
            self.assertTrue(cpu_started.wait(2.))
            inference_failed.set()
            raise TrainingCancelled('Cancelled')
        torch = FakeTorch()
        with patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(recognition, '_torch', return_value=torch), \
             patch.object(recognition, '_source_baseline', side_effect=sources), \
             patch.object(recognition, 'classify_samples', side_effect=classifier):
            with self.assertRaises(TrainingCancelled):
                recognition._uncached_recognition({}, np.zeros(16000), {'quality': 'deep', 'device': 'cuda'}, None, None)
        self.assertTrue(cpu_finished.is_set())
        self.assertFalse(any(t.name.startswith('ArcAudio-source-features') for t in threading.enumerate()))


if __name__ == '__main__': unittest.main()
