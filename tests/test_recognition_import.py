"""Real recognition agreement, attention labeling and incremental import checks."""
import json, pathlib, sys, tempfile, unittest, zipfile
from unittest.mock import patch
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import bootstrap
import numpy as np
from engine import dataset
from engine.recognition import status, classify_samples, evidence_from_scores, _load_model, WEIGHTS
from engine.recognition.yamnet import log_mel_patches
from training_runtime import TrainingCancelled


def archive(path, shift=0):
    notes = [{'b': i * .5 + shift, 'c': i % 2, 'x': 0 if i % 2 == 0 else 3, 'y': 0, 'd': 1 if i // 2 % 2 == 0 else 0} for i in range(40)]
    info = {'_version': '2.1.0', '_songName': 'Cache fixture', '_beatsPerMinute': 120,
            '_difficultyBeatmapSets': [{'_beatmapCharacteristicName': 'Standard', '_difficultyBeatmaps': [{'_difficulty': 'Expert', '_beatmapFilename': 'Expert.dat'}]}]}
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('Info.dat', json.dumps(info)); z.writestr('Expert.dat', json.dumps({'version': '3.3.0', 'colorNotes': notes}))


class ImportCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'data'); self.root = pathlib.Path(self.temp.name)
        self.cache = patch.object(dataset, 'CACHE_ROOT', self.root / 'cache'); self.cache.start()
        self.source = self.root / 'source'; self.source.mkdir(); self.dest = self.root / 'dataset'
        archive(self.source / 'one.zip')
    def tearDown(self): self.cache.stop(); self.temp.cleanup()

    def test_warm_scan_never_reopens_zip(self):
        first = dataset.scan_dataset(self.source)
        with patch.object(dataset, '_parse_archive', side_effect=AssertionError('Reopened unchanged ZIP')):
            second = dataset.scan_dataset(self.source)
        self.assertEqual(first['noteCount'], second['noteCount']); self.assertEqual(second['archives'][0]['metadataCache'], 'hit')

    def test_modified_archive_invalidates_metadata(self):
        before = dataset.file_digest(self.source / 'one.zip'); dataset.scan_dataset(self.source)
        archive(self.source / 'one.zip', .25)
        after = dataset.scan_dataset(self.source)
        self.assertNotEqual(before, after['archives'][0]['sha256']); self.assertEqual(after['archives'][0]['metadataCache'], 'created')

    def test_unchanged_import_skips_validation_and_copy(self):
        first = dataset.import_references_fast(self.source, self.dest)
        with patch.object(dataset, 'import_archive', side_effect=AssertionError('Reimported unchanged source')):
            second = dataset.import_references_fast(self.source, self.dest)
        self.assertEqual(first['processed'], 1); self.assertEqual(second['processed'], 0); self.assertEqual(second['unchanged'], 1)

    def test_corrupted_destination_is_repaired(self):
        first = dataset.import_references_fast(self.source, self.dest)
        destination = pathlib.Path(first['accepted'][0]['path']); destination.write_bytes(b'corruption')
        second = dataset.import_references_fast(self.source, self.dest)
        self.assertEqual(second['processed'], 1); self.assertEqual(dataset.scan_dataset(self.dest)['mapCount'], 1)

    def test_cancellation_does_not_commit_manifest(self):
        with self.assertRaises(TrainingCancelled): dataset.import_references_fast(self.source, self.dest, cancel_callback=lambda: True)
        self.assertFalse((self.dest / '.reference-imports.json').is_file())

    def test_temporary_io_failure_is_retried_without_source_change(self):
        with patch.object(dataset, 'import_archive', side_effect=PermissionError('Temporary sharing violation')):
            first = dataset.import_references_fast(self.source, self.dest)
        self.assertTrue(first['rejected'][0]['retryable'])
        second = dataset.import_references_fast(self.source, self.dest)
        self.assertEqual(second['processed'], 1); self.assertEqual(len(second['accepted']), 1)

    def test_invalid_content_is_not_reparsed_until_it_changes(self):
        (self.source / 'one.zip').write_bytes(b'not a zip')
        first = dataset.import_references_fast(self.source, self.dest)
        self.assertFalse(first['rejected'][0]['retryable'])
        with patch.object(dataset, 'import_archive', side_effect=AssertionError('Revalidated permanent failure')):
            second = dataset.import_references_fast(self.source, self.dest)
        self.assertEqual(second['processed'], 0); self.assertEqual(len(second['rejected']), 1)

    def test_old_unclassified_rejection_is_retried(self):
        self.dest.mkdir()
        source = self.source / 'one.zip'
        old = {'version': dataset.ARCHIVE_PARSE_VERSION, 'sources': {str(source.resolve()): {
            'fingerprint': dataset._fingerprint(source), 'rejected': {'name': source.name, 'reason': 'Permission denied'}}}}
        (self.dest / '.reference-imports.json').write_text(json.dumps(old), 'utf-8')
        result = dataset.import_references_fast(self.source, self.dest)
        self.assertEqual(result['processed'], 1); self.assertEqual(len(result['accepted']), 1)


class Recognition(unittest.TestCase):
    def test_piano_is_not_inferred_as_drums_from_attack_band(self):
        analysis = {'duration': 2., 'frames': [{'time': 0., 'energy': .8, 'bass': .7, 'onset': .5, 'high': .4}, {'time': 2., 'energy': .8}],
                    'musicAnalysis': {'roleOnsets': {'lowPercussion': [{'time': .17, 'strength': .8}], 'midTransient': [{'time': .523, 'strength': .9}], 'tonalChange': [{'time': 1.127, 'strength': .8}]}}}
        scores = np.zeros((4, 521), np.float32); scores[:, 148] = .9
        evidence = evidence_from_scores(analysis, scores, np.array([0., .48, .96, 1.44]))
        self.assertEqual({event['source'] for event in evidence['events']}, {'piano'})
        self.assertEqual([event['time'] for event in evidence['events']], [.17, .523, 1.127])
        self.assertEqual(max(frame['sources']['drums'] for frame in evidence['frames']), 0.)

    def test_music_without_known_instrument_retains_measured_attacks(self):
        analysis = {'duration': 2., 'frames': [{'time': 0., 'energy': .8, 'onset': .5}, {'time': 2., 'energy': .8}],
                    'musicAnalysis': {'roleOnsets': {'lowPercussion': [{'time': .17, 'strength': .8}], 'midTransient': [{'time': .523, 'strength': .9}], 'tonalChange': [{'time': 1.127, 'strength': .8}]}}}
        scores = np.zeros((4, 521), np.float32); scores[:, 132] = .95
        evidence = evidence_from_scores(analysis, scores, np.array([0., .48, .96, 1.44]))
        self.assertEqual({event['source'] for event in evidence['events']}, {'other'})
        self.assertEqual([event['time'] for event in evidence['events']], [.17, .523, 1.127])
        self.assertTrue(any(frame['sources']['other'] > 0. for frame in evidence['frames']))
        self.assertTrue(all(event['classificationUncertain'] for event in evidence['events']))
        self.assertEqual([event['strength'] for event in evidence['events']], [.8, .9, .8])

    @unittest.skipUnless(status()['ready'], 'Optional trained weights are not installed')
    def test_real_trained_silence_inference(self):
        result = classify_samples(np.zeros(48000, np.float32), device='cpu')
        self.assertEqual(result['scores'].shape[1], 521)
        self.assertIn(494, np.argsort(result['scores'].mean(0))[-10:])

    @unittest.skipUnless(status()['ready'], 'Optional trained weights are not installed')
    def test_tensorflow_feature_and_network_agreement(self):
        import tensorflow as tf
        from tf_keras import layers, Model
        samples = np.random.default_rng(903).normal(0, .1, 32000).astype(np.float32)
        patches = log_mel_patches(samples)
        minimum = 15600; length = minimum + max(0, int(np.ceil((len(samples) - minimum) / 7680))) * 7680
        padded = tf.constant(np.pad(samples, (0, length - len(samples))))
        magnitude = tf.abs(tf.signal.stft(padded, 400, 160, fft_length=512))
        mel = tf.signal.linear_to_mel_weight_matrix(64, 257, 16000, 125, 7500)
        official = tf.signal.frame(tf.math.log(tf.matmul(magnitude, mel) + .001), 96, 48, axis=0).numpy()
        feature_error = float(np.max(np.abs(patches - official)))
        self.assertLess(feature_error, 3e-4)
        inputs = layers.Input(shape=(96, 64)); net = layers.Reshape((96, 64, 1))(inputs)
        sizes = [(2, 32), (1, 64), (2, 128), (1, 128), (2, 256), (1, 256), (2, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (2, 1024), (1, 1024)]
        for i, (stride, channels) in enumerate(sizes, 1):
            prefix = f'layer{i}'
            if i == 1:
                net = layers.Conv2D(channels, 3, stride, padding='same', use_bias=False, name=prefix + '/conv')(net)
                net = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4, name=prefix + '/conv/bn')(net)
                net = layers.ReLU()(net)
            else:
                net = layers.DepthwiseConv2D(3, stride, padding='same', use_bias=False, name=prefix + '/depthwise_conv')(net)
                net = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4, name=prefix + '/depthwise_conv/bn')(net)
                net = layers.ReLU()(net)
                net = layers.Conv2D(channels, 1, padding='same', use_bias=False, name=prefix + '/pointwise_conv')(net)
                net = layers.BatchNormalization(center=True, scale=False, epsilon=1e-4, name=prefix + '/pointwise_conv/bn')(net)
                net = layers.ReLU()(net)
        net = layers.GlobalAveragePooling2D()(net); predictions = layers.Dense(521, activation='sigmoid')(net)
        reference = Model(inputs, predictions); reference.load_weights(str(WEIGHTS))
        expected = reference(official, training=False).numpy()
        model, torch, _ = _load_model(device='cpu')
        with torch.inference_mode(): actual = model(torch.from_numpy(patches))[0].numpy()
        score_error = float(np.max(np.abs(expected - actual)))
        self.assertLess(score_error, 3e-4)
        report = {'model': 'Official pretrained YAMNet', 'comparison': 'Independent TensorFlow vs PyTorch execution',
                  'melFeatureMaxAbsoluteError': feature_error, 'classScoreMaxAbsoluteError': score_error,
                  'classCount': 521, 'testedPatches': len(patches), 'weights': str(WEIGHTS),
                  'assertionTolerance': 3e-4, 'source': 'https://github.com/tensorflow/models/tree/master/research/audioset/yamnet',
                  'accuracyClaim': 'Numerical implementation agreement only; this is not labeled real-song detection accuracy.'}
        (ROOT / 'data/test-artifacts/recognition13-numerical.json').write_text(json.dumps(report, indent=2), 'utf-8')


if __name__ == '__main__': unittest.main()
