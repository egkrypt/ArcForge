"""Complete song/cold-cache/lighting/object/cancellation tests for the real learner."""
import io, json, pathlib, sys, tempfile, unittest, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import bootstrap
import numpy as np
from scipy.io import wavfile
from engine.composition import train_composition, composition_profile, composition_status
from training_runtime import TrainingCancelled


def make_archive(path, variation=0, boxes=False):
    sr = 22050; duration = 16.; time = np.arange(int(sr * duration)) / sr
    signal = (.04 * np.sin(2 * np.pi * (440 + variation * 8) * time)).astype(np.float32)
    rng = np.random.default_rng(290 + variation)
    for second in np.arange(8.12, 15.9, .125):
        index = int(second * sr); count = min(1000, len(signal) - index); local = np.arange(count) / sr
        signal[index:index + count] += .65 * rng.normal(size=count) * np.exp(-local * 90)
    audio = io.BytesIO(); wavfile.write(audio, sr, signal)
    notes = []
    for second in np.r_[np.arange(.6, 8, .5), np.arange(8.12, 15.5, .125)]:
        i = len(notes); notes.append({'b': round(float((second - .08) * 200 / 60), 5), 'c': i % 2,
                                     'x': 0 if i % 2 == 0 else 3, 'y': (i // 4 + variation) % 2, 'd': 1 if i // 2 % 2 == 0 else 0})
    lights = []
    for second in np.arange(.6, 8, 1.):
        lights.append({'b': float((second - .08) * 200 / 60), 'et': 1, 'i': 3, 'f': .35, 'customData': {'color': [.15, .2, .7]}})
    for second in np.arange(8.12, 15.9, .125):
        lights.append({'b': float((second - .08) * 200 / 60), 'et': 0, 'i': 6 if int(second * 8) % 2 else 2, 'f': 1.2,
                       'customData': {'color': [1., .1, .2] if int(second * 8) % 2 else [.2, .1, 1.]}})
    chart = {'version': '3.3.0', 'colorNotes': notes, 'bpmEvents': [], 'basicBeatmapEvents': lights,
             'colorBoostBeatmapEvents': [{'b': 26.6, 'o': True}, {'b': 50., 'o': False}],
             'bombNotes': [{'b': 10., 'x': 1, 'y': 0}, {'b': 12., 'x': 1, 'y': 1, 'customData': {'fake': True}}],
             'obstacles': [{'b': 16., 'x': 0, 'y': 0, 'd': 1., 'w': 1, 'h': 5}],
             'sliders': [{'b': 14., 'c': 0, 'x': 0, 'y': 0, 'd': 0, 'tb': 15., 'tx': 0, 'ty': 1, 'tc': 1}],
             'burstSliders': [{'b': 35., 'c': 1, 'x': 3, 'y': 0, 'd': 1, 'tb': 36., 'tx': 3, 'ty': 1, 'sc': 4, 's': 1.}]}
    if boxes: chart['lightColorEventBoxGroups'] = [{'b': 0, 'g': 1, 'e': []}, {'b': 12, 'g': 2, 'e': []}]
    info = {'_version': '2.1.0', '_songName': 'Fixture ' + str(variation), '_songFilename': 'song.wav', '_beatsPerMinute': 200,
            '_songTimeOffset': .08, '_difficultyBeatmapSets': [{'_beatmapCharacteristicName': 'Standard', '_difficultyBeatmaps':
                [{'_difficulty': 'ExpertPlus', '_beatmapFilename': 'ExpertPlus.dat'}, {'_difficulty': 'Expert', '_beatmapFilename': 'Expert.dat'}]}]}
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('Info.dat', json.dumps(info)); archive.writestr('song.wav', audio.getvalue())
        archive.writestr('ExpertPlus.dat', json.dumps(chart)); archive.writestr('Expert.dat', json.dumps(chart))
    return chart, signal


class WholeSongLearning(unittest.TestCase):
    def test_actual_phrase_and_lighting_learning_cached_and_cancel_safe(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as root:
            folder = pathlib.Path(root); data = folder / 'dataset'; model = folder / 'models'; data.mkdir()
            first_chart, samples = None, None
            for i in range(6):
                chart, signal = make_archive(data / f'map-{i}.zip', i, boxes=True)
                if not i: first_chart, samples = chart, signal
            phases = []; report = train_composition(data, model, progress=phases.append)
            self.assertEqual(report['archiveCount'], 6); self.assertEqual(report['mapCount'], 12)
            self.assertGreater(report['phraseExamples'], 100)
            self.assertGreater(report['lightingEvents'], 200)
            # Identical difficulties' lightshows only contribute once per ZIP.
            self.assertEqual(report['uniqueLightingShows'], 6)
            self.assertEqual(report['unsupportedObjects']['lightColorEventBoxGroups'], 12)
            self.assertEqual(report['nativeObjects']['bomb'], 12)
            self.assertEqual(report['nativeObjects']['arc'], 12); self.assertEqual(report['nativeObjects']['chain'], 12)
            self.assertEqual(report['excludedObjects']['nonPhysicalOrTransformed'], 12)
            self.assertEqual(phases[-1]['progress'], 1.)
            self.assertTrue((model / 'composition-report.json').is_file())
            self.assertTrue(composition_status(model)['ready'])
            waveform = folder / 'query.wav'; wavfile.write(waveform, 22050, samples)
            analysis = {'duration': 16., 'bpm': 200, 'audioPath': str(waveform), 'offset': .12, 'beatTimes': np.arange(.12, 16, .3).tolist()}
            quiet = composition_profile(model, analysis, 4., {'rank': 9}); busy = composition_profile(model, analysis, 11., {'rank': 9})
            self.assertTrue(busy['ready']); self.assertGreater(busy['densityRatio'], quiet['densityRatio'] * 1.3)
            self.assertGreater(busy['lighting']['eventsPerSecond'], quiet['lighting']['eventsPerSecond'])
            self.assertGreater(busy['lighting']['eventTypes'].get('0', 0), busy['lighting']['eventTypes'].get('1', 0))
            self.assertGreater(quiet['lighting']['eventTypes'].get('1', 0), quiet['lighting']['eventTypes'].get('0', 0))
            self.assertAlmostEqual(sum(busy['lighting']['eventTypes'].values()), 1.)
            self.assertTrue(busy['lighting']['commonColors']); self.assertTrue(busy['lighting']['colorTransitions'])
            self.assertTrue(busy['matches'][0].get('chart'))
            self.assertEqual(composition_profile(model, analysis, 11., {'rank': 9}), busy)
            cached = train_composition(data, model)
            self.assertEqual(cached['featureCacheHits'], 6)
            old_checkpoint = (model / 'composition.json').read_bytes()
            with self.assertRaises(TrainingCancelled): train_composition(data, model, cancel_callback=lambda: True)
            self.assertEqual((model / 'composition.json').read_bytes(), old_checkpoint)

    def test_untrained_fallback_and_unsafe_archive_are_truthful(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as root:
            folder = pathlib.Path(root); data = folder / 'dataset'; data.mkdir()
            self.assertFalse(composition_profile(folder / 'models', {'duration': 10, 'bpm': 120}, 3)['ready'])
            with zipfile.ZipFile(data / 'unsafe.zip', 'w') as archive:
                archive.writestr('../Info.dat', '{}')
            with self.assertRaises(ValueError): train_composition(data, folder / 'models')
            self.assertFalse((folder / 'models/composition.json').exists())


if __name__ == '__main__': unittest.main(verbosity=2)
