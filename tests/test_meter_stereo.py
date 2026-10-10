"""Known accent phases and genuine stereo signals test musical measurements."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
from engine.meter_stereo import estimate_downbeats, stereo_features


class MetricalPhase(unittest.TestCase):
    def test_strong_second_phase_accents_follow_nonzero_grid_offset(self):
        grid = .17 + np.arange(64) * .30
        roles = {'lowPercussion': [{'time': float(t + .008), 'strength': 1.4 if i % 4 == 2 else .15}
                                   for i, t in enumerate(grid)],
                 'tonalChange': [{'time': float(t), 'strength': .8} for t in grid[2::4]]}
        measured = estimate_downbeats(grid, 200, .17, roles, 20.)
        self.assertEqual(measured['phaseBeat'], 2)
        self.assertGreater(measured['confidence'], .75)
        np.testing.assert_allclose(measured['times'], grid[2::4], atol=1e-5)
        self.assertIn('not verified', measured['meterAssumption'])

    def test_equal_accents_do_not_invent_a_downbeat(self):
        grid = np.arange(32) * .5
        roles = {'lowPercussion': [{'time': float(t), 'strength': 1.} for t in grid]}
        measured = estimate_downbeats(grid, 120, 0, roles, 16.)
        self.assertIsNone(measured['phaseBeat'])
        self.assertEqual(measured['times'], [])
        self.assertEqual(measured['confidence'], 0.)

    def test_off_grid_accents_and_short_audio_are_not_confident_evidence(self):
        roles = {'lowPercussion': [{'time': float(t), 'strength': 3.} for t in np.arange(.25, 16, .5)]}
        measured = estimate_downbeats([], 120, 0, roles, 16.)
        self.assertFalse(measured['available'])
        self.assertIsNone(measured['phaseBeat'])
        self.assertFalse(estimate_downbeats([], 120, 0, {}, 1.)['available'])

    def test_drifted_grid_keeps_its_beat_phase(self):
        grid = np.cumsum(np.linspace(.31, .28, 40))
        roles = {'lowPercussion': [{'time': float(t), 'strength': 1.} for t in grid[1::4]]}
        measured = estimate_downbeats(grid, 200, 0, roles, float(grid[-1] + .5))
        self.assertEqual(measured['phaseBeat'], 1)
        np.testing.assert_allclose(measured['times'], grid[1::4], atol=1e-5)

    def test_one_loud_accent_does_not_set_the_entire_songs_downbeat(self):
        grid = np.arange(64) * .5
        roles = {'lowPercussion': [{'time': float(t), 'strength': 4. if i == 2 else .1}
                                   for i, t in enumerate(grid)]}
        measured = estimate_downbeats(grid, 120, 0, roles, 32.)
        self.assertIsNone(measured['phaseBeat'])
        self.assertEqual(measured['evidence']['supportingBars'], 1)


class StereoMeasurement(unittest.TestCase):
    def test_panned_channels_measure_left_then_right_without_fabricated_correlation(self):
        sr = 8000; tone = np.sin(2 * np.pi * 400 * np.arange(sr) / sr)
        left = np.column_stack([tone, np.zeros(sr)])
        right = np.column_stack([np.zeros(sr), tone])
        measured = stereo_features(np.vstack([left, right]), sr)
        self.assertTrue(measured['available'])
        self.assertEqual([w['pan'] for w in measured['windows']], [-1., -1., 1., 1.])
        self.assertAlmostEqual(measured['summary']['balance'], 0.)
        self.assertTrue(all(w['correlation'] is None for w in measured['windows']))
        self.assertEqual(measured['duration'], 2.)

    def test_measured_correlation_distinguishes_matched_and_inverted_channels(self):
        rng = np.random.default_rng(81); mono = rng.normal(0, .2, 8000)
        matched = stereo_features(np.column_stack([mono, mono * .5]), 8000)
        inverted = stereo_features(np.column_stack([mono, -mono]), 8000)
        self.assertAlmostEqual(matched['summary']['correlation'], 1.)
        self.assertAlmostEqual(matched['summary']['balance'], -.6)
        self.assertAlmostEqual(inverted['summary']['correlation'], -1.)

    def test_mono_and_silence_are_distinguished(self):
        self.assertFalse(stereo_features(np.zeros(1000), 8000)['available'])
        silence = stereo_features(np.zeros((1000, 2)), 8000)
        self.assertTrue(silence['available'])
        self.assertIsNone(silence['summary']['pan'])
        self.assertIsNone(silence['summary']['correlation'])

    def test_integer_pcm_and_partial_window_are_supported(self):
        audio = np.zeros((3000, 2), np.int16); audio[:, 0] = 16384
        measured = stereo_features(audio, 8000)
        self.assertEqual(len(measured['windows']), 1)
        self.assertEqual(measured['windows'][0]['leftRms'], .5)
        self.assertEqual(measured['windows'][0]['end'], .375)
        self.assertEqual(measured['summary']['pan'], -1.)
        self.assertIsNone(measured['summary']['correlation'])


if __name__ == '__main__': unittest.main()
