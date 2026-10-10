"""Overlapping musical evidence must survive unrelated named sound classes.

These controlled tests verify fusion semantics, not real instrument accuracy.
"""
import copy, pathlib, sys, unittest
from unittest.mock import patch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
import numpy as np
from engine.arcaudio.settings import SOURCES
from engine.recognition import evidence_from_scores


def spectral_fixture():
    source = {name: 0. for name in SOURCES}
    source.update({'drums': .60, 'snare': .45, 'bass': .50, 'melody': .80,
                   'lead': .60, 'chords': .65, 'piano': .75, 'vocals': .40})
    frame = {'time': 0., 'sources': source,
             'sourceConfidence': {name: .30 for name in SOURCES},
             'energy': .60, 'evidence': 'mixed_audio_DSP_proxies',
             'componentEvidence': {'harmonicFraction': .70, 'tonalConcentration': .60}}
    return {'frames': [frame], 'events': [], 'method': 'controlled_DSP_fixture'}


def fused(posterior):
    scores = np.zeros((2, 521), np.float32)
    scores[:, 132] = .95
    for index, value in posterior.items(): scores[:, index] = value
    with patch('engine.arcaudio.sources.extract_sources', return_value=copy.deepcopy(spectral_fixture())):
        return evidence_from_scores({'duration': .5}, scores, np.array([0., .48]))['frames'][0]


class OverlappingRecognition(unittest.TestCase):
    def test_named_vocal_does_not_erase_measured_instrumental_components(self):
        before = fused({})
        vocal = fused({24: .95})
        self.assertGreater(vocal['sources']['vocals'], .90)
        for source in ['melody', 'chords', 'lead', 'bass', 'drums', 'snare']:
            self.assertEqual(vocal['sources'][source], before['sources'][source], source)
        self.assertEqual(vocal['sourcePosteriors']['melody'], 0.)
        self.assertLessEqual(vocal['sourceConfidence']['melody'], .30)

    def test_named_piano_does_not_erase_rhythm_bass_or_voice(self):
        before = fused({})
        piano = fused({148: .90})
        self.assertGreater(piano['sources']['piano'], .85)
        for source in ['bass', 'drums', 'snare', 'vocals']:
            self.assertEqual(piano['sources'][source], before['sources'][source], source)

    def test_named_drums_do_not_erase_pitched_components(self):
        before = fused({})
        drums = fused({157: .90})
        self.assertGreater(drums['sources']['drums'], .85)
        for source in ['melody', 'chords', 'lead', 'bass', 'vocals']:
            self.assertEqual(drums['sources'][source], before['sources'][source], source)

    def test_plucked_timbre_is_not_relabelled_as_verified_piano(self):
        harp = fused({194: .80, 148: .01})
        self.assertAlmostEqual(harp['sources']['piano'], .01, places=5)
        self.assertAlmostEqual(harp['instrumentEvidence']['piano'], .01, places=5)
        self.assertAlmostEqual(harp['instrumentEvidence']['plucked_strings'], .80, places=5)
        self.assertGreater(harp['sources']['melody'], .75)

    def test_guitar_class_is_visible_without_asserting_a_lead_role(self):
        guitar = fused({135: .85})
        self.assertAlmostEqual(guitar['instrumentEvidence']['guitar'], .85, places=5)
        self.assertEqual(guitar['sourcePosteriors']['lead'], 0.)
        self.assertEqual(guitar['sources']['lead'], 0.)

    def test_absolute_silence_still_wins_over_class_scores(self):
        baseline = spectral_fixture(); baseline['frames'][0]['energy'] = 0.
        scores = np.zeros((2, 521), np.float32); scores[:, 148] = .90
        with patch('engine.arcaudio.sources.extract_sources', return_value=baseline):
            frame = evidence_from_scores({'duration': .5}, scores, np.array([0., .48]))['frames'][0]
        self.assertEqual(max(frame['sources'].values()), 0.)


if __name__ == '__main__': unittest.main()
