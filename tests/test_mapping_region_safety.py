"""Independent safety review of manual attention: no invented labels or timing."""
import copy, pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
import bootstrap
import numpy as np
from engine.arcaudio import analyze_evidence, apply_focus_to_result
from engine.arcaudio.focus import normalize_focus_regions
from engine.arcaudio.settings import SOURCES
from engine.attention_candidates import select_attention_candidates
from modding import BeatClock
from test_mapping_source_regions import evidence, custom, event


class MappingRegionSafety(unittest.TestCase):
    def test_requested_spoken_does_not_override_measured_absolute_silence(self):
        frames = [{'time': float(t), 'energy': 0., 'rhythmDensity': 0., 'spectralActivity': 0.,
                   'sources': {s: 0. for s in SOURCES},
                   'sourceConfidence': {s: .9 for s in SOURCES}} for t in np.arange(0, 8, .25)]
        # A stale detector artifact must not make silence musical merely because
        # the user requested that source. The original timing remains inspectable.
        events = [event(t, 'spoken') for t in (2.25, 3.25, 4.25)]
        result = analyze_evidence(frames, events, settings={'minImportance': 0.},
            mapping_regions=[custom(start=2., end=5.)], duration=8.)
        self.assertFalse(result['selectedEvents'])
        self.assertTrue(all(e['reason'] == 'no_musical_information' for e in result['events']))
        self.assertEqual([e['time'] for e in result['events']], [2.25, 3.25, 4.25])

    def test_custom_weights_preserve_detector_confidence_and_raw_timbre(self):
        frames, events = evidence([event(t, source, confidence=.10) for t in (2.31, 3.43, 4.57)
                                  for source in ('spoken', 'piano')])
        original = copy.deepcopy(frames)
        result = analyze_evidence(frames, events, settings={'minImportance': 0.},
            mapping_regions=[custom(start=2., end=6., sources={'spoken': 1., 'piano': .05})], duration=12.)
        self.assertEqual(frames, original)
        self.assertTrue(result['selectedEvents'])
        self.assertEqual({e['source'] for e in result['selectedEvents']}, {'spoken'})
        self.assertTrue(all(e['confidence'] == .10 for e in result['events']))
        self.assertEqual(result['frames'][12]['sources'], original[12]['sources'])
        self.assertEqual(result['frames'][12]['sourceConfidence'], original[12]['sourceConfidence'])

    def test_missing_requested_source_does_not_create_grid_candidates(self):
        frames, events = evidence([event(t, 'kick') for t in np.arange(2.25, 5.9, .25)])
        result = analyze_evidence(frames, events, mapping_regions=[custom(start=2., end=6., sources={'piano': 1.})], duration=12.)
        self.assertFalse(result['selectedEvents'])
        self.assertTrue(result['mappingFocus']['regions'][0]['warnings'])
        self.assertEqual({e['source'] for e in result['events']}, {'kick'})

    def test_auto_attention_restores_original_events_without_mutating_cache(self):
        frames, events = evidence()
        original = analyze_evidence(frames, events, duration=12.)
        cached = copy.deepcopy(original)
        focus = normalize_focus_regions([custom()], duration=12.)
        redirected = apply_focus_to_result(original, {'duration': 12}, focus)
        restored = apply_focus_to_result(redirected, {'duration': 12}, [])
        self.assertEqual(original, cached)
        self.assertEqual(restored['selectedEvents'], cached['selectedEvents'])
        self.assertEqual(restored['frames'], cached['frames'])

    def test_mapper_focus_is_local_and_does_not_mutate_cached_analysis(self):
        times = (2.113, 2.691, 3.207, 4.12699, 4.127)
        frames, events = evidence([event(t, source) for t in times for source in ('kick', 'spoken')])
        result = analyze_evidence(frames, events, duration=12.)
        cached = copy.deepcopy(result)
        analysis = {'duration': 12., 'bpm': 120., 'offset': 0.,
                    'beatTimes': list(np.arange(0, 12.1, .5)), '_arcAudioAttention': result}
        config = {'rank': 9, 'nps': 13, 'style': 'Modern', 'arcAudioMode': 'arcaudio3',
                  'mappingRegions': [custom(end=4.127)]}
        points, _, _ = select_attention_candidates(analysis, config, np.random.default_rng(7),
            {'doubles': np.zeros((3, 16))}, BeatClock(120), (np.array([0., 12.]), np.ones((2, 4))))
        self.assertEqual(result, cached)
        self.assertTrue(points)
        for point in points:
            if 2.113 <= point['time'] < 4.127:
                self.assertEqual(point['source'], 'spoken')
                self.assertIn(point['audioEvidence']['eventTime'], times)
                self.assertGreaterEqual(point['audioEvidence']['eventTime'], 2.113)
                self.assertLess(point['audioEvidence']['eventTime'], 4.127)


if __name__ == '__main__': unittest.main()
