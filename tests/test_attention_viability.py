"""A strict filter conflict must not export a near-empty Expert+ as success."""
import copy
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
from engine.attention_viability import assess_attention_viability, require_viable_attention


def reproduction():
    analysis = {'duration': 301.03, '_arcAudioMapperReport': {
        'detectedEvents': 8407, 'eligibleForegroundEvents': 25,
        'summary': {'rejectionReasons': {'low_importance': 6511, 'supporting_layer': 1845},
                    'states': {'silence': 5, 'music_focused': 848, 'rhythm_focused': 126}}}}
    config = {'arcAudioMode': 'arcaudio3', 'nps': 13, 'rank': 9,
              'arcAudioSettings': {'minImportance': .75, 'maxDensity': .7}}
    points = [{'time': 28 + i * 10, 'chord': i == 0} for i in range(25)]
    return analysis, config, points


class AttentionViability(unittest.TestCase):
    def test_exact_26_note_failure_identifies_filter_not_njs(self):
        analysis, config, points = reproduction()
        before = copy.deepcopy((analysis, config, points))
        result = assess_attention_viability(analysis, config, points)
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['candidateHits'], 26)
        self.assertEqual(result['activeNpsCeiling'], 9.1)
        self.assertIn('6,511', result['message'])
        self.assertIn('Reset attention defaults', result['message'])
        self.assertIn('NJS controls approach speed', result['message'])
        self.assertEqual((analysis, config, points), before)
        with self.assertRaisesRegex(ValueError, 'only 26 notes'):
            require_viable_attention(analysis, config, points)

    def test_low_count_alone_does_not_diagnose_sparse_music(self):
        analysis, config, points = reproduction()
        analysis['_arcAudioMapperReport']['detectedEvents'] = 45
        self.assertEqual(assess_attention_viability(analysis, config, points)['status'], 'passed')
        analysis, config, points = reproduction()
        analysis['_arcAudioMapperReport']['summary']['states'] = {'silence': 950, 'music_focused': 20}
        self.assertEqual(assess_attention_viability(analysis, config, points)['status'], 'passed')

    def test_balanced_settings_and_existing_modes_keep_behavior(self):
        analysis, config, points = reproduction()
        config['arcAudioSettings'] = {}
        self.assertEqual(assess_attention_viability(analysis, config, points)['status'], 'passed')
        for mode in ('off', 'legacy'):
            config['arcAudioMode'] = mode
            self.assertEqual(assess_attention_viability(analysis, config, points)['status'], 'not_applicable')

    def test_sections_short_excerpts_and_low_density_are_not_full_song_conflicts(self):
        for override in ({'mappingRegions': [{'start': 30, 'end': 40}]},
                         {'continuationContext': [{'start': 30, 'end': 40}]}, {'nps': 2}):
            analysis, config, points = reproduction()
            config.update(override)
            self.assertEqual(assess_attention_viability(analysis, config, points)['status'], 'passed')
        analysis, config, points = reproduction()
        analysis['duration'] = 12
        self.assertEqual(assess_attention_viability(analysis, config, points)['status'], 'passed')

    def test_njs_is_never_used_as_a_note_density_target(self):
        analysis, config, points = reproduction()
        baseline = assess_attention_viability(analysis, config, points)
        for njs in (5, 14, 40):
            self.assertEqual(assess_attention_viability(analysis, dict(config, njs=njs), points), baseline)

    def test_actual_generator_stops_before_inferno_or_placement(self):
        import numpy as np
        from engine.generate import generate_map
        analysis, config, points = reproduction()
        analysis.update(bpm=182.5, offset=0, audioPath='fixture.ogg')
        artifacts = ROOT / 'data/test-artifacts'
        artifacts.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='viability-', dir=artifacts) as directory:
            np.savez(pathlib.Path(directory) / 'model.npz', fixture=np.zeros(1))
            with patch('engine.music.ensure_music_analysis', side_effect=lambda a: a), \
                 patch('engine.styles.style_profile', return_value={}), \
                 patch('engine.generate._candidate_times', return_value=(points, .12, 16.67)), \
                 patch('engine.inferno_adapter.predict') as inferno:
                with self.assertRaisesRegex(ValueError, 'only 26 notes'):
                    generate_map(analysis, dict(config, backend='hybrid'), directory)
                inferno.assert_not_called()


if __name__ == '__main__':
    unittest.main()
