"""Boundary constraints and authored-content preservation, independent of ML."""
import copy
import pathlib
import sys
import time
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
from engine.regions import validate_regions, config_at_time, normalize_goals
from engine.continuation import plan_continuation, adapt_continuation, rhythm_bias
from modding import BeatClock, precision, decode_precision


def note(second, hand=0, direction=1, x=None):
    return {'b': second * 2, 'c': hand, 'x': (0 if hand == 0 else 3) if x is None else x,
            'y': 1, 'd': direction, 'a': 0, 'customData': {'fixture': 'authored'}}


def fixture():
    chart = {'version': '3.3.0', 'bpmEvents': [], 'colorNotes': [note(1), note(4), note(7, 1), note(10, 1)],
             'sliders': [], 'burstSliders': [], 'bombNotes': [{'b': .6, 'x': 1, 'y': 0}],
             'obstacles': [{'b': .2, 'd': .4, 'x': 0, 'y': 0, 'w': 1, 'h': 3}],
             'basicBeatmapEvents': [{'b': 0., 'et': 0, 'i': 1, 'f': .5}],
             'customData': {'customEvents': [{'b': .4, 't': 'AnimateTrack', 'd': {'track': 'keep'}}]}}
    analysis = {'bpm': 120., 'duration': 12., 'musicAnalysis': {'sections': [
        {'start': 0, 'end': 6, 'label': 'Drive', 'primaryRole': 'midTransient', 'activity': .7},
        {'start': 6, 'end': 12, 'label': 'Drop', 'primaryRole': 'lowPercussion', 'activity': .9}]}}
    config = {'bpm': 120, 'rank': 9, 'nps': 7, 'style': 'Flow', 'philosophy': {'movement': 55, 'variety': 80}}
    return chart, analysis, config


class MappingRegions(unittest.TestCase):
    def test_bounds_and_overlap_rejected(self):
        for regions in ([{'start': 2, 'end': 13}], [{'start': 3, 'end': 3}],
                        [{'start': 2, 'end': 6}, {'start': 5.9, 'end': 9}],
                        [{'start': float('nan'), 'end': 4}],
                        [{'start': 2, 'end': 4, 'overrides': {'movement': 101}}]):
            with self.assertRaises(ValueError): validate_regions(regions, 12)

    def test_only_explicit_overrides_replace_global_goals(self):
        regions = validate_regions([{'id': 'drop', 'start': 2, 'end': 4, 'primaryStyle': 'Challenge',
            'secondaryStyles': ['Flow', 'Powerhouse'], 'overrides': {'movement': 80, 'contrast': 95}}], 12)
        config = {'style': 'Modern', 'nps': 8, 'philosophy': {'musicality': 85, 'tech': 55,
            'movement': 55, 'variety': 80, 'contrast': 90, 'stamina': 65}, 'mappingRegions': regions}
        local = config_at_time(config, 3)
        self.assertEqual(local['philosophy']['movement'], 80)
        self.assertEqual(local['philosophy']['variety'], 80)
        self.assertEqual(local['philosophy']['stamina'], 65)
        self.assertEqual(local['normalizedGoals']['contrast'], .95)
        self.assertAlmostEqual(sum(v['weight'] for v in local['styleWeights']), 1)
        self.assertEqual(local['styleWeights'][0], {'style': 'Challenge', 'weight': .7})
        self.assertEqual(config_at_time(config, 4)['style'], 'Modern')
        self.assertEqual(config_at_time(config, 1)['philosophy']['movement'], 55)
        self.assertEqual(config['philosophy']['movement'], 55)

    def test_zero_is_an_explicit_goal_not_inheritance(self):
        self.assertEqual(normalize_goals({'movement': 0})['movement'], 0)
        with self.assertRaises(ValueError): normalize_goals({'unknown': 1})


class ContinueMapping(unittest.TestCase):
    def test_both_boundaries_and_every_hit_time_are_preserved(self):
        original, analysis, config = fixture(); before = copy.deepcopy(original)
        ranges = [{'start': 1.5, 'end': 3.7}]
        plan = plan_continuation(original, analysis, config, ranges)
        generated = {'colorNotes': [note(2), note(2.5), note(3)], 'bpmEvents': []}
        result = adapt_continuation(original, generated, analysis, config, plan)
        added = result['beatmap']['colorNotes'][len(original['colorNotes']):]
        self.assertEqual([n['b'] for n in added], [4, 5, 6])
        self.assertEqual([n['d'] for n in added], [0, 1, 0])
        self.assertEqual(result['report']['playtest']['counts'].get('parity', 0), 0)
        self.assertEqual(result['beatmap']['colorNotes'][:len(original['colorNotes'])], original['colorNotes'])
        self.assertEqual(result['beatmap']['customData'], original['customData'])
        self.assertEqual(result['beatmap']['basicBeatmapEvents'], original['basicBeatmapEvents'])
        self.assertEqual(original, before)
        self.assertGreater(result['report']['adaptedPlacements'], 0)

    def test_fill_protects_authored_notes_inside_and_outside(self):
        original, analysis, config = fixture()
        authored = note(2.4, 0, 0); authored['customData']['keep'] = [1, 2, 3]
        original['colorNotes'].append(authored)
        plan = plan_continuation(original, analysis, config, [{'start': 1.5, 'end': 3.8}], {'operation': 'fill'})
        generated = {'colorNotes': [note(2.4), note(3)], 'bpmEvents': []}
        result = adapt_continuation(original, generated, analysis, config, plan)
        self.assertEqual(result['beatmap']['colorNotes'][:len(original['colorNotes'])], original['colorNotes'])
        self.assertEqual(sum(n['b'] == 4.8 for n in result['beatmap']['colorNotes']), 1)
        self.assertTrue(result['report']['skippedCandidates'])
        self.assertEqual(result['report']['changedHitTimes'], 0)

    def test_replace_changes_only_selected_heads_even_crossing_tails(self):
        original, analysis, config = fixture()
        original['colorNotes'].append(note(2.4))
        original['sliders'] = [{'b': 2, 'tb': 5, 'c': 0, 'x': 0, 'y': 1, 'd': 1, 'tx': 0, 'ty': 1, 'td': 0}]
        plan = plan_continuation(original, analysis, config, [{'start': 1.5, 'end': 3.8}], {'operation': 'replace'})
        result = adapt_continuation(original, {'colorNotes': [note(2), note(2.5), note(3)]}, analysis, config, plan)
        self.assertNotIn(4.8, [n['b'] for n in result['beatmap']['colorNotes']])
        self.assertEqual(result['beatmap']['sliders'], original['sliders'])
        self.assertEqual(result['beatmap']['obstacles'], original['obstacles'])
        self.assertTrue(result['report']['preservedOutside'])

    def test_multiple_regions_use_both_their_own_boundaries(self):
        original, analysis, config = fixture()
        plan = plan_continuation(original, analysis, config, [
            {'start': 1.5, 'end': 3.7, 'primaryStyle': 'Flow'},
            {'start': 7.5, 'end': 9.7, 'primaryStyle': 'Challenge', 'overrides': {'movement': 80}}])
        generated = {'colorNotes': [note(t) for t in [2, 2.5, 3]] + [note(t, 1) for t in [8, 8.5, 9]]}
        result = adapt_continuation(original, generated, analysis, config, plan)
        self.assertEqual(result['report']['newNotes'], 6)
        self.assertEqual(result['report']['playtest']['counts'].get('parity', 0), 0)
        self.assertEqual(len(result['report']['contextSummaries']), 2)
        self.assertEqual(result['beatmap']['colorNotes'][:4], original['colorNotes'])

    def test_touching_regions_share_a_continuous_transition(self):
        original, analysis, config = fixture()
        plan = plan_continuation(original, analysis, config, [
            {'start': 1.5, 'end': 2.5, 'primaryStyle': 'Flow'},
            {'start': 2.5, 'end': 3.7, 'primaryStyle': 'Challenge'}])
        result = adapt_continuation(original, {'colorNotes': [note(2), note(2.5), note(3)]}, analysis, config, plan)
        self.assertEqual(result['report']['playtest']['counts'].get('parity', 0), 0)

    def test_new_sustains_match_adapted_heads_and_stop_at_region_end(self):
        original, analysis, config = fixture()
        plan = plan_continuation(original, analysis, config, [{'start': 1.5, 'end': 3.7}])
        generated = {'colorNotes': [note(2), note(2.5), note(3)], 'sliders': [
            {'b': 4, 'tb': 6, 'c': 0, 'x': 0, 'y': 1, 'd': 1, 'tx': 0, 'ty': 1, 'td': 1}],
            'burstSliders': [{'b': 6, 'tb': 8, 'c': 0, 'x': 0, 'y': 1, 'd': 1, 'tx': 0, 'ty': 1, 'sc': 3, 's': 1}]}
        result = adapt_continuation(original, generated, analysis, config, plan)
        self.assertEqual(result['beatmap']['sliders'][0]['d'], 0)
        self.assertEqual(result['beatmap']['sliders'][0]['td'], 0)
        self.assertEqual(result['beatmap']['burstSliders'][0]['tb'], 7.4)
        self.assertEqual(result['beatmap']['burstSliders'][0]['d'], 0)

    def test_local_rhythm_is_a_candidate_preference_not_a_new_timestamp(self):
        original, analysis, config = fixture()
        original['colorNotes'] += [note(.5, 1, 0), note(4.5, 1, 0), note(5, 0, 0)]
        plan = plan_continuation(original, analysis, config, [{'start': 1.5, 'end': 3.7}])
        self.assertEqual(len(plan['contextSummaries'][0]['rhythmPhaseWeights']), 16)
        self.assertNotEqual(rhythm_bias(plan['generationConfig'], 2, 4), rhythm_bias(plan['generationConfig'], 2.125, 4.25))
        self.assertEqual(rhythm_bias(plan['generationConfig'], 10, 20), 1)

    def test_bpm_change_is_remapped_without_moving_audio_hits(self):
        original, analysis, config = fixture()
        original['bpmEvents'] = [{'b': 4, 'm': 180}]
        original['colorNotes'] = []
        plan = plan_continuation(original, analysis, config, [{'start': 3, 'end': 5}], {'matchNps': False})
        generated = {'bpmEvents': [], 'colorNotes': [note(3.5, 0, 0), note(4, 1, 1)]}
        result = adapt_continuation(original, generated, analysis, config, plan)
        clock = BeatClock(120, result['beatmap']['bpmEvents'])
        self.assertEqual([round(clock.time(n['b']), 5) for n in result['beatmap']['colorNotes']], [3.5, 4.])
        self.assertEqual(result['beatmap']['bpmEvents'], original['bpmEvents'])

    def test_imported_song_offset_does_not_shift_generated_audio_hits(self):
        original, analysis, config = fixture()
        config['songTimeOffset'] = .37
        original['colorNotes'] = []
        original['bpmEvents'] = [{'b': 4, 'm': 180}]
        plan = plan_continuation(original, analysis, config, [{'start': 3, 'end': 5}], {'matchNps': False})
        generated = {'colorNotes': [note(3.5, 0, 0), note(4, 1, 1)], 'bpmEvents': []}
        result = adapt_continuation(original, generated, analysis, config, plan)
        clock = BeatClock(120, result['beatmap']['bpmEvents'])
        self.assertEqual([round(clock.time(n['b']) + .37, 5) for n in result['beatmap']['colorNotes']], [3.5, 4.])

    def test_impossible_timing_is_reported_without_touching_authored_content(self):
        original, analysis, config = fixture()
        original['colorNotes'] = [note(1), note(1.1, 0, 0)]
        plan = plan_continuation(original, analysis, config, [{'start': 1.04, 'end': 1.08}], {'operation': 'replace'})
        result = adapt_continuation(original, {'colorNotes': [note(1.06)]}, analysis, config, plan)
        self.assertEqual(result['beatmap']['colorNotes'][:2], original['colorNotes'])
        self.assertEqual(result['beatmap']['colorNotes'][2]['b'], 2.12)
        self.assertTrue(result['report']['needsReview'])
        self.assertTrue(any(i['code'] == 'handTiming' for i in result['report']['boundaryIssues']))

    def test_mapping_extensions_precision_offsets_survive_reranking(self):
        original, analysis, config = fixture()
        plan = plan_continuation(original, analysis, config, [{'start': 1.5, 'end': 3.7}])
        generated = {'colorNotes': [note(t, x=precision(-.08)) for t in (2, 2.5, 3)]}
        for n in generated['colorNotes']: n['y'] = precision(n['y'])
        result = adapt_continuation(original, generated, analysis, config, plan)
        added = result['beatmap']['colorNotes'][4:]
        self.assertTrue(all(abs(n['x']) >= 1000 and abs(n['y']) >= 1000 for n in added))
        for n in added:
            value = decode_precision(n['x'])
            self.assertAlmostEqual(value - round(value), -.08)

    def test_proposal_cost_is_bounded_for_a_dense_gap(self):
        original, analysis, config = fixture()
        original['colorNotes'] = []
        plan = plan_continuation(original, analysis, config, [{'start': 1, 'end': 11}], {'matchNps': False})
        generated = {'colorNotes': [note(1 + i * .12, i % 2) for i in range(80)]}
        started = time.perf_counter()
        result = adapt_continuation(original, generated, analysis, config, plan)
        self.assertEqual(result['report']['newNotes'], 80)
        self.assertLess(time.perf_counter() - started, 8.)


if __name__ == '__main__': unittest.main()
