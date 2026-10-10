"""Regional musical intent changes admission, never source labels or hit times."""
import copy
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
import bootstrap
import numpy as np
from engine.arcaudio import analyze_evidence, apply_focus_to_result
from engine.arcaudio.focus import normalize_focus_regions, focus_at_time, source_allowed
from engine.arcaudio.settings import SOURCES
from engine.regions import validate_regions, config_at_time
from engine.attention_candidates import select_attention_candidates
from engine.continuation import plan_continuation
from modding import BeatClock
from test_regions_continuation import fixture as continuation_fixture


def evidence(events=None):
    strengths = {source: .05 for source in SOURCES}
    strengths.update(drums=.98, kick=.9, snare=.85, melody=.6, piano=.5, spoken=.35)
    frames = [{'time': float(t), 'sources': dict(strengths), 'sourceConfidence': {source: .9 for source in SOURCES},
               'energy': .8, 'rhythmDensity': .8, 'spectralActivity': .8}
              for t in np.arange(0, 12, .25)]
    events = events if events is not None else [event(float(t), source)
        for i, t in enumerate(np.arange(.7, 11.7, .125)) for source in [(['kick', 'piano', 'spoken', 'melody'])[i % 4]]]
    return frames, events


def event(second, source, **changes):
    return {'time': second, 'source': source, 'strength': .95, 'confidence': .9,
            'timingEvidence': 'observed_onset', **changes}


def custom(start=2.113, end=6.127, sources=None, **changes):
    return {'id': 'focus', 'start': start, 'end': end, 'analysisMode': 'custom',
            'sources': sources or {'spoken': 1.}, 'sourcePolicy': 'only', **changes}


def analyze(regions=None, events=None, **settings):
    frames, events = evidence(events)
    return analyze_evidence(frames, events, settings=settings, beat_times=list(np.arange(0, 12, .5)),
                            mapping_regions=regions, duration=12)


class RegionalSourceAttention(unittest.TestCase):
    def test_canonical_range_and_sources_reject_invalid_input(self):
        region = validate_regions([custom(npsRange=[7, 13], transitionSeconds=8)], 12)[0]
        self.assertEqual(region['npsRange'], [7., 13.])
        self.assertEqual(region['sources'], {'spoken': 1.})
        self.assertEqual(region['transitionSeconds'], (region['end'] - region['start']) / 2)
        invalid = [custom(nps=8, npsRange=[7, 13]), custom(npsRange=[12, 7]),
                   custom(sources={'made_up': 1}), custom(sources={'spoken': float('nan')}),
                   custom(sources={'spoken': True}), custom(sources={'spoken': 0}),
                   custom(analysisMode='magic'), custom(sourcePolicy='anything'), custom(transitionSeconds=9)]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_regions([value], 12)

    def test_auto_regions_leave_all_attention_decisions_unchanged(self):
        baseline = analyze()
        automatic = analyze([{'start': 2.113, 'end': 6.127, 'analysisMode': 'auto'}])
        for field in ('frames', 'events', 'selectedEvents', 'summary'):
            self.assertEqual(automatic[field], baseline[field], field)
        self.assertNotIn('mappingFocus', automatic)

    def test_only_spoken_has_exact_half_open_event_boundaries(self):
        times = [2.11299, 2.113, 2.251, 3.1, 4.12699, 4.127, 4.251]
        events = [event(time, source) for time in times for source in ('kick', 'spoken')]
        result = analyze([custom(end=4.127)], events)
        for selected in result['selectedEvents']:
            if 2.113 <= selected['time'] < 4.127:
                self.assertEqual(selected['source'], 'spoken')
        excluded = [e for e in result['events'] if e['source'] == 'kick' and 2.113 <= e['time'] < 4.127]
        self.assertTrue(excluded)
        self.assertTrue(all(e['reason'] == 'manual_source_excluded' for e in excluded))
        for edge in (2.11299, 4.127, 4.251):
            self.assertTrue(all(e.get('mappingFocus') is None for e in result['events'] if e['time'] == edge))
        self.assertTrue(any(e['source'] == 'spoken' and e['selected'] for e in result['events']))

    def test_focus_does_not_change_automatic_admission_outside_its_range(self):
        events = [event(float(t), ['kick', 'spoken', 'piano'][i % 3]) for i, t in enumerate(np.arange(.7, 11.7, .055))]
        baseline, directed = analyze(events=events), analyze([custom()], events)
        original = {e['id']: e for e in baseline['events']}
        for candidate in directed['events']:
            if not 2.113 <= candidate['time'] < 6.127:
                self.assertEqual(candidate, original[candidate['id']])

    def test_generic_drums_include_components_but_kick_is_specific(self):
        drums = normalize_focus_regions([custom(sources={'drums': 1.})], duration=12)[0]
        kick = normalize_focus_regions([custom(sources={'kick': 1.})], duration=12)[0]
        for source in ('kick', 'snare', 'percussion', 'drums'):
            self.assertTrue(source_allowed(drums, source))
        self.assertTrue(source_allowed(kick, 'kick'))
        self.assertFalse(source_allowed(kick, 'snare'))
        self.assertFalse(source_allowed(kick, 'drums'))

    def test_full_mix_considers_secondary_attacks_with_a_finite_budget(self):
        events = [event(float(t), ['kick', 'piano', 'spoken'][i % 3]) for i, t in enumerate(np.arange(.7, 11.7, .045))]
        baseline = analyze(events=events, secondaryAllowance=0.)
        mixed = analyze([{'start': .5, 'end': 11.8, 'analysisMode': 'all'}], events, secondaryAllowance=0.)
        self.assertGreater(sum(e['source'] == 'piano' for e in mixed['selectedEvents']),
                           sum(e['source'] == 'piano' for e in baseline['selectedEvents']))
        self.assertLess(len(mixed['selectedEvents']), len(events) * .8)
        self.assertTrue(any(e['reason'] == 'attention_budget' for e in mixed['events']))
        self.assertTrue(all(e['admission']['importancePassed'] for e in mixed['selectedEvents']))

    def test_custom_weights_prioritize_coincident_information_not_confidence(self):
        events = [event(t, source) for t in (3., 4., 5.) for source in ('piano', 'spoken')]
        result = analyze([custom(sources={'spoken': 1., 'piano': .15})], events)
        self.assertEqual({e['source'] for e in result['selectedEvents']}, {'spoken'})
        self.assertTrue(all(e['confidence'] == .9 for e in result['events']))
        self.assertTrue(all(f['sourceConfidence']['spoken'] == .9 for f in result['frames']))
        self.assertTrue(all(f['sources']['piano'] == .5 for f in result['frames']))
        self.assertFalse(result['mappingFocus']['sourceWeightsAreConfidence'])

    def test_primary_policy_can_keep_a_strong_secondary_but_only_cannot(self):
        events = [event(float(t), ['spoken', 'kick'][i % 2]) for i, t in enumerate(np.arange(2.3, 5.9, .25))]
        primary = analyze([custom(sourcePolicy='primary')], events, minImportance=.1)
        only = analyze([custom()], events, minImportance=.1)
        self.assertTrue(any(e['source'] == 'kick' and e['selected'] for e in primary['events']))
        self.assertFalse(any(e['source'] == 'kick' and e['selected'] for e in only['events']))

    def test_absent_or_rejected_source_reports_no_evidence_without_filling(self):
        missing = analyze([custom()], [event(3., 'kick'), event(4., 'piano')])
        self.assertFalse(missing['selectedEvents'])
        self.assertEqual(missing['mappingFocus']['regions'][0]['observedAttacks'], 0)
        self.assertTrue(missing['mappingFocus']['regions'][0]['warnings'])
        weak = analyze([custom()], [event(3., 'spoken', confidence=.1, strength=.3)], minImportance=.8)
        self.assertFalse(weak['selectedEvents'])
        self.assertEqual(weak['events'][0]['confidence'], .1)
        self.assertEqual(weak['events'][0]['reason'], 'low_importance')

    def test_sustain_or_estimated_grid_is_not_a_requested_source_attack(self):
        result = analyze([custom()], [event(3., 'spoken', sustained=True),
                                     event(4., 'spoken', timingEvidence='estimated_grid')], minImportance=0.)
        self.assertFalse(result['selectedEvents'])
        self.assertEqual({e['reason'] for e in result['events']}, {'sustained_background', 'unobserved_source_timing'})

    def test_backward_focus_and_explicit_auto_mask_are_canonical(self):
        older = [{'start': 2., 'end': 6., 'source': 'spoken'}]
        regions = normalize_focus_regions([{'start': 3., 'end': 4., 'analysisMode': 'auto'}], older, 12)
        self.assertEqual([(r['start'], r['end']) for r in regions], [(2., 3.), (4., 6.)])
        self.assertIsNone(focus_at_time(regions, 3.))
        self.assertEqual(focus_at_time(regions, 2.)['sources'], {'spoken': 1.})
        with self.assertRaises(ValueError):
            normalize_focus_regions([], [{'start': 2, 'end': 6, 'source': 'spoken'},
                                          {'start': 5, 'end': 7, 'source': 'piano'}], 12)

    def test_cached_detection_can_be_redirected_without_relabeling(self):
        original = analyze()
        untouched = copy.deepcopy(original)
        regions = normalize_focus_regions([custom()], duration=12)
        changed = apply_focus_to_result(original, {'duration': 12, 'beatTimes': list(np.arange(0, 12, .5))}, regions)
        self.assertEqual(original, untouched)
        self.assertTrue(all(e['source'] == 'spoken' for e in changed['selectedEvents'] if 2.113 <= e['time'] < 6.127))
        restored = apply_focus_to_result(changed, {'duration': 12, 'beatTimes': list(np.arange(0, 12, .5))}, [])
        self.assertEqual(restored['selectedEvents'], original['selectedEvents'])

    def test_snap_and_hybrid_assist_never_cross_a_focus_boundary(self):
        events = [event(2.113, 'spoken'), event(3., 'spoken'), event(4.127, 'kick')]
        result = analyze([custom(end=4.127)], events)
        analysis = {'duration': 12., 'bpm': 120., 'offset': 0., 'beatTimes': [2.1, 2.6, 3.1, 3.6, 4.1, 4.13, 4.6],
                    '_arcAudioAttention': result}
        config = {'rank': 9, 'nps': 13, 'style': 'Modern', 'arcAudioMode': 'arcaudio3', 'mappingRegions': [custom(end=4.127)]}
        legacy = [{'time': 2.1, 'role': 'midTransient', 'score': 1.}, {'time': 4.13, 'role': 'midTransient', 'score': 1.}]
        points, _, _ = select_attention_candidates(analysis, config, np.random.default_rng(1), {'doubles': np.zeros((3, 16))},
            BeatClock(120), (np.array([0., 12.]), np.ones((2, 4))), legacy=legacy)
        self.assertTrue(any(p['time'] == 2.113 for p in points))
        self.assertFalse(any(p['time'] == 2.1 for p in points))
        self.assertTrue(all(p['source'] == 'spoken' for p in points if 2.113 <= p['time'] < 4.127))
        self.assertTrue(all(not row['start'] < 2.113 < row['end'] and not row['start'] < 4.127 < row['end']
                            for row in analysis['_arcAudioMapperReport']['densityCurve']))

    def test_activity_resolves_nps_range_and_transitions_inherit_at_edges(self):
        regions = validate_regions([custom(start=2., end=8., npsRange=[4, 14], primaryStyle='Challenge',
            overrides={'movement': 90}, transitionSeconds=1.)], 12)
        config = {'nps': 6., 'style': 'Flow', 'philosophy': {'movement': 50}, 'mappingRegions': regions}
        self.assertEqual(config_at_time(config, 4, activity=0.)['_resolvedNps'], 4.)
        self.assertEqual(config_at_time(config, 4, activity=1.)['_resolvedNps'], 14.)
        self.assertEqual(config_at_time(config, 2, activity=1.)['_resolvedNps'], 6.)
        self.assertEqual(config_at_time(config, 2, activity=1.)['philosophy']['movement'], 50.)
        self.assertEqual(config_at_time(config, 2.5, activity=1.)['_resolvedNps'], 10.)
        self.assertEqual(config_at_time(config, 8, activity=1.)['_resolvedNps'], 6.)
        self.assertAlmostEqual(sum(item['weight'] for item in config_at_time(config, 2.5)['styleWeights']), 1.)

    def test_neighbor_fades_meet_without_a_valley_to_global_defaults(self):
        regions = validate_regions([
            {'id': 'A', 'start': 2., 'end': 10., 'nps': 5., 'primaryStyle': 'Flow',
             'overrides': {'movement': 40}, 'transitionSeconds': 1.},
            {'id': 'B', 'start': 10., 'end': 18., 'nps': 13., 'primaryStyle': 'Speed',
             'overrides': {'movement': 80}, 'transitionSeconds': 1.}], 20.)
        config = {'nps': 2., 'style': 'Modern', 'philosophy': {'movement': 20}, 'mappingRegions': regions}
        before, edge, after = [config_at_time(config, t) for t in (9.99999, 10., 10.00001)]
        self.assertAlmostEqual(before['_resolvedNps'], 9., places=4)
        self.assertEqual(edge['_resolvedNps'], 9.)
        self.assertAlmostEqual(after['_resolvedNps'], 9., places=4)
        self.assertEqual(edge['philosophy']['movement'], 60.)
        self.assertEqual({w['style']: w['weight'] for w in edge['styleWeights']}, {'Flow': .5, 'Speed': .5})
        regions[1]['transitionSeconds'] = 0.
        self.assertAlmostEqual(config_at_time(config, 9.99999)['_resolvedNps'], 13., places=3)
        self.assertEqual(config_at_time(config, 10.)['_resolvedNps'], 13.)
        regions[0]['transitionSeconds'] = 0.
        self.assertEqual(config_at_time(config, 9.99999)['_resolvedNps'], 5.)
        self.assertEqual(config_at_time(config, 10.)['_resolvedNps'], 13.)

    def test_range_exceeds_global_target_without_a_second_density_reduction(self):
        events = [event(float(t), 'spoken') for t in np.arange(.7, 11.7, .125)]
        regions = [custom(start=2., end=10., npsRange=[8., 13.])]
        result = analyze(regions, events)
        analysis = {'duration': 12., 'bpm': 120., 'offset': 0., 'beatTimes': list(np.arange(0, 12, .5)),
                    '_arcAudioAttention': result}
        config = {'rank': 9, 'nps': 2., 'style': 'Modern', 'arcAudioMode': 'arcaudio3', 'mappingRegions': regions}
        points, _, _ = select_attention_candidates(analysis, config, np.random.default_rng(1), {'doubles': np.zeros((3, 16))},
            BeatClock(120), (np.array([0., 12.]), np.ones((2, 4))))
        local = [point for point in points if 2 <= point['time'] < 10]
        self.assertGreater(len(local) / 8, 6.)
        rows = [row for row in analysis['_arcAudioMapperReport']['densityCurve'] if 2 <= row['start'] < 10]
        self.assertTrue(rows)
        self.assertTrue(all(8 <= row['targetNps'] <= 13 for row in rows))
        self.assertTrue(all(row['densityGoalPolicy'] == 'activity_resolved_range' for row in rows))
        self.assertTrue(any(row['limitedBy'] for row in rows))

    def test_continuation_keeps_complete_focus_and_split_region_density(self):
        chart, analysis, config = continuation_fixture()
        config['mappingRegions'] = validate_regions([
            custom(start=2., end=5., npsRange=[6, 12], primaryStyle='Speed'),
            {'id': 'rest', 'start': 5., 'end': 9., 'analysisMode': 'auto', 'nps': 3., 'primaryStyle': 'Flow'}], 12)
        plan = plan_continuation(chart, analysis, config, [{'start': 3., 'end': 7.}], {'matchNps': False})
        generated = plan['generationConfig']
        self.assertEqual(generated['_attentionMappingRegions'], config['mappingRegions'])
        self.assertEqual([(r['start'], r['end']) for r in generated['mappingRegions']], [(3., 5.), (5., 7.)])
        self.assertEqual(generated['mappingRegions'][0]['npsRange'], [6., 12.])
        self.assertEqual(generated['mappingRegions'][1]['nps'], 3.)
        self.assertEqual(generated['mappingRegions'][0]['primaryStyle'], 'Speed')

    def test_explicit_gap_focus_overrides_only_its_selected_range(self):
        chart, analysis, config = continuation_fixture()
        config['mappingRegions'] = validate_regions([custom(start=2., end=9., sources={'drums': 1.})], 12)
        gap = custom(start=3., end=7., sources={'spoken': 1.})
        gap['id'] = 'gap'
        generated = plan_continuation(chart, analysis, config, [gap])['generationConfig']
        directives = normalize_focus_regions(generated['_attentionMappingRegions'], duration=12)
        self.assertEqual(focus_at_time(directives, 2.5)['sources'], {'drums': 1.})
        self.assertEqual(focus_at_time(directives, 3.)['sources'], {'spoken': 1.})
        self.assertEqual(focus_at_time(directives, 7.)['sources'], {'drums': 1.})


if __name__ == '__main__':
    unittest.main()
