"""Independent attention fixtures and real synthetic-DSP evaluation.

Source-selection tests use explicit known evidence. DSP tests measure known
attack/sustain behavior, never claim real singer or piano recognition accuracy.
"""
import copy, json, pathlib, sys, tempfile, time, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
import numpy as np
from scipy.io import wavfile
from engine.arcaudio import analyze_attention, analyze_evidence, export_human_supervision, normalize_settings
from engine.arcaudio.phrases import build_phrases
from engine.arcaudio.sources import SOURCES
from engine.music import analyze_music


def controlled(sources, energy=.45, density=.4, duration=8., event_rates=None, confidence=.97):
    strengths = {s: float(sources.get(s, 0)) for s in SOURCES}
    frames = [{'time': float(t), 'sources': strengths.copy(), 'sourceConfidence': {s: confidence for s in SOURCES},
               'energy': energy, 'spectralActivity': density, 'rhythmDensity': density, 'intensity': .35 * energy + .65 * density,
               'evidence': 'controlled_fixture'} for t in np.arange(0, duration, .25)]
    events = []
    for source, rate in (event_rates or {max(sources, key=sources.get): 3.}).items():
        for second in np.arange(.11 + .017 * list(SOURCES).index(source), duration, 1 / rate):
            events.append({'time': round(float(second), 5), 'source': source, 'strength': .88, 'confidence': confidence,
                           'roles': ['midTransient'], 'timingEvidence': 'observed_onset'})
    events.sort(key=lambda e: e['time'])
    grid = np.arange(0, duration + .1, .3).tolist()
    phrases = build_phrases({'bpm': 200, 'duration': duration, 'beatTimes': grid}, frames, events)
    return frames, events, phrases, grid


SCENARIOS = {
    'quiet_vocal_piano': ({'vocals': .94, 'piano': .55, 'drums': .10, 'background': .30}, .24, .23, {'vocals': 2., 'piano': 6.}),
    'vocal_full_drums': ({'vocals': .58, 'drums': .96, 'bass': .80, 'piano': .40}, .87, .90, {'vocals': 3., 'drums': 8., 'bass': 4.}),
    'instrumental_drop': ({'drums': .97, 'bass': .87, 'melody': .77}, .93, .95, {'drums': 10., 'bass': 4., 'melody': 5.}),
    'dense_chorus': ({'drums': .92, 'bass': .87, 'piano': .88, 'vocals': .86, 'melody': .85}, .98, .97,
                     {'drums': 8., 'bass': 6., 'piano': 6., 'vocals': 3., 'melody': 6.}),
    'spoken_section': ({'spoken': .95, 'vocals': .20, 'piano': .16, 'drums': .05}, .24, .27, {'spoken': 3.5, 'piano': 2.}),
    'piano_only': ({'piano': .96, 'melody': .25, 'chords': .25}, .48, .40, {'piano': 4.}),
    'drum_only': ({'drums': .96, 'percussion': .90}, .66, .82, {'drums': 8.}),
    'high_energy_instrumental': ({'melody': .94, 'lead': .67, 'chords': .50, 'drums': .12}, .98, .80, {'melody': 8., 'lead': 4.}),
    'competing_instruments': ({'drums': .82, 'bass': .80, 'piano': .94, 'melody': .89, 'vocals': .76}, .90, .94,
                              {'drums': 6., 'bass': 6., 'piano': 8., 'melody': 8., 'vocals': 4.}),
    'sparse_arrangement': ({'melody': .92, 'background': .60}, .18, .07, {'melody': .5}),
}


def scenario_result(name):
    strengths, energy, density, rates = SCENARIOS[name]
    frames, events, phrases, beats = controlled(strengths, energy, density, event_rates=rates)
    return analyze_evidence(frames, events, phrases, beat_times=beats)


class MusicalAttention(unittest.TestCase):
    def test_quiet_voice_gets_focus_not_every_background_piano_note(self):
        result = scenario_result('quiet_vocal_piano')
        self.assertGreater(sum(f['primarySource'] == 'vocals' for f in result['frames']), 28)
        self.assertTrue(result['selectedEvents'])
        self.assertTrue(all(e['source'] == 'vocals' for e in result['selectedEvents']))
        self.assertLess(result['summary']['meanDensityTarget'], .35)

    def test_vocals_do_not_override_full_drums(self):
        result = scenario_result('vocal_full_drums')
        self.assertEqual(set(f['primarySource'] for f in result['frames']), {'drums'})
        self.assertGreater(result['summary']['selectedSourceCounts'].get('drums', 0), result['summary']['selectedSourceCounts'].get('vocals', 0))

    def test_instrumental_drop_keeps_strong_activity(self):
        result = scenario_result('instrumental_drop')
        self.assertEqual(set(f['primarySource'] for f in result['frames']), {'drums'})
        self.assertGreater(result['summary']['meanDensityTarget'], .45)
        self.assertGreater(len(result['selectedEvents']), 30)

    def test_dense_chorus_competes_instead_of_mapping_every_layer(self):
        result = scenario_result('dense_chorus')
        self.assertIn('high_density_overloaded', result['summary']['states'])
        self.assertGreater(result['summary']['eventReduction'], .55)
        for frame in result['frames']: self.assertLessEqual(sum(frame['attention'].values()), 1.00001)
        self.assertTrue(result['summary']['rejectionReasons'].get('supporting_layer'))

    def test_spoken_section_can_supply_off_grid_events(self):
        result = scenario_result('spoken_section')
        self.assertEqual(set(f['primarySource'] for f in result['frames']), {'spoken'})
        self.assertTrue(all(e['source'] == 'spoken' for e in result['selectedEvents']))
        self.assertTrue(any(e['gridDistance'] > .015 for e in result['selectedEvents']))
        input_events = controlled(*SCENARIOS['spoken_section'][:3], event_rates=SCENARIOS['spoken_section'][3])[1]
        self.assertTrue(set(e['time'] for e in result['selectedEvents']).issubset(set(e['time'] for e in input_events)))

    def test_piano_only_has_instrumental_focus(self):
        result = scenario_result('piano_only')
        self.assertEqual(set(f['primarySource'] for f in result['frames']), {'piano'})
        self.assertEqual(set(e['source'] for e in result['selectedEvents']), {'piano'})

    def test_drums_only_and_high_instrumental_keep_distinct_targets(self):
        self.assertEqual(set(f['primarySource'] for f in scenario_result('drum_only')['frames']), {'drums'})
        self.assertEqual(set(f['primarySource'] for f in scenario_result('high_energy_instrumental')['frames']), {'melody'})

    def test_multiple_competing_instruments_have_finite_attention(self):
        result = scenario_result('competing_instruments')
        self.assertGreater(result['summary']['eventReduction'], .55)
        self.assertLessEqual(result['summary']['sourceSwitches'], 1)
        self.assertGreaterEqual(len(result['selectedEvents']), 10)

    def test_sudden_transition_changes_target_within_context_window(self):
        first, a, _, beats = controlled({'vocals': .95, 'piano': .25}, .2, .2, event_rates={'vocals': 2.})
        second, b, _, _ = controlled({'drums': .95, 'bass': .8}, .9, .9, event_rates={'drums': 8.})
        frames = first[:16] + [{**f, 'time': f['time'] + 4} for f in second[:16]]
        events = [e for e in a if e['time'] < 4] + [{**e, 'time': e['time'] + 4} for e in b if e['time'] < 4]
        result = analyze_evidence(frames, events, beat_times=beats)
        self.assertEqual(result['frames'][8]['primarySource'], 'vocals')
        self.assertEqual(result['frames'][22]['primarySource'], 'drums')
        self.assertLessEqual(result['summary']['sourceSwitches'], 2)
        self.assertTrue(any(t['kind'] == 'context_change' for t in result['transitions']))

    def test_repeated_musical_phrases_have_shared_motif_evidence(self):
        frames, _, _, grid = controlled({'piano': .9}, .5, .35, duration=9.6)
        events = [{'time': round(bar + offset, 5), 'source': 'piano', 'strength': .9, 'confidence': .95}
                  for bar in np.arange(0, 9.6, 1.2) for offset in [.1, .4, .7, 1.]]
        phrases = build_phrases({'duration': 9.6, 'bpm': 200, 'beatTimes': grid}, frames, events)
        result = analyze_evidence(frames, events, phrases)
        self.assertGreater(result['summary']['repeatedPhrases'], 4)
        self.assertEqual(len(set(p['motifId'] for p in phrases)), 1)

    def test_sparse_arrangement_stays_sparse_even_if_requested_dense(self):
        result = scenario_result('sparse_arrangement')
        self.assertLess(result['summary']['meanDensityTarget'], .15)
        self.assertLessEqual(len(result['selectedEvents']), 4)

    def test_jitter_does_not_switch_foreground_every_frame(self):
        frames, events, phrases, beats = controlled({'vocals': .85, 'piano': .85}, .6, .6, event_rates={'vocals': 3., 'piano': 3.})
        for index, frame in enumerate(frames):
            frame['sources']['vocals'] += .015 * (-1) ** index
            frame['sources']['piano'] -= .015 * (-1) ** index
        result = analyze_evidence(frames, events, phrases, beat_times=beats)
        self.assertLessEqual(result['summary']['sourceSwitches'], 1)

    def test_sustains_and_duplicate_sources_are_not_independent_hits(self):
        frames, events, phrases, beats = controlled({'drums': .95, 'bass': .8}, .9, .9, event_rates={'drums': 4.})
        events += [{**e, 'source': 'bass', 'sustained': True} for e in events]
        events += [{**e, 'source': 'kick'} for e in events if e['source'] == 'drums']
        result = analyze_evidence(frames, events, phrases, beat_times=beats)
        self.assertFalse(any(e.get('sustained') for e in result['selectedEvents']))
        times = sorted(e['time'] for e in result['selectedEvents'])
        self.assertEqual(len(set(times)), len(times))
        self.assertGreater(result['summary']['rejectionReasons'].get('redundant_simultaneous', 0), 0)

    def test_optional_modes_and_tuning_do_not_mutate_analysis(self):
        analysis = {'duration': 8., 'bpm': 200, 'frames': [], 'musicAnalysis': {'version': 4}}
        original = copy.deepcopy(analysis)
        self.assertFalse(analyze_attention(analysis, mode='off')['enabled'])
        self.assertFalse(analyze_attention(analysis, mode='legacy')['enabled'])
        self.assertEqual(analysis, original)
        frames, events, phrases, grid = controlled({'drums': .95}, .8, .9, event_rates={'drums': 8.})
        a = analyze_evidence(frames, events, phrases, settings={'maxDensity': 0}, beat_times=grid)
        b = analyze_evidence(frames, events, phrases, settings={'maxDensity': .25}, beat_times=grid)
        c = analyze_evidence(frames, events, phrases, settings={'maxDensity': 1}, beat_times=grid)
        self.assertFalse(a['selectedEvents']); self.assertLess(len(b['selectedEvents']), len(c['selectedEvents']))
        self.assertLessEqual(max(f['densityTarget'] for f in b['frames']), .25)
        self.assertEqual(normalize_settings({'competition': float('nan'), 'bad': 4})['competition'], .75)

    def test_uncertainty_is_visible_and_conservative(self):
        frames, events, phrases, beats = controlled({'drums': .95}, .8, .9, event_rates={'drums': 8.})
        confident = analyze_evidence(frames, events, phrases, beat_times=beats)
        for frame in frames: frame['sourceConfidence'] = {s: .15 for s in SOURCES}
        for event in events: event['confidence'] = .15
        unsure = analyze_evidence(frames, events, phrases, beat_times=beats)
        self.assertLess(unsure['summary']['meanConfidence'], confident['summary']['meanConfidence'])
        self.assertLessEqual(len(unsure['selectedEvents']), len(confident['selectedEvents']))
        self.assertTrue(all(not f['allowSecondary'] for f in unsure['frames']))

    def test_human_hit_export_is_weak_supervision_and_ambiguous(self):
        frames, events, phrases, beats = controlled({'drums': .95}, .8, .9, event_rates={'drums': 4.})
        events.append({**events[0], 'source': 'bass'})
        attention = analyze_evidence(frames, events, phrases, beat_times=beats)
        chart = {'colorNotes': [{'b': events[0]['time'] * 200 / 60, 'x': 0, 'y': 0, 'c': 0, 'd': 1}], 'bpmEvents': []}
        rows = export_human_supervision({'bpm': 200, 'id': 'fixture'}, chart, attention)
        positives = [r for r in rows['rows'] if r['humanRepresented']]
        self.assertEqual(len(positives), 2)
        self.assertTrue(all(r['trainingWeight'] == .5 for r in positives))
        self.assertFalse(rows['trainedModel'])
        self.assertTrue(rows['limitations'])

    def test_human_export_uses_import_offset_and_ignores_decorative_notes(self):
        frames, events, phrases, beats = controlled({'drums': .95}, .8, .9, event_rates={'drums': 4.})
        attention = analyze_evidence(frames, events, phrases, beat_times=beats)
        second = events[1]['time']; offset = .18
        chart = {'colorNotes': [{'b': (second - offset) * 200 / 60, 'x': 0, 'y': 0, 'c': 0, 'd': 1},
                               {'b': 1, 'x': 2, 'y': 0, 'c': 1, 'd': 0, 'customData': {'fake': True}},
                               {'b': 2, 'x': 2, 'y': 1, 'c': 1, 'd': 0, 'customData': {'uninteractable': True}}], 'bpmEvents': []}
        report = export_human_supervision({'bpm': 200, 'id': 'fixture', 'songTimeOffset': offset}, chart, attention)
        self.assertEqual(report['hitCount'], 1)
        matched = [r for r in report['rows'] if r['humanRepresented']]
        self.assertEqual(len(matched), 1)
        self.assertAlmostEqual(matched[0]['time'], second)
        self.assertEqual(matched[0]['humanTimingDistance'], 0.)


class ActualDSP(unittest.TestCase):
    def test_sustained_energy_and_measured_attacks_are_distinct(self):
        sr = 22050; duration = 16.; t = np.arange(int(sr * duration)) / sr
        samples = (.55 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        rng = np.random.default_rng(204)
        known_attacks = np.arange(8.12, 15.8, .125)
        for second in known_attacks:
            start = int(second * sr); count = min(1200, len(samples) - start); local = np.arange(count) / sr
            samples[start:start + count] += .55 * rng.normal(size=count) * np.exp(-local * 130)
        grid = np.arange(.12, duration, .3).tolist()
        music = analyze_music(samples, sr, 200, .12, grid)
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as folder:
            audio = pathlib.Path(folder) / 'attention-fixture.wav'; wavfile.write(audio, sr, samples)
            analysis = {'duration': duration, 'bpm': 200, 'offset': .12, 'audioPath': str(audio), 'frames': [], 'beatTimes': grid, 'musicAnalysis': music}
            first = analyze_attention(analysis)
            second = analyze_attention(analysis, {'vocalPriority': .8})
        self.assertTrue(first['limitations'])
        self.assertLessEqual(max(f['sourceConfidence']['vocals'] for f in first['frames']), .30)
        self.assertLessEqual(max(f['sourceConfidence']['piano'] for f in first['frames']), .27)
        quiet = [e for e in first['selectedEvents'] if 1 < e['time'] < 7]
        busy = [e for e in first['selectedEvents'] if 9 < e['time'] < 15]
        self.assertGreater(len(busy), len(quiet) * 2 + 5)
        self.assertFalse(any(e['time'] not in {raw['time'] for raw in first['events']} for e in first['selectedEvents']))
        errors = [min(abs(e['time'] - known_attacks)) for e in busy]
        self.assertLess(float(np.median(errors)), .035)
        self.assertLess(second['performance']['sourceAnalysisSeconds'], first['performance']['sourceAnalysisSeconds'])


if __name__ == '__main__': unittest.main(verbosity=2)
