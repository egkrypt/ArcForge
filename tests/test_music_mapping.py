"""Musical behavior tests; no test learns from implementation output."""
import pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
import numpy as np
from engine.music import analyze_music
from engine.generate import generate_map, _mapping_clock
from engine.diagnostics import analyze_playability, repair_playability, recommend_jump_settings


def fixture():
    sr = 22050; bpm = 190.; duration = 24.; offset = .12
    times = np.arange(int(sr * duration)) / sr
    samples = (.70 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)
    rng = np.random.default_rng(71)
    # Loud sustained tone in first/last eight seconds; sparse deliberate attacks.
    for sec in np.arange(offset, 8, 1.2):
        index = int(sec * sr); length = min(1300, len(samples) - index); t = np.arange(length) / sr
        samples[index:index + length] += .10 * np.sin(2 * np.pi * 900 * t) * np.exp(-t * 45)
    # Actual fast attacks only in the middle section. This checks rhythmic
    # behavior independently of amplitude, which is loud throughout the song.
    for sec in np.arange(8 + offset, 16, 60 / bpm / 4):
        index = int(sec * sr); length = min(900, len(samples) - index); t = np.arange(length) / sr
        samples[index:index + length] += .23 * rng.normal(size=length) * np.exp(-t * 110)
    for sec in np.arange(8 + offset, 16, 60 / bpm):
        index = int(sec * sr); length = min(2200, len(samples) - index); t = np.arange(length) / sr
        samples[index:index + length] += .40 * np.sin(2 * np.pi * 65 * t) * np.exp(-t * 45)
    grid = np.arange(offset, duration, 60 / bpm).tolist()
    music = analyze_music(samples, sr, bpm, offset, grid)
    frames = [{'time': float(t), 'energy': .7, 'bass': 0., 'onset': 0., 'high': 0.} for t in np.arange(0, duration, .05)]
    analysis = {'bpm': bpm, 'duration': duration, 'offset': offset, 'beatTimes': grid, 'frames': frames,
                'musicAnalysis': music, 'sections': music['sections']}
    return analysis


class MusicalMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.analysis = fixture()

    def test_loud_sustain_is_not_fast_gameplay(self):
        generated = generate_map(self.analysis, {'nps': 12, 'rank': 9, 'backend': 'local', 'mode': 'Standard', 'seed': 78}, ROOT / 'models/style')
        clock, _ = _mapping_clock(self.analysis, {})
        seconds = np.array([clock.time(n['b']) for n in generated['beatmap']['colorNotes']])
        quiet_nps = int(((seconds >= 1) & (seconds < 7)).sum()) / 6
        drop_nps = int(((seconds >= 9) & (seconds < 15)).sum()) / 6
        self.assertLess(quiet_nps, 2.5)
        self.assertGreater(drop_nps, 8.5)
        self.assertGreater(drop_nps, quiet_nps * 3)
        self.assertTrue(generated['report']['passed'])
        self.assertEqual(generated['report']['consecutiveIdenticalBars'], 0)

    def test_difficulty_changes_measured_density(self):
        low = generate_map(self.analysis, {'nps': 4, 'rank': 5, 'backend': 'local', 'mode': 'Standard', 'seed': 78}, ROOT / 'models/style')
        high = generate_map(self.analysis, {'nps': 12, 'rank': 9, 'backend': 'local', 'mode': 'Standard', 'seed': 78}, ROOT / 'models/style')
        self.assertGreater(high['report']['noteCount'], low['report']['noteCount'] * 1.7)
        self.assertTrue(low['report']['passed'])

    def test_style_has_an_actual_placement_effect(self):
        a = generate_map(self.analysis, {'nps': 10, 'rank': 9, 'backend': 'local', 'mode': 'Standard', 'seed': 79, 'style': 'Flow'}, ROOT / 'models/style')
        b = generate_map(self.analysis, {'nps': 10, 'rank': 9, 'backend': 'local', 'mode': 'Standard', 'seed': 79, 'style': 'Technical'}, ROOT / 'models/style')
        positions = lambda g: [(n['x'], n['y'], n['d']) for n in g['beatmap']['colorNotes']]
        self.assertNotEqual(positions(a), positions(b))

    def test_clock_includes_offset_and_drift(self):
        a = {'bpm': 190, 'duration': 40, 'offset': .13,
             'beatTimes': [.13 + i * .317 if i <= 32 else .13 + 32 * .317 + (i - 32) * .315 for i in range(100)]}
        clock, events = _mapping_clock(a, {})
        self.assertGreaterEqual(len(events), 2)
        self.assertGreater(max(e['m'] for e in events), min(e['m'] for e in events))
        for sec in [1., 12., 26.]: self.assertAlmostEqual(clock.time(clock.beat(sec)), sec, places=5)

    def test_proxy_labels_are_honest(self):
        music = self.analysis['musicAnalysis']
        self.assertIn('no stems', music['instrumentLabels'])
        self.assertIn('lowPercussion', music['roleOnsets'])
        self.assertTrue(music['motifs'])


class PlaytestDiagnostics(unittest.TestCase):
    def test_repairs_preserve_every_hit_time_and_count(self):
        chart = {'colorNotes': [{'b': 1, 'c': 0, 'x': 1, 'y': 0, 'd': 1}, {'b': 1.25, 'c': 0, 'x': 1, 'y': 0, 'd': 1},
                               {'b': 2, 'c': 1, 'x': 2, 'y': 0, 'd': 1}], 'bpmEvents': [],
                 'obstacles': [{'b': .5, 'd': 1, 'x': 1, 'y': 0, 'w': 1, 'h': 5}], 'sliders': [], 'burstSliders': [], 'bombNotes': []}
        analysis = {'bpm': 190, 'duration': 5}
        before = analyze_playability(chart, analysis, {'rank': 9})
        self.assertFalse(before['passed'])
        repaired = repair_playability(chart, analysis, {'rank': 9})
        self.assertEqual(repaired['removedHits'], 0)
        self.assertEqual(repaired['changedHitTimes'], 0)
        self.assertEqual([n['b'] for n in chart['colorNotes']], [n['b'] for n in repaired['beatmap']['colorNotes']])
        self.assertTrue(repaired['report']['passed'])
        self.assertTrue(repaired['changes'])

    def test_one_saber_impossible_timing_is_reported_not_hidden(self):
        chart = {'colorNotes': [{'b': 1, 'c': 1, 'x': 2, 'y': 0, 'd': 1}, {'b': 1.1, 'c': 1, 'x': 2, 'y': 0, 'd': 0}], 'bpmEvents': []}
        result = repair_playability(chart, {'bpm': 190, 'duration': 5}, {'mode': 'OneSaber', 'rank': 9})
        self.assertFalse(result['report']['passed'])
        self.assertEqual(len(result['beatmap']['colorNotes']), 2)

    def test_song_offset_is_included_in_bounds_and_findings(self):
        chart={'colorNotes':[{'b':10,'c':0,'x':0,'y':1,'d':1}],'bpmEvents':[]}
        report=analyze_playability(chart,{'bpm':120,'duration':5},{'songTimeOffset':1})
        self.assertTrue(any(i['code']=='audioBounds' and i['time']==6 for i in report['issues']))

    def test_non_interactable_decoration_does_not_block_a_swing(self):
        chart={'colorNotes':[{'b':1,'c':0,'x':0,'y':0,'d':1}],'bpmEvents':[],
               'obstacles':[{'b':0,'d':4,'x':0,'y':0,'w':1,'h':5,'customData':{'uninteractable':True}}]}
        report=analyze_playability(chart,{'bpm':120,'duration':5},{})
        self.assertFalse(any(i['code']=='wallOcclusion' for i in report['issues']))

    def test_jump_advice_uses_density(self):
        easy = recommend_jump_settings({'peakFourSecondNps': 2})
        hard = recommend_jump_settings({'peakFourSecondNps': 12, 'p95RecoveryMetersPerSecond': 4})
        self.assertGreater(hard['njs'], easy['njs'])
        self.assertGreater(hard['readingTimeMs'], easy['readingTimeMs'])


if __name__ == '__main__': unittest.main()
