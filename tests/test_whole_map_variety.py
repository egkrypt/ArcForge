"""Musically explained repetition, bounded preferences, and unchanged hit times."""
import copy
import pathlib
import random
import sys
import unittest
from unittest.mock import patch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
import bootstrap
import numpy as np
from engine.variety import VarietyState, phrase_signature, variety_multiplier, analyze_variety
from engine.generate import generate_map, _mapping_clock
from test_music_mapping import fixture


def phrase(bar, shift=0, variation=False):
    rows = []
    for i in range(8):
        hand = i % 2
        note = {'b': bar * 4 + i / 2, 'c': hand, 'x': (0 if hand == 0 else 3) + shift,
                'y': 1, 'd': 1 if (i // 2) % 2 == 0 else 0, 'a': 0}
        if variation and i in (4, 5): note['y'] = 0
        rows.append((bar * 4 + i / 2, note))
    return rows


class WholeMapVariety(unittest.TestCase):
    def test_learned_envelope_does_not_prune_busy_attention_as_quiet_context(self):
        from engine.generate import _composition_budget
        analysis=fixture()
        points=[{'time':8+i*.08,'musicalBeat':16+i*.16,'chord':False,'score':1.,'targetNps':10,
            'attentionState':'rhythm_focused','section':'rhythm_focused','activity':1.,'source':'kick'} for i in range(25)]
        profile={'ready':True,'densityRatio':.08,'confidence':1.,'doubleFraction':0.,'matches':[]}
        with patch('engine.composition.composition_profile',return_value=profile):
            selected,_,_=_composition_budget((points,.12,16.6),analysis,{'rank':9,'arcAudioMode':'arcaudio','philosophy':{'contrast':90}})
        self.assertGreaterEqual(len(selected),round(len(points)*.9))
        self.assertTrue({p['time'] for p in selected}<={p['time'] for p in points})

    def test_relative_signature_finds_translated_copy_but_respects_rhythm(self):
        original = phrase(0); translated = phrase(1, shift=1)
        self.assertEqual(phrase_signature(original), phrase_signature(translated))
        moved = phrase(1, variation=True)
        self.assertNotEqual(phrase_signature(original), phrase_signature(moved))
        shifted_rhythm = [(beat + .25, note) for beat, note in original]
        self.assertNotEqual(phrase_signature(original), phrase_signature(shifted_rhythm))

    def test_matching_audio_motifs_are_reported_separately_from_accidental_copying(self):
        chart = {'colorNotes': [n for bar in range(4) for _, n in phrase(bar, variation=bar == 3)]}
        analysis = {'bpm': 120, 'duration': 8, 'musicAnalysis': {'motifs': [
            {'beat': 0, 'motifId': 'A'}, {'beat': 4, 'motifId': 'A'},
            {'beat': 8, 'motifId': 'B'}, {'beat': 12, 'motifId': 'A'}]}}
        report = analyze_variety(chart, analysis)
        self.assertEqual(report['consecutiveIdenticalBars'], 2)
        self.assertEqual(report['musicallyMatchedRepeatedBars'], 1)
        self.assertEqual(report['unmatchedConsecutiveRepeatedBars'], 1)
        self.assertEqual(report['distinctRelativeMotionBars'], 2)
        self.assertGreater(report['meanRepeatedPhraseVariation'], 0)

    def test_recurring_musical_theme_receives_less_suppression(self):
        state = VarietyState(); history = {0: [], 1: []}
        for beat, note in phrase(0):
            state.record(history[note['c']], note, beat, 'A')
            history[note['c']].append(dict(note, musicalBeat=beat))
        for beat, note in phrase(1)[:2]:
            state.record(history[note['c']], note, beat, 'A')
            history[note['c']].append(dict(note, musicalBeat=beat))
        beat, note = phrase(1)[2]
        explained = variety_multiplier(history[note['c']], note, note['c'], beat, state=state, phrase_id='A')
        accidental = variety_multiplier(history[note['c']], note, note['c'], beat, state=state, phrase_id='B')
        self.assertGreater(explained, accidental)
        self.assertGreaterEqual(accidental, .35)
        self.assertLessEqual(explained, 1.08)

    def test_guard_is_deterministic_and_does_not_consume_rng(self):
        state = VarietyState(); histories = {0: [], 1: []}
        for bar in range(20):
            for beat, note in phrase(bar):
                state.record(histories[note['c']], note, beat, None)
                histories[note['c']].append(dict(note, musicalBeat=beat))
        note = phrase(20)[0][1]
        before = random.getstate(); np_before = np.random.get_state()
        values = [variety_multiplier(histories[0], note, 0, 80, state=state) for _ in range(20)]
        self.assertEqual(len(set(values)), 1)
        self.assertEqual(random.getstate(), before)
        np_after = np.random.get_state()
        self.assertTrue(np.array_equal(np_before[1], np_after[1]))
        self.assertEqual(np_before[2:], np_after[2:])
        self.assertEqual(variety_multiplier(histories[0], note, 0, 80, state=state, variety=0), 1.)

    def test_optional_learned_motion_changes_layout_without_changing_candidate_times(self):
        analysis = fixture(); config = {'nps': 10, 'rank': 9, 'backend': 'local', 'mode': 'Standard', 'seed': 87, 'style': 'Flow'}
        original_config = copy.deepcopy(config)
        legacy = generate_map(analysis, config, ROOT / 'models/style')
        profile = {'ready': True, 'densityRatio': 1.4, 'movement': .7, 'complexity': .8,
                   'doubleFraction': 1., 'patternVariety': .85, 'repeatVariation': .35,
                   'confidence': .9, 'matches': [{'song': 'independent fixture', 'distance': .1}]}
        with patch('engine.composition.composition_profile', return_value=profile):
            learned = generate_map(analysis, dict(config, learnWholeLevel=True), ROOT / 'models/style')
        self.assertEqual([n['b'] for n in legacy['beatmap']['colorNotes']], [n['b'] for n in learned['beatmap']['colorNotes']])
        self.assertNotEqual([(n['x'], n['y'], n['d']) for n in legacy['beatmap']['colorNotes']],
                            [(n['x'], n['y'], n['d']) for n in learned['beatmap']['colorNotes']])
        self.assertTrue(learned['report']['wholeLevelLearning']['ready'])
        self.assertIn('dominantMotionNgrams', learned['report']['variety'])
        self.assertNotIn('variety', legacy['report'])
        self.assertEqual(config, original_config)
        self.assertTrue(learned['report']['passed'])

    def test_learned_phrase_budget_reduces_events_without_inventing_hits(self):
        analysis = fixture(); config = {'nps': 12, 'rank': 9, 'backend': 'local', 'seed': 89, 'style': 'Flow'}
        original = generate_map(analysis, config, ROOT / 'models/style')
        def profile(_folder, _analysis, second, _config):
            return {'ready': True, 'densityRatio': .08 if 8 <= second < 12 else 1.5,
                    'movement': .5, 'complexity': .5, 'doubleFraction': 1., 'patternVariety': .7,
                    'repeatVariation': .1, 'confidence': 1., 'matches': []}
        with patch('engine.composition.composition_profile', side_effect=profile):
            learned = generate_map(analysis, dict(config, learnWholeLevel=True), ROOT / 'models/style')
        before = {n['b'] for n in original['beatmap']['colorNotes']}
        after = {n['b'] for n in learned['beatmap']['colorNotes']}
        self.assertTrue(after <= before)
        self.assertLess(len(after), len(before))
        clock, _ = _mapping_clock(analysis, config)
        count = lambda result: sum(12 <= clock.time(n['b']) < 16 for n in result['beatmap']['colorNotes'])
        self.assertEqual(count(learned), count(original))
        self.assertEqual(learned['report']['wholeLevelLearning']['candidatesAfter'], len(after))


if __name__ == '__main__': unittest.main()
