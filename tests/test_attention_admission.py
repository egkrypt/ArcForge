"""Independent attention-admission contracts, using annotated source evidence.

These fixtures test which observed events deserve representation, not instrument
recognition accuracy or generated NPS. They deliberately include ignored layers.
"""
import copy
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap
from engine.arcaudio.attention import calculate_attention
from engine.arcaudio.events import select_events
from engine.arcaudio.settings import SOURCES


def observed(second, source, strength=.9, confidence=.45, **extra):
    return {'time': second, 'source': source, 'strength': strength,
            'confidence': confidence, 'timingEvidence': 'observed_onset', **extra}


def decided_frames(**changes):
    frame = {'sources': {s: 0. for s in SOURCES},
             'sourceConfidence': {s: .45 for s in SOURCES},
             'energy': .8, 'spectralActivity': .8, 'rhythmDensity': .8,
             'state': 'music_focused', 'primarySource': 'melody',
             'secondarySource': 'drums', 'allowSecondary': True,
             'secondaryPolicy': 'observed_competition',
             'attention': {'melody': .50, 'drums': .35, 'piano': .15},
             'confidence': .40, 'densityTarget': .8, 'attentionBudget': .8}
    frame.update(changes)
    return [{**copy.deepcopy(frame), 'time': i * .25} for i in range(32)]


def evidence_frames(sources, confidence=.4, **changes):
    strengths = {s: sources.get(s, 0.) for s in SOURCES}
    frame = {'sources': strengths, 'sourceConfidence': {s: confidence for s in SOURCES},
             'energy': .85, 'spectralActivity': .8, 'rhythmDensity': .85}
    frame.update(changes)
    return [{**copy.deepcopy(frame), 'time': i * .25} for i in range(32)]


class AttentionAdmission(unittest.TestCase):
    def test_dense_observed_secondary_accents_survive_without_a_third_layer(self):
        frames = decided_frames()
        events = []
        for i in range(24):
            events.extend([observed(.2 + i * .25, 'melody', .96),
                           observed(.325 + i * .25, 'snare', .90),
                           observed(.3875 + i * .25, 'piano', .99)])
        original = copy.deepcopy((frames, events))
        result = select_events(frames, events)
        selected = [e for e in result if e['selected']]
        self.assertGreater(sum(e['source'] == 'snare' for e in selected), 8)
        self.assertTrue(any(e['source'] == 'melody' for e in selected))
        self.assertFalse(any(e['source'] == 'piano' for e in selected))
        self.assertTrue(all(e['reason'] == 'supporting_layer' for e in result if e['source'] == 'piano'))
        self.assertLess(len(selected), len(events) * .65)
        self.assertTrue(all(e['time'] in {raw['time'] for raw in events} for e in selected))
        self.assertTrue(all(e['admission']['importancePassed'] for e in selected))
        self.assertEqual((frames, events), original)

    def test_quiet_voice_does_not_admit_louder_piano_accompaniment(self):
        frames = evidence_frames({'vocals': .94, 'piano': .55, 'drums': .08},
                                 confidence=.95, energy=.22, spectralActivity=.2, rhythmDensity=.2)
        events = [observed(.2 + i * .5, 'vocals', .8, .95) for i in range(16)]
        events += [observed(.3 + i * .125, 'piano', .99, .95) for i in range(60)]
        decisions = calculate_attention(frames, events)
        self.assertTrue(all(f['primarySource'] == 'vocals' for f in decisions))
        self.assertTrue(all(not f['allowSecondary'] for f in decisions))
        selected = [e for e in select_events(decisions, events) if e['selected']]
        self.assertTrue(selected)
        self.assertEqual({e['source'] for e in selected}, {'vocals'})

    def test_uncertain_but_dense_musical_layers_can_compete(self):
        frames = evidence_frames({'melody': .92, 'drums': .80, 'bass': .18}, confidence=.4)
        events = [observed(.2 + i * .25, 'melody', .96, .4) for i in range(28)]
        events += [observed(.325 + i * .25, 'percussion', .94, .4) for i in range(28)]
        decisions = calculate_attention(frames, events)
        self.assertTrue(any(f['secondaryPolicy'] == 'observed_competition' for f in decisions))
        selected = [e for e in select_events(decisions, events) if e['selected']]
        self.assertTrue(any(e['reason'] == 'secondary_accent' for e in selected))
        self.assertEqual({e['attentionSource'] for e in selected}, {'melody', 'drums'})
        self.assertTrue(all(sum(f['attention'].values()) <= 1.00001 for f in decisions))

    def test_sustained_alias_with_no_attacks_cannot_block_real_secondary(self):
        frames = evidence_frames({'melody': .99, 'lead': .95, 'drums': .55}, confidence=.4)
        for frame in frames:
            frame['sourceConfidence']['melody'] = .95
            frame['sourceConfidence']['lead'] = .99
        events = [observed(.2 + i * .25, 'melody', .94, .95) for i in range(28)]
        events += [observed(.325 + i * .25, 'snare', .90, .4) for i in range(28)]
        decisions = calculate_attention(frames, events)
        self.assertTrue(all(f['primarySource'] == 'melody' for f in decisions))
        self.assertEqual(decisions[16]['secondarySource'], 'drums')
        self.assertGreater(decisions[16]['attention']['lead'], decisions[16]['attention']['drums'])
        self.assertTrue(all(f['secondarySource'] != 'lead' for f in decisions))

    def test_uncertain_secondary_requires_real_strong_attack(self):
        frames = decided_frames()
        events = [observed(.3, 'snare', .94),
                  observed(.9, 'snare', .94, timingEvidence='estimated_grid'),
                  observed(1.5, 'snare', .40),
                  observed(2.1, 'snare', .94, .15),
                  observed(2.7, 'snare', .94, sustained=True)]
        result = select_events(frames, events, settings={'minImportance': 0.})
        self.assertEqual([e['time'] for e in result if e['selected']], [.3])
        self.assertEqual([e['reason'] for e in result[1:]],
                         ['weak_secondary_evidence'] * 3 + ['sustained_background'])
        self.assertFalse(result[1]['admission']['observedAttack'])

    def test_minimum_importance_is_authoritative_in_compensation_path(self):
        frames = decided_frames()
        events = [observed(.3 + i * .3, 'snare', .7) for i in range(20)]
        permissive = select_events(frames, events, settings={'minImportance': .1})
        strict = select_events(frames, events, settings={'minImportance': .8})
        self.assertTrue(any(e['selected'] for e in permissive))
        self.assertFalse(any(e['selected'] for e in strict))
        self.assertTrue(all(e['reason'] == 'low_importance' for e in strict))
        self.assertTrue(all(not e['admission']['importancePassed'] for e in strict))

    def test_role_aliases_cannot_promote_a_different_source(self):
        frames = decided_frames()
        events = [observed(.3, 'kick', .96, roles=['lowPercussion']),
                  observed(.9, 'bass', .99, roles=['lowPercussion'])]
        result = select_events(frames, events)
        self.assertTrue(result[0]['selected'])
        self.assertEqual(result[0]['attentionSource'], 'drums')
        self.assertEqual(result[0]['admission']['target'], 'secondary')
        self.assertFalse(result[1]['selected'])
        self.assertEqual(result[1]['reason'], 'supporting_layer')

    def test_coincident_sources_spend_budget_once_and_zero_budget_stays_empty(self):
        frames = decided_frames()
        events = [observed(.3, 'melody', .95), observed(.31, 'kick', .94),
                  observed(.315, 'snare', .93), observed(.32, 'piano', .99)]
        result = select_events(frames, events)
        selected = [e for e in result if e['selected']]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['coincidentSources'], ['kick', 'melody', 'piano', 'snare'])
        self.assertTrue(any(e['reason'] == 'redundant_simultaneous' for e in result))
        self.assertFalse(any(e['selected'] for e in select_events(frames, events, settings={'maxDensity': 0.})))
        silence = decided_frames(state='silence', attentionBudget=0.)
        silent = select_events(silence, events)
        self.assertTrue(all(e['reason'] == 'no_musical_information' for e in silent))


if __name__ == '__main__': unittest.main(verbosity=2)
