"""Measured phrase repetition and a bounded, deterministic placement preference.

Timing and musical recurrence are not errors. The guard discourages accidentally
copying the same movement through unrelated phrases while preserving a motif's
recognizable rhythm. It never adds events, consumes RNG, or rejects valid cuts.
"""
from __future__ import annotations
import collections
import math
from modding import BeatClock, decode_precision

ANGLES = (90, 270, 180, 0, 135, 45, 225, 315, 0)


def _bar(beat):
    return int((float(beat) + 1e-7) // 4)


def _phase(beat):
    return round(float(beat) % 4, 3) % 4


def _position(note):
    return decode_precision(note['x']), decode_precision(note['y'])


def phrase_signature(rows):
    """Timing + relative motion; translating a pattern does not disguise copying."""
    anchors, signature = {}, []
    for beat, note in sorted(rows, key=lambda row: (row[0], row[1]['c'])):
        hand = int(note['c']); x, y = _position(note)
        anchors.setdefault(hand, (x, y))
        anchor = anchors[hand]
        signature.append((_phase(beat), hand,
                          round(x - anchor[0], 2), round(y - anchor[1], 2), int(note['d'])))
    return tuple(signature)


def _motion(history, placement, musical_beat):
    if not history:
        return (0., 0., 0., int(placement['d']))
    previous = history[-1]; px, py = _position(previous); x, y = _position(placement)
    phase = previous.get('musicalBeat')
    gap = round(float(musical_beat) - float(phase), 3) if phase is not None else None
    return (gap, round(x - px, 2), round(y - py, 2), int(placement['d']))


class VarietyState:
    """Per-generation history, without files, global caches, or random state."""
    def __init__(self):
        self.bars = collections.defaultdict(list)
        self.motifs = {}
        self.motif_first_bar = {}
        self.anchors = collections.defaultdict(dict)
        self.signature_rows = collections.defaultdict(list)
        self.moves = {0: collections.deque(maxlen=12), 1: collections.deque(maxlen=12)}
        self.ngrams = collections.Counter(); self.sequence_totals = collections.Counter()

    def record(self, history, note, musical_beat, phrase_id=None):
        hand = int(note['c']); move = _motion(history, note, musical_beat)
        if len(self.moves[hand]) >= 3:
            key = (hand, tuple(list(self.moves[hand])[-3:] + [move]))
            self.ngrams[key] += 1; self.sequence_totals[hand] += 1
        self.moves[hand].append(move)
        bar = _bar(musical_beat)
        self.bars[bar].append((float(musical_beat), dict(note)))
        x, y = _position(note); self.anchors[bar].setdefault(hand, (x, y))
        ax, ay = self.anchors[bar][hand]
        self.signature_rows[bar].append((_phase(musical_beat), hand,
                                        round(x - ax, 2), round(y - ay, 2), int(note['d'])))
        if phrase_id is not None:
            identity = str(phrase_id); self.motifs[bar] = identity
            self.motif_first_bar.setdefault(identity, bar)


def variety_multiplier(history, placement, hand, musical_beat, *, state,
                       phrase_id=None, variety=.8, contrast=.9, repeat_variation=.15):
    """A soft [0.35,1.08] weight; physical masks retain final authority.

    Recognized musical repeats receive less suppression. Learned repeatVariation
    describes how much authored motion changes on recurring audio, not a demand
    for randomness. The caller can blend it with user goals and model confidence.
    """
    variety = max(0., min(1., float(variety))); contrast = max(0., min(1., float(contrast)))
    repeat_variation = max(0., min(1., float(repeat_variation)))
    if variety <= 0:
        return 1.
    bar = _bar(musical_beat); identity = None if phrase_id is None else str(phrase_id)
    repeated_music = identity is not None and state.motif_first_bar.get(identity, bar) < bar
    allowance = .20 + .65 * repeat_variation if repeated_music else 1.
    move = _motion(history, placement, musical_beat)
    penalty = 0.
    if len(state.moves[hand]) >= 3:
        key = (hand, tuple(list(state.moves[hand])[-3:] + [move]))
        count = state.ngrams[key]
        dominance = count / max(1, state.sequence_totals[hand])
        if count >= 3 and dominance > .15:
            penalty += variety * allowance * min(3., (count - 2) * dominance) * .45
    x, y = _position(placement); ax, ay = state.anchors[bar].get(hand, (x, y))
    row = (_phase(musical_beat), hand, round(x - ax, 2), round(y - ay, 2), int(placement['d']))
    proposal = tuple(state.signature_rows.get(bar, [])) + (row,)
    for previous_bar in (bar - 1, bar - 2):
        previous = tuple(state.signature_rows.get(previous_bar, []))
        if len(proposal) >= 3 and len(previous) >= len(proposal) and previous[:len(proposal)] == proposal:
            same_motif = identity is not None and state.motifs.get(previous_bar) == identity
            strength = (.20 + .65 * repeat_variation) if same_motif else 1.
            penalty += variety * strength * (.45 + .35 * contrast) / (bar - previous_bar)
    # Bounded weights leave the network's musically coherent vocabulary intact.
    return max(.35, min(1.08, math.exp(-penalty)))


def _motif_for(analysis, beat, second):
    for phrase in analysis.get('_arcAudioAttention', {}).get('phrases', []):
        if phrase['start'] <= second < phrase['end']:
            value = phrase.get('motifId', phrase.get('id'))
            return None if value is None else str(value)
    for motif in analysis.get('musicAnalysis', {}).get('motifs', []):
        if abs(float(motif.get('beat', -1000)) - _bar(beat) * 4) < .1:
            value = motif.get('motifId')
            return None if value is None else str(value)
    return None


def analyze_variety(chart, analysis, config=None):
    """Inspectable counts, not an invented overall quality percentage."""
    config = config or {}; bpm = float(config.get('bpm') or analysis.get('bpm', 120))
    clock = BeatClock(bpm, chart.get('bpmEvents', [])); offset = float(config.get('songTimeOffset', 0))
    grid = analysis.get('beatTimes') or []
    if grid:
        import numpy as np
        def musical_beat(second): return float(np.interp(second, grid, np.arange(len(grid))))
    else:
        def musical_beat(second): return (second - float(analysis.get('offset', 0))) * bpm / 60
    state = VarietyState(); histories = {0: [], 1: []}; bar_motifs = {}
    for note in sorted(chart.get('colorNotes', []), key=lambda n: (n['b'], n['c'])):
        custom = note.get('customData', {}) or {}
        if custom.get('fake') or custom.get('uninteractable') or custom.get('interactable') is False:
            continue
        second = clock.time(note['b']) + offset; beat = musical_beat(second); hand = note['c']
        motif = _motif_for(analysis, beat, second)
        state.record(histories[hand], note, beat, motif)
        histories[hand].append(dict(note, musicalBeat=beat)); bar_motifs[_bar(beat)] = motif
    bars = sorted(state.bars); signatures = {bar: phrase_signature(state.bars[bar]) for bar in bars}
    repeated, intentional, unintentional, motif_pairs = [], [], [], []
    for a, b in zip(bars, bars[1:]):
        if b != a + 1: continue
        equal = signatures[a] == signatures[b]
        same_motif = bar_motifs.get(a) is not None and bar_motifs.get(a) == bar_motifs.get(b)
        if equal:
            repeated.append({'beforeBar': a, 'bar': b, 'musicalMotifMatches': same_motif})
            (intentional if same_motif else unintentional).append(b)
    motif_first = {}
    for bar in bars:
        motif = bar_motifs.get(bar)
        if motif is None: continue
        if motif in motif_first:
            earlier = motif_first[motif]; left, right = signatures[earlier], signatures[bar]
            rhythm_left = [(r[0], r[1]) for r in left]; rhythm_right = [(r[0], r[1]) for r in right]
            identical_rows = sum(a == b for a, b in zip(left, right))
            motif_pairs.append({'motifId': motif, 'firstBar': earlier, 'bar': bar,
                'sameRhythm': rhythm_left == rhythm_right,
                'motionVariation': round(1 - identical_rows / max(len(left), len(right), 1), 4)})
        else: motif_first[motif] = bar
    dominant = [{'hand': hand, 'count': count, 'shareOfHandSequences': round(count / max(1, state.sequence_totals[hand]), 4),
                 'motionSequence': [list(move) for move in sequence]}
                for (hand, sequence), count in state.ngrams.most_common(8)]
    return {'barsWithNotes': len(bars), 'distinctRelativeMotionBars': len(set(signatures.values())),
            'consecutiveIdenticalBars': len(repeated), 'musicallyMatchedRepeatedBars': len(intentional),
            'unmatchedConsecutiveRepeatedBars': len(unintentional), 'repeatedBars': repeated[:128],
            'dominantMotionNgrams': dominant, 'motifVariations': motif_pairs[:128],
            'meanRepeatedPhraseVariation': round(sum(p['motionVariation'] for p in motif_pairs) / len(motif_pairs), 4) if motif_pairs else None,
            'interpretation': 'Same rhythm and translated relative hand motion are counted together. Repeated musical motifs are reported separately, not automatically treated as errors.'}
