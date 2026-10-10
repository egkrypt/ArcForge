"""Honest weak supervision: human hits indicate timing, not a verified instrument."""
from __future__ import annotations
import bisect, collections
from ..features import note_seconds
from ..diagnostics import _physical


def export_human_supervision(analysis, chart, attention=None, tolerance_seconds=.045, settings=None):
    """Export candidate/context/hit relationships without claiming source labels.

    Simultaneous sources are ambiguous; their positive labels receive 1/N weight.
    Unmatched candidates are 'not represented', not proof they are unimportant.
    A mapper can deliberately omit valid events for difficulty or playability.
    """
    if attention is None:
        from . import analyze_attention
        attention = analyze_attention(analysis, settings)
    bpm = float(analysis.get('bpm', 120))
    offset = float(analysis.get('songTimeOffset', 0))
    v3 = chart.get('colorNotes', []); v2 = chart.get('_notes', [])
    if v3:
        hits = sorted(note_seconds(float(note['b']), bpm, offset=offset, events=chart.get('bpmEvents', [])) for note in v3 if _physical(note))
    else:
        hits = sorted(note_seconds(float(note['_time']), bpm, offset=offset) for note in v2 if int(note.get('_type', 0)) in [0, 1] and _physical(note))
    frames = attention.get('frames', []); frame_times = [f['time'] for f in frames]
    raw = []
    for event in attention.get('events', []):
        second = float(event['time']); index = bisect.bisect_left(hits, second)
        nearby = [(abs(hits[j] - second), j) for j in [index - 1, index] if 0 <= j < len(hits)]
        distance, hit_index = min(nearby) if nearby else (None, None)
        positive = distance is not None and distance <= tolerance_seconds
        f = frames[min(len(frames) - 1, max(0, bisect.bisect_right(frame_times, second) - 1))] if frames else {}
        raw.append({'time': second, 'source': event['source'], 'sourceConfidence': event.get('confidence', .3),
                    'strength': event.get('strength', 0), 'importance': event.get('importance', 0),
                    'selectedByAttention': bool(event.get('selected')), 'humanRepresented': positive,
                    'humanTimingDistance': round(distance, 5) if distance is not None else None,
                    'matchedHitIndex': hit_index if positive else None, 'state': f.get('state'),
                    'energy': f.get('energy', 0), 'rhythmDensity': f.get('rhythmDensity', 0),
                    'attentionShare': f.get('attention', {}).get(event.get('attentionSource', event['source']), 0),
                    'contextConfidence': f.get('confidence', 0), 'phraseId': event.get('phraseId')})
    counts = collections.Counter(row['matchedHitIndex'] for row in raw if row['humanRepresented'])
    for row in raw:
        competing = counts.get(row['matchedHitIndex'], 0) if row['humanRepresented'] else 0
        row['ambiguousSourcesAtHit'] = competing
        row['trainingWeight'] = round(1 / competing, 5) if competing else 1.
    return {'schema': 'arcaudio.human-attention-supervision.v1', 'analysisVersion': '2.0',
            'audioIdentity': analysis.get('sourceSha256') or analysis.get('id'), 'toleranceSeconds': tolerance_seconds,
            'hitCount': len(hits), 'candidateCount': len(raw), 'representedCandidates': sum(r['humanRepresented'] for r in raw),
            'rows': raw, 'limitations': ['Weak timing labels are not verified instrument-importance labels.',
                                        'Candidates coincident with a human hit are ambiguous; weights divide across them.',
                                        'Difficulty, playability and intentional omissions affect human note choices.'],
            'trainedModel': False}
