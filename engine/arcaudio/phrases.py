"""Offline phrase context, motif recurrence, and evidence-based transitions."""
from __future__ import annotations
import bisect
import numpy as np


def build_phrases(analysis, frames, events):
    """Reuse the existing measured beat/motif clock; do not assume verse/chorus labels."""
    duration = float(analysis.get('duration', frames[-1]['time'] if frames else 0))
    bpm = max(20., float(analysis.get('bpm', 120)))
    beats = np.asarray(analysis.get('beatTimes') or [], dtype=float)
    if len(beats) > 3:
        starts = beats[::4].tolist()
    else:
        starts = np.arange(max(0., float(analysis.get('offset', 0))), duration, 240 / bpm).tolist()
    if not starts or starts[0] > .001: starts.insert(0, 0.)
    starts = sorted(set([s for s in starts if 0 <= s < duration] + [duration]))
    known = analysis.get('musicAnalysis', {}).get('motifs', [])
    event_times = np.asarray([float(e['time']) for e in events], dtype=float)
    frame_times = np.asarray([float(f['time']) for f in frames], dtype=float)
    signatures, phrases = [], []
    for index, (start, end) in enumerate(zip(starts, starts[1:])):
        if end <= start: continue
        selected = (event_times >= start) & (event_times < end)
        positions = (event_times[selected] - start) / (end - start)
        # Source-aware signatures distinguish a voiced phrase from an identically
        # timed drum pattern. This is recurrence evidence, not semantic recognition.
        signature = np.zeros((4, 16), dtype=float)
        for event, position in zip([e for e, yes in zip(events, selected) if yes], positions):
            source = event.get('source', 'other')
            row = 0 if source in ['drums', 'kick', 'snare', 'percussion'] else 1 if source in ['vocals', 'spoken'] else 2 if source == 'bass' else 3
            signature[row, min(15, int(position * 16))] += min(1., float(event.get('strength', 1)))
        vector = signature.ravel(); norm = float(np.linalg.norm(vector))
        normalized = vector / max(norm, 1e-8)
        similarities = [float(normalized @ old) for old in signatures]
        match = int(np.argmax(similarities)) if similarities and max(similarities) >= .93 and norm > 0 else None
        if match is None: match = len(signatures); signatures.append(normalized)
        frame_ids = np.flatnonzero((frame_times >= start) & (frame_times < end))
        mean_energy = float(np.mean([frames[i].get('energy', 0) for i in frame_ids])) if len(frame_ids) else 0.
        mean_density = float(np.mean([frames[i].get('rhythmDensity', 0) for i in frame_ids])) if len(frame_ids) else 0.
        motif = next((m for m in known if abs(float(m.get('start', -100)) - start) < .12), {})
        phrases.append({'id': f'phrase-{index + 1}', 'start': round(float(start), 5), 'end': round(float(end), 5),
                        'motifId': match, 'legacyMotifId': motif.get('motifId'), 'repeated': bool(similarities and max(similarities) >= .93 and norm > 0),
                        'repeatConfidence': round(max(similarities) if similarities and norm > 0 else 0., 4),
                        'eventCount': int(selected.sum()), 'energy': round(mean_energy, 4), 'rhythmDensity': round(mean_density, 4),
                        'evidence': 'Four-beat source/onset signatures and neighboring phrase activity'})
    for index, phrase in enumerate(phrases):
        previous = phrases[index - 1] if index else phrase
        following = phrases[index + 1] if index + 1 < len(phrases) else phrase
        phrase['previousEnergy'] = previous['energy']; phrase['nextEnergy'] = following['energy']
        phrase['buildEvidence'] = round(max(0., following['rhythmDensity'] - previous['rhythmDensity']), 4)
        phrase['endingEvidence'] = round(max(0., phrase['rhythmDensity'] - following['rhythmDensity']), 4)
        phrase['responseTo'] = previous['id'] if index and phrase['rhythmDensity'] < previous['rhythmDensity'] * .55 and phrase['eventCount'] else None
    return phrases


def phrase_at(phrases, second):
    if not phrases: return None
    starts = [p['start'] for p in phrases]
    index = min(len(phrases) - 1, max(0, bisect.bisect_right(starts, second) - 1))
    return phrases[index] if second <= phrases[index]['end'] + .001 else None


def detect_transitions(frames, phrases):
    transitions = []
    for index, current in enumerate(frames):
        if not index: continue
        previous = frames[index - 1]
        if current['state'] != previous['state']:
            transitions.append({'time': current['time'], 'kind': 'context_change', 'from': previous['state'],
                                'to': current['state'], 'confidence': current['confidence']})
        # Compare a context window, not adjacent FFT fluctuations.
        old = frames[max(0, index - 4)]
        for source, entered, exited in [('vocals', 'vocal_entrance', 'vocal_exit'), ('spoken', 'spoken_entrance', 'spoken_exit'),
                                        ('drums', 'drum_entrance', 'drum_break')]:
            a, b = old['sources'].get(source, 0), current['sources'].get(source, 0)
            kind = entered if a < .20 and b >= .50 else exited if a >= .50 and b < .20 else None
            if kind and not any(t['kind'] == kind and current['time'] - t['time'] < 1.5 for t in transitions):
                transitions.append({'time': current['time'], 'kind': kind, 'source': source,
                                    'confidence': round(min(current['sourceConfidence'].get(source, .3), abs(b - a)), 4)})
    for phrase in phrases:
        if phrase.get('repeated'):
            transitions.append({'time': phrase['start'], 'kind': 'repeated_phrase', 'phraseId': phrase['id'],
                                'motifId': phrase['motifId'], 'confidence': phrase['repeatConfidence']})
        if phrase.get('buildEvidence', 0) > .22:
            transitions.append({'time': phrase['start'], 'kind': 'rising_activity', 'phraseId': phrase['id'],
                                'confidence': min(.85, phrase['buildEvidence'])})
        if phrase.get('endingEvidence', 0) > .28:
            transitions.append({'time': phrase['end'], 'kind': 'phrase_release', 'phraseId': phrase['id'],
                                'confidence': min(.85, phrase['endingEvidence'])})
    return sorted(transitions, key=lambda t: (t['time'], t['kind']))
