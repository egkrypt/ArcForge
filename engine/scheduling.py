"""Select supported musical hits under a real hand-recovery budget.

Two hands are two reusable resources. A uniform minimum global spacing is not
equivalent: two close accents may be valid, while three close hits may not be.
This module selects existing opportunities; it never invents or shifts a hit.
"""
from __future__ import annotations
import bisect


def _feasible(times, minimum, hands):
    return all(t - times[i - hands] >= minimum - .0001
               for i, t in enumerate(times) if i >= hands)


def select_supported(pool, quota, minimum, hands=2, prefix=()):
    """Prefer salient hits, then correct a greedy choice's density deficit.

    The chronological fallback maximizes count for identical hand cooldowns.
    It is used only when salience-first selection would miss a reachable quota;
    its unused positions are upgraded by score without lowering that count.
    Prefix contains already chosen times immediately before this phrase.
    """
    quota = max(0, int(quota)); hands = max(1, int(hands))
    ordered = sorted(pool, key=lambda p: (float(p['time']), -float(p.get('score', 0))))
    fixed = sorted(float(t) for t in prefix)[-hands:]
    chosen = []; times = list(fixed)
    for point in sorted(ordered, key=lambda p: (-float(p.get('score', 0)), float(p['time']))):
        if len(chosen) >= quota: break
        index = bisect.bisect_right(times, float(point['time']))
        proposal = times[:index] + [float(point['time'])] + times[index:]
        if _feasible(proposal, minimum, hands): chosen.append(point); times = proposal
    scored_count = len(chosen)
    if scored_count < min(quota, len(ordered)):
        chronological = []; previous = list(fixed)
        for point in ordered:
            if len(chronological) >= quota: break
            second = float(point['time'])
            if len(previous) < hands or second - previous[-hands] >= minimum - .0001:
                chronological.append(point); previous.append(second)
        if len(chronological) > len(chosen):
            chosen = chronological
            # Keep strong accents where they can replace weaker selected hits
            # without sacrificing the corrected note count or recovery budget.
            selected_ids = {id(p) for p in chosen}
            for point in sorted(ordered, key=lambda p: -float(p.get('score', 0))):
                if id(point) in selected_ids: continue
                for old in sorted(chosen, key=lambda p: float(p.get('score', 0))):
                    if float(old.get('score', 0)) >= float(point.get('score', 0)): break
                    trial = [p for p in chosen if p is not old] + [point]
                    if _feasible(sorted(fixed + [float(p['time']) for p in trial]), minimum, hands):
                        chosen = trial; selected_ids.remove(id(old)); selected_ids.add(id(point)); break
    chosen.sort(key=lambda p: p['time'])
    return chosen, {'availableEvents': len(ordered), 'quota': quota,
        'selectedEvents': len(chosen), 'backfilledEvents': max(0, len(chosen) - scored_count),
        'limitedBy': 'musicalEvidence' if len(ordered) < quota else 'physicalRecovery' if len(chosen) < quota else None}


class HandAssignment:
    """Bounded phrase ownership, with alternation as the default motion language."""
    def __init__(self):
        self.phrase_hands = {}; self.recent = []; self.next_hand = 0

    def preferred(self, candidate, histories, minimum, mode='Standard'):
        if mode == 'OneSaber': return 1
        if 'forcedHand' in candidate: return int(candidate['forcedHand'])
        hand = self.next_hand; phrase = candidate.get('audioMotif')
        tonal = candidate.get('role') == 'tonalChange' and candidate.get('section') != 'Drop'
        # Missing motif evidence must never collapse every melodic hit onto red.
        if tonal and phrase is not None:
            identity = str(phrase)
            if identity not in self.phrase_hands:
                counts = [self.recent[-16:].count(h) for h in (0, 1)]
                self.phrase_hands[identity] = counts.index(min(counts)) if counts[0] != counts[1] else hand
            owner = self.phrase_hands[identity]
            history = histories[owner]
            recovered = not history or candidate['time'] - history[-1]['seconds'] >= max(minimum, .28)
            excessive_run = len(self.recent) >= 2 and self.recent[-2:] == [owner, owner]
            imbalance = self.recent[-16:].count(owner) - self.recent[-16:].count(1 - owner)
            if recovered and not excessive_run and imbalance < 4: hand = owner
        return hand

    def record(self, hand):
        self.recent.append(int(hand)); self.recent = self.recent[-32:]
        self.next_hand = 1 - int(hand)
