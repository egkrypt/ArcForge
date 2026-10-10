"""Metrical-phase estimates and measurements from actual stereo channels.

The downbeat detector assumes four beats per bar; it does not identify meter.
Stereo balance measures channel energy, not the position of separate instruments.
"""
from __future__ import annotations

import math
import numpy as np


def estimate_downbeats(beat_times, bpm, offset, role_onsets, duration):
    """Score four phases using low-frequency and tonal-change attack accents.

    ``beat_times`` identifies consecutive estimated beats; phaseBeat is the
    zero-based phase within that sequence. Confidence is an evidence-margin
    heuristic, not a calibrated probability of a correct musical downbeat.
    """
    result = {
        'available': False, 'phaseBeat': None, 'confidence': 0., 'times': [],
        'meterAssumption': 'Four beats per bar; musical meter is not verified.',
        'evidence': {'phaseScores': [0., 0., 0., 0.], 'matchedAttacks': 0,
                     'beatCount': 0, 'roleWeights': {'lowPercussion': 1., 'tonalChange': .65},
                     'method': 'Mean low/tonal accent strength per beat for four grid phases'},
        'limitation': 'Accent-based phase estimate; syncopation, other meters and unaccented bars can make it wrong.',
    }
    try:
        duration = float(duration)
        grid = np.asarray([] if beat_times is None else beat_times, dtype=float)
        if grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            result['reason'] = 'Invalid consecutive beat times.'
            return result
        if not math.isfinite(duration) or duration <= 0:
            result['reason'] = 'No positive audio duration.'
            return result
        if len(grid) < 2:
            bpm, offset = float(bpm), float(offset)
            if not math.isfinite(bpm) or bpm <= 0 or not math.isfinite(offset):
                result['reason'] = 'No usable beat grid or tempo.'
                return result
            if len(grid): offset = float(grid[0])
            grid = np.arange(offset, duration, 60. / bpm)
    except (TypeError, ValueError, OverflowError):
        result['reason'] = 'Invalid audio timing.'
        return result
    result['evidence']['beatCount'] = len(grid)
    if len(grid) < 12:
        result['reason'] = 'At least three assumed bars are needed for phase evidence.'
        return result

    # Keep original indices even if the first detected beat precedes audio zero.
    valid_beats = (grid >= 0) & (grid < duration)
    beat_accents = np.zeros(len(grid), dtype=float)
    intervals = np.diff(grid)
    gaps = np.r_[intervals[0], (intervals[:-1] + intervals[1:]) / 2., intervals[-1]]
    roles = role_onsets if isinstance(role_onsets, dict) else {}
    matched = 0
    for role, weight in result['evidence']['roleWeights'].items():
        # Multiple peaks within one channel/beat must not masquerade as accents.
        role_accents = np.zeros(len(grid), dtype=float)
        for attack in roles.get(role, []) or []:
            try:
                second, strength = float(attack['time']), float(attack.get('strength', 1.))
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(second) or not math.isfinite(strength) or strength <= 0 or second < 0 or second >= duration:
                continue
            index = int(np.searchsorted(grid, second))
            candidates = [i for i in (index - 1, index) if 0 <= i < len(grid)]
            index = min(candidates, key=lambda i: abs(float(grid[i]) - second))
            tolerance = min(.085, float(gaps[index]) * .22)
            distance = abs(float(grid[index]) - second)
            if not valid_beats[index] or distance > tolerance:
                continue
            accent = min(strength, 4.) * math.exp(-.5 * (distance / max(tolerance / 2., 1e-6)) ** 2)
            role_accents[index] = max(role_accents[index], accent)
            matched += 1
        beat_accents += role_accents * weight
    scores = np.array([float(beat_accents[p::4][valid_beats[p::4]].mean())
                       if valid_beats[p::4].any() else 0. for p in range(4)])
    order = np.argsort(scores)
    strongest, runner_up = float(scores[order[-1]]), float(scores[order[-2]])
    phase = int(order[-1])
    # A single loud accent must not establish the bar position for a full song.
    informative_bars = supporting_bars = 0
    for start in range(0, len(grid) - 3, 4):
        if not valid_beats[start:start + 4].all(): continue
        accents = beat_accents[start:start + 4]
        local_order = np.argsort(accents)
        local_top = float(accents[local_order[-1]])
        if local_top <= 1e-12 or local_top - float(accents[local_order[-2]]) < local_top * .10:
            continue
        informative_bars += 1
        supporting_bars += int(local_order[-1]) == phase
    consistency = supporting_bars / informative_bars if informative_bars else 0.
    accented_beats = int(np.count_nonzero(beat_accents))
    confidence = max(0., (strongest - runner_up) / max(strongest, 1e-12)) * min(1., accented_beats / 12.) * consistency
    result['evidence'].update(phaseScores=np.round(scores, 5).tolist(), matchedAttacks=matched,
                              accentedBeats=accented_beats, informativeBars=informative_bars,
                              supportingBars=supporting_bars)
    result['confidence'] = round(confidence, 4)
    result['available'] = matched > 0
    if confidence < .08 or supporting_bars < 3:
        result['reason'] = 'No clear accent preference between the assumed phases.'
        return result
    result['phaseBeat'] = phase
    result['times'] = [round(float(t), 5) for t in grid[phase::4] if 0 <= t < duration]
    return result


def stereo_features(samples, sample_rate):
    """Measure 0.5-second stereo windows; mono is explicitly unavailable.

    Balance/pan is normalized right-minus-left channel energy: -1 left, +1
    right. Correlation is measured Pearson correlation and is undefined for
    silence or a channel with no changing signal; those fields are null.
    """
    result = {'available': False, 'channels': 0, 'duration': 0., 'windowSeconds': .5,
              'windows': [], 'summary': {'balance': None, 'pan': None, 'correlation': None},
              'measurement': 'Actual L/R energy balance and centered channel correlation',
              'limitation': 'Pan is a channel-energy proxy, not separated instrument positioning.'}
    try:
        source = np.asarray(samples)
        sample_rate = float(sample_rate)
        result['channels'] = int(source.shape[1]) if source.ndim == 2 else 1 if source.ndim == 1 else 0
        if source.ndim != 2 or source.shape[1] != 2:
            result['reason'] = 'Two actual audio channels are required; mono is not duplicated.'
            return result
        if not math.isfinite(sample_rate) or sample_rate <= 0 or not len(source):
            result['reason'] = 'No usable stereo audio or sample rate.'
            return result
        if not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
            result['reason'] = 'Real numeric audio samples are required.'
            return result
        audio = source.astype(np.float64, copy=False)
        if np.issubdtype(source.dtype, np.integer):
            limits = np.iinfo(source.dtype)
            if limits.min == 0:
                midpoint = (float(limits.max) + 1.) / 2.
                audio = (audio - midpoint) / midpoint
            else: audio = audio / max(abs(int(limits.min)), int(limits.max))
        if not np.isfinite(audio).all():
            result['reason'] = 'Stereo samples contain non-finite values.'
            return result
    except (TypeError, ValueError, OverflowError):
        result['reason'] = 'Invalid stereo audio.'
        return result

    def measurements(count, sums, squares, product):
        powers = squares / count
        total = float(powers.sum())
        balance = float((powers[1] - powers[0]) / total) if total > 1e-20 else None
        variances = np.maximum(0., squares - sums * sums / count)
        denominator = float(np.sqrt(variances[0] * variances[1]))
        correlation = float(np.clip((product - sums[0] * sums[1] / count) / denominator, -1., 1.)) if denominator > 1e-20 else None
        return {'leftRms': round(float(np.sqrt(powers[0])), 6), 'rightRms': round(float(np.sqrt(powers[1])), 6),
                'balance': round(balance, 5) if balance is not None else None,
                'pan': round(balance, 5) if balance is not None else None,
                'correlation': round(correlation, 5) if correlation is not None else None}

    step = max(1, int(round(sample_rate * .5)))
    total_sums, total_squares, total_product = np.zeros(2), np.zeros(2), 0.
    for start in range(0, len(audio), step):
        end = min(start + step, len(audio)); block = audio[start:end]
        sums = block.sum(axis=0); squares = np.einsum('ij,ij->j', block, block)
        product = float(np.dot(block[:, 0], block[:, 1]))
        total_sums += sums; total_squares += squares; total_product += product
        result['windows'].append({'time': round((start + end) / (2. * sample_rate), 5),
                                  'start': round(start / sample_rate, 5), 'end': round(end / sample_rate, 5),
                                  **measurements(end - start, sums, squares, product)})
    result.update(available=True, duration=round(len(audio) / sample_rate, 5),
                  summary=measurements(len(audio), total_sums, total_squares, total_product))
    return result
