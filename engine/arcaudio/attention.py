"""Contextual competition and stable foreground selection; detection never places notes."""
from __future__ import annotations
import copy, math
import numpy as np
from scipy import ndimage
from .settings import SOURCES, COMPETITORS, normalize_settings
from .phrases import phrase_at


def _finite(value, fallback=0.):
    try: number = float(value)
    except (TypeError, ValueError): return fallback
    return number if math.isfinite(number) else fallback


def _state_scores(sources, energy, activity, spectral):
    """Soft evidence scores, with a separate competing-information context."""
    voice = max(sources['vocals'], sources['spoken'])
    rhythm = max(sources['drums'], .65 * sources['bass'])
    # A trained recognizer can confirm music without resolving its instrument.
    # Observed unknown attacks are still musical information, never silence or
    # invented drum/piano detections. Retain that uncertainty in the source.
    instrumental = max(sources['piano'], sources['melody'], sources['lead'], sources['chords'], sources['other']*.65)
    strong = np.array([voice, rhythm, instrumental, sources['bass'], sources['other']])
    # A single source's aliases (e.g. kick + drums) never count as competition.
    competing = float(np.sort(strong)[-3:].mean())
    dominance = max(voice, rhythm, instrumental) - float(np.sort([voice, rhythm, instrumental])[-2])
    silence = float(np.clip((.035 - energy) / .035, 0, 1)) * (1 - max(voice, rhythm, instrumental))
    return {
        'silence': silence * 1.5,
        'low_energy_vocal_space': voice * (1 - rhythm) * (1 - instrumental * .50) * (1 - energy * .75),
        'vocal_focused': voice * (1 - rhythm * .55) * (1 - instrumental * .50),
        'rhythm_focused': rhythm * (.6 + .4 * activity) * (1 - voice * .25),
        'music_focused': instrumental * (1 - rhythm * .60) * (1 - voice * .60),
        'vocal_music': min(voice, instrumental) * (1 - rhythm * .35) * 1.15,
        'high_density_overloaded': competing * (1 - dominance) * (.55 + .45 * spectral) * 1.2,
    }


def calculate_attention(frames, events, phrases=None, settings=None):
    settings = normalize_settings(settings); phrases = phrases or []
    if not frames: return []
    frames = sorted(copy.deepcopy(frames), key=lambda f: _finite(f.get('time')))
    times = np.array([max(0., _finite(f.get('time'))) for f in frames])
    step = max(.025, float(np.median(np.diff(times))) if len(times) > 1 else .25)
    smoothing = settings['smoothingSeconds'] / step
    # Both sides of the moment inform attention; event timestamps stay untouched.
    matrix = np.array([[np.clip(_finite(f.get('sources', {}).get(s)), 0, 1) for s in SOURCES] for f in frames])
    confidence = np.array([[np.clip(_finite(f.get('sourceConfidence', {}).get(s), .35), 0, 1) for s in SOURCES] for f in frames])
    smoothed = ndimage.gaussian_filter1d(matrix, sigma=max(.2, smoothing / 2.5), axis=0, mode='nearest')
    event_rates = {s: np.zeros(len(times)) for s in COMPETITORS}
    from .settings import attention_source
    for source in event_rates:
        source_times = np.array(sorted(_finite(e.get('time')) for e in events if attention_source(e.get('source')) == source and not e.get('sustained')))
        event_rates[source] = (np.searchsorted(source_times, times + .75, side='right') - np.searchsorted(source_times, times - .75)) / 1.5
    previous_primary, primary_since = None, -100.
    previous_state, state_since = None, -100.
    result = []
    for i, frame in enumerate(frames):
        second = float(times[i]); raw_sources = dict(zip(SOURCES, matrix[i])); sources = dict(zip(SOURCES, smoothed[i]))
        source_confidence = dict(zip(SOURCES, confidence[i]))
        energy = np.clip(_finite(frame.get('energy')), 0, 1)
        rhythm_density = np.clip(_finite(frame.get('rhythmDensity')), 0, 1)
        spectral = np.clip(_finite(frame.get('spectralActivity'), rhythm_density), 0, 1)
        phrase = phrase_at(phrases, second)
        novelty = np.abs(matrix[i] - smoothed[i])
        state_scores = _state_scores(sources, energy, rhythm_density, spectral)
        proposed_state = max(state_scores, key=state_scores.get)
        proposed_strength = state_scores[proposed_state]
        abrupt = max(raw_sources.values()) < .03 and energy < .003
        if previous_state is None or proposed_state == previous_state:
            state = proposed_state
        elif abrupt or (second - state_since >= max(.50, settings['smoothingSeconds']) and
                         proposed_strength >= state_scores[previous_state] + .035):
            state = proposed_state
        else: state = previous_state
        if state != previous_state: state_since = second
        scores = {}
        for source in COMPETITORS:
            strength = sources[source]
            measured_conf = source_confidence[source]
            # Confidence moderates evidence, but a quiet, high-confidence voice
            # can outrank a louder accompaniment. Track RMS is never this score.
            score = strength * (.35 + .65 * measured_conf)
            source_index = SOURCES.index(source)
            phrase_priority = .08 * settings['phraseAwareness'] * float(novelty[source_index])
            if phrase:
                position = (second - phrase['start']) / max(.1, phrase['end'] - phrase['start'])
                phrase_priority += .10 * settings['phraseAwareness'] * (position < .15 or position > .87)
            score *= .70 + .30 * min(1., event_rates[source][i] / 3.)
            score += phrase_priority * strength
            if source in ['vocals', 'spoken']:
                score *= .65 + settings['vocalPriority']
                score *= 1.35 if state in ['vocal_focused', 'low_energy_vocal_space'] else .72 if state == 'rhythm_focused' else 1.
            if source in ['drums', 'bass']:
                score *= .65 + settings['drumPriority']
                score *= 1.22 if state in ['rhythm_focused', 'high_density_overloaded'] else .70 if state == 'low_energy_vocal_space' else 1.
            if source in ['piano', 'chords', 'melody', 'lead']:
                score *= 1.18 if state == 'music_focused' else .62 if state == 'low_energy_vocal_space' else 1.
            if source == 'background': score *= 1 - .94 * settings['backgroundSuppression']
            # Repeated sustained material is accompaniment, not an extra stream.
            if event_rates[source][i] < .20: score *= 1 - .45 * settings['repetitionSuppression']
            scores[source] = max(0., float(score))
        proposed_primary = max(scores, key=scores.get)
        if max(scores.values()) < .025 or state == 'silence':
            primary = None
        elif previous_primary and proposed_primary != previous_primary and second - primary_since < max(.50, settings['smoothingSeconds']):
            primary = previous_primary
        elif previous_primary and proposed_primary != previous_primary and scores[proposed_primary] < scores.get(previous_primary, 0) * 1.18 + .025:
            primary = previous_primary
        else: primary = proposed_primary
        if primary != previous_primary: primary_since = second
        # Competition distributes a finite budget. This is not ten independent
        # thresholds, and sustained aliases cannot spend the budget twice.
        # A fourth-power winner-take-all budget erased supporting evidence
        # even when the primary classification itself was uncertain. Moderate
        # competition still spends a finite budget without claiming certainty.
        exponent = 1. + 1.25 * settings['competition']
        weights = {s: v ** exponent for s, v in scores.items()}
        if primary and weights.get(primary, 0) > 0:
            weights[primary] *= 1.15  # stability preference, not an absolute lock
        total = sum(weights.values())
        shares = {s: v / total if total > 1e-12 and primary else 0. for s, v in weights.items()}
        rounded_shares = {s: round(float(v), 6) for s, v in shares.items()}
        if sum(rounded_shares.values()) > 1.:
            largest = max(rounded_shares, key=rounded_shares.get)
            rounded_shares[largest] = max(0., rounded_shares[largest] - (sum(rounded_shares.values()) - 1.) - 1e-9)
        ranked = sorted(shares, key=shares.get, reverse=True)
        # A loud sustained alias cannot occupy the secondary slot while an
        # independent observed phrase is ignored. Select among sources that
        # actually supply attacks; the source classifier does not create them.
        secondary = next((s for s in ranked if s != primary and s!='background'
                          and shares[s] >= .08 and event_rates[s][i] >= .40), None)
        primary_evidence = source_confidence.get(primary, 0) if primary else 0
        margin = scores.get(primary, 0) - scores.get(secondary, 0) if primary else 0
        decision_confidence = np.clip(.68 * primary_evidence + .22 * max(0., margin) + .10 * (primary == previous_primary), 0, .99) if primary else .95 if state == 'silence' else .15
        allow_secondary = bool(secondary and settings['secondaryAllowance'] > .05 and decision_confidence >= .50
                               and event_rates[secondary][i] >= .40 and state not in ['silence', 'low_energy_vocal_space']
                               and shares[secondary] >= .12 * (1 - settings['secondaryAllowance']))
        secondary_policy='confident_secondary' if allow_secondary else 'disabled'
        # Instrument identity can be uncertain even when two independently
        # observed musical layers compete clearly. Requiring a calibrated
        # instrument label here made every DSP mix single-source and erased
        # strong alternating accents. This path allows only event-level accents
        # (validated by EventSelector), never a continuous second note stream.
        observed_competition=bool(secondary and primary and settings['secondaryAllowance']>.05
            and state in ['music_focused','rhythm_focused','vocal_music','high_density_overloaded']
            and rhythm_density>=.30 and spectral>=.10
            and event_rates[secondary][i]>=1.
            and source_confidence.get(secondary,0)>=.25
            and scores[secondary]>=scores[primary]*.40)
        if not allow_secondary and observed_competition:
            allow_secondary=True;secondary_policy='observed_competition'
        # An accompaniment does not become an independent stream just because
        # it has many attacks. A clearly dominant voice keeps its phrasing space;
        # mixed choruses can still combine sources when their evidence competes.
        if state == 'vocal_focused' and primary in ['vocals','spoken'] and secondary:
            allow_secondary &= scores[secondary] >= scores[primary] * .55
        if not allow_secondary:secondary_policy='disabled'
        primary_activity = min(1., event_rates.get(primary, np.zeros(len(times)))[i] / 8.) if primary else 0.
        secondary_activity = min(1., event_rates.get(secondary, np.zeros(len(times)))[i] / 8.) if allow_secondary else 0.
        activity = primary_activity + secondary_activity * settings['secondaryAllowance'] * .35
        # Density is information-rate driven. RMS can label a climax but never
        # manufactures activity in a loud sustain or an empty passage.
        density = (activity * .80 + rhythm_density * .20) * (.60 + .40 * decision_confidence)
        if state == 'low_energy_vocal_space': density = min(.40, primary_activity * (.8 + .2 * settings['vocalPriority']))
        if state == 'high_density_overloaded': density *= .90 - .20 * settings['competition']
        if not primary or max(raw_sources.values()) < .01: density = 0.
        density = np.clip(density, 0., settings['maxDensity'])
        budget = np.clip((.18 + .82 * activity) * (.55 + .45 * decision_confidence), 0, 1) if primary else 0.
        if state == 'high_density_overloaded': budget *= .90
        mode = 'combined' if allow_secondary and state == 'vocal_music' else 'primary_secondary' if allow_secondary else 'single_source'
        result.append({**frame, 'time': round(second, 5), 'sources': {s: round(float(raw_sources[s]), 4) for s in SOURCES},
                       'smoothedSources': {s: round(float(sources[s]), 4) for s in SOURCES},
                       'sourceConfidence': {s: round(float(source_confidence[s]), 4) for s in SOURCES},
                       'energy': round(float(energy), 4), 'spectralActivity': round(float(spectral), 4),
                       'rhythmDensity': round(float(rhythm_density), 4),
                       'intensity': round(float(np.clip(_finite(frame.get('intensity'), .35 * energy + .65 * rhythm_density), 0, 1)), 4),
                       'state': state, 'contextScores': {s: round(float(v), 4) for s, v in state_scores.items()},
                       'primarySource': primary, 'secondarySource': secondary,
                       'attention': rounded_shares,
                       'attentionBudget': round(float(budget), 4), 'densityTarget': round(float(density), 4),
                       'confidence': round(float(decision_confidence), 4), 'allowSecondary': allow_secondary,
                       'secondaryPolicy':secondary_policy,
                       'selectionMode': mode, 'phraseId': phrase['id'] if phrase else None})
        previous_state, previous_primary = state, primary
    return result
