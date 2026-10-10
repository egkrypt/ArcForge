"""Event importance, duplicate rejection and finite information-budget admission."""
from __future__ import annotations
import bisect, copy, math
import numpy as np
from .settings import normalize_settings, attention_source
from .phrases import phrase_at


def _secondary_attack_support(event, frame, relative_share, strength, confidence):
    """Low label confidence may combine sources, but only at an actual accent.

    The context layer has already chosen at most one secondary source. This
    guard cannot promote a third source or turn a sustained/estimated grid
    position into an onset. It is used only by the uncertainty compensation
    policy; confidently identified secondary phrasing retains the usual
    importance and information-budget checks.
    """
    policy = frame.get('secondaryPolicy', 'confident_secondary')
    if policy != 'observed_competition': return True
    return bool(event.get('timingEvidence', 'observed_onset') == 'observed_onset'
                and not event.get('sustained')
                and strength >= .45 and confidence >= .25
                and relative_share >= .12)


def select_events(frames, events, phrases=None, settings=None, beat_times=None, focus_regions=None):
    settings = normalize_settings(settings); phrases = phrases or []
    if not frames: return []
    frame_times = [f['time'] for f in frames]; beats = np.asarray(beat_times or [], dtype=float)
    candidates = []
    events = sorted(copy.deepcopy(events), key=lambda e: float(e['time']))
    focus_regions = focus_regions or []
    if focus_regions:
        from .focus import focus_decision, focus_event_times, same_focus_region
        focus_times = focus_event_times(events)
    seen_source = {}
    for index, event in enumerate(events):
        # Reusing detection after a user changes directives must not retain a
        # previous region's decision metadata on an automatic event.
        event.pop('mappingFocus', None)
        second = float(event['time'])
        if not math.isfinite(second) or second < 0: continue
        frame_id = min(len(frames) - 1, max(0, bisect.bisect_right(frame_times, second) - 1))
        frame = frames[frame_id]
        if focus_regions:
            frame = focus_decision(frame, second, focus_regions, event, focus_times)
        source = event.get('source', 'other'); group = attention_source(source)
        share = frame['attention'].get(group, 0.)
        max_share = max(frame['attention'].values()) if frame['attention'] else 0.
        relative_share = share / max(max_share, 1e-8)
        manual_focus = bool(frame.get('mappingFocus'))
        if manual_focus:
            relative_share = float(frame['_focusRelativeWeight'])
        strength = float(np.clip(float(event.get('strength', .5)), 0, 1))
        confidence = float(np.clip(float(event.get('confidence', frame['sourceConfidence'].get(source, .35))), 0, 1))
        neighbors = [e for e in events[max(0, index - 10):index + 11] if abs(float(e['time']) - second) <= 1. and attention_source(e.get('source')) == group]
        baseline = float(np.median([float(e.get('strength', .5)) for e in neighbors])) if neighbors else strength
        temporal_salience = np.clip(strength / max(baseline, .20), .25, 1.5)
        if len(beats) > 1:
            k = int(np.searchsorted(beats, second)); nearby = beats[max(0, k - 1):min(len(beats), k + 1)]
            distance = float(np.min(np.abs(nearby - second))) if len(nearby) else None
            period = float(np.median(np.diff(beats[max(0, k - 3):min(len(beats), k + 4)]))) if len(beats) > 2 else .5
            # Subdivision support, but an authentic off-grid voice/onset remains
            # eligible. This measures salience, never changes the timestamp.
            phase = abs(((second - beats[max(0, min(len(beats) - 1, k - 1))]) / max(period, .02)) % .25)
            grid_distance = min(phase, .25 - phase) * max(period, .02)
            rhythmic_salience = .55 + .45 * np.exp(-.5 * (grid_distance / .025) ** 2)
        else: distance, grid_distance, rhythmic_salience = None, None, .70
        phrase = phrase_at(phrases, second)
        phrase_salience = 1.
        if phrase:
            position = (second - phrase['start']) / max(.1, phrase['end'] - phrase['start'])
            phrase_salience += .13 * settings['phraseAwareness'] * (position < .12 or position > .88)
            if phrase.get('repeated'): phrase_salience += .05 * settings['phraseAwareness']
        redundant = source in seen_source and second - seen_source[source] < .12 and strength <= baseline * 1.05
        novelty = 1 - .35 * settings['repetitionSuppression'] if redundant else 1.
        importance = strength * (.7 + .3 * temporal_salience) * rhythmic_salience * (.45 + .55 * confidence)
        importance *= ((1 - settings['attentionStrength']) + settings['attentionStrength'] * (.05 + .95 * relative_share))
        importance *= phrase_salience * novelty
        if group == 'background': importance *= 1 - .95 * settings['backgroundSuppression']
        importance = float(np.clip(importance, 0, 1))
        primary = group == frame['primarySource']
        secondary = bool(frame['allowSecondary'] and group == frame['secondarySource'])
        secondary_supported = _secondary_attack_support(event, frame, relative_share, strength, confidence)
        allowed = primary or secondary
        if manual_focus:
            allowed = bool(frame['_focusAllowed'])
            primary = bool(frame['_focusPrimary'])
            secondary = allowed and not primary
        importance_threshold = settings['minImportance'] * (1.2 - .2 * frame['confidence'])
        reason = 'candidate'
        if manual_focus and not allowed: reason = 'manual_source_excluded'
        elif event.get('sustained'): reason = 'sustained_background'
        elif manual_focus and event.get('timingEvidence', 'observed_onset') != 'observed_onset': reason = 'unobserved_source_timing'
        elif frame['state'] == 'silence' or frame['attentionBudget'] <= .01: reason = 'no_musical_information'
        elif not allowed and settings['attentionStrength'] > .20: reason = 'supporting_layer'
        elif importance < importance_threshold: reason = 'low_importance'
        elif secondary and not primary and not secondary_supported: reason = 'weak_secondary_evidence'
        candidate = {**event, 'id': f'event-{index + 1}', 'attentionSource': group, 'importance': round(importance, 5),
                     'confidence': round(confidence, 4), 'sourceShare': round(share, 5),
                     'contextConfidence': frame['confidence'], 'state': frame['state'], 'phraseId': phrase['id'] if phrase else event.get('phraseId'),
                     'selected': False, 'reason': reason,
                     'gridDistance': round(grid_distance, 5) if grid_distance is not None else None,
                     'beatDistance': round(distance, 5) if distance is not None else None,
                     'temporalSalience': round(float(temporal_salience), 4), 'rhythmicSalience': round(float(rhythmic_salience), 4),
                     'admission': {'target': 'primary' if primary else 'secondary' if secondary else 'supporting',
                                   'secondaryPolicy': frame.get('secondaryPolicy', 'confident_secondary' if frame['allowSecondary'] else 'disabled'),
                                   'relativeAttention': round(float(relative_share), 5),
                                   'importanceThreshold': round(float(importance_threshold), 5),
                                   'importancePassed': bool(importance >= importance_threshold),
                                   'observedAttack': event.get('timingEvidence', 'observed_onset') == 'observed_onset' and not bool(event.get('sustained')),
                                   'secondarySupport': bool(secondary_supported) if secondary else None},
                     'timingEvidence': event.get('timingEvidence', 'observed_onset'), '_frame': frame_id}
        if manual_focus:
            candidate['mappingFocus'] = copy.deepcopy(frame['mappingFocus'])
            candidate['_focusDensity'] = float(frame['densityTarget'])
            candidate['_focusConfidence'] = float(frame['confidence'])
        candidates.append(candidate); seen_source[source] = second
    # A single perceptual instant spends the budget once. Kick + bass + piano at
    # the same time are one candidate, with supporting evidence kept for review.
    buckets = []
    for candidate in candidates:
        if (buckets and candidate['time'] - buckets[-1][0]['time'] <= .032
                and (not focus_regions or same_focus_region(focus_regions, candidate['time'], buckets[-1][0]['time']))): buckets[-1].append(candidate)
        else: buckets.append([candidate])
    admitted = []
    for bucket in buckets:
        eligible = sorted([e for e in bucket if e['reason'] == 'candidate'], key=lambda e: e['importance'], reverse=True)
        if not eligible: continue
        winner = eligible[0]; winner['coincidentSources'] = sorted(set(e['source'] for e in bucket))
        admitted.append(winner)
        for other in eligible[1:]: other['reason'] = 'redundant_simultaneous'
    # Sliding information windows prioritize stronger phrase events. This is a
    # candidate ceiling, not map NPS: difficulty/hand constraints remain downstream.
    admitted_times = sorted(e['time'] for e in admitted)
    selected_times = []
    for event in sorted(admitted, key=lambda e: (-e['importance'], e['time'])):
        frame = frames[event['_frame']]
        density = event.get('_focusDensity', frame['densityTarget'])
        raw_capacity = bisect.bisect_right(admitted_times, event['time'] + .50) - bisect.bisect_left(admitted_times, event['time'] - .50)
        uncertainty = .60 + .40 * event.get('_focusConfidence', frame['confidence'])
        keep_fraction = 1 - settings['attentionStrength'] * settings['competition'] * .40
        capacity = max(1, int(math.ceil(min(raw_capacity * keep_fraction * uncertainty,
                                           max(1., density * 16.)))))
        if settings['maxDensity'] <= 0: capacity = 0
        event['admission']['windowCapacity'] = capacity
        nearby = selected_times[bisect.bisect_left(selected_times, event['time'] - .50):bisect.bisect_right(selected_times, event['time'] + .50)]
        if focus_regions:
            nearby = [second for second in nearby if same_focus_region(focus_regions, event['time'], second)]
        event['admission']['nearbySelectedAtDecision'] = len(nearby)
        if len(nearby) >= capacity: event['reason'] = 'attention_budget'
        elif any(abs(second - event['time']) < .035 for second in nearby): event['reason'] = 'redundant_nearby'
        else:
            event['selected'] = True; event['reason'] = 'primary_event' if event['attentionSource'] == frame['primarySource'] else 'secondary_accent'
            bisect.insort(selected_times, event['time'])
    for event in candidates:
        event.pop('_frame', None)
        event.pop('_focusDensity', None)
        event.pop('_focusConfidence', None)
    if focus_regions:
        from .focus import focus_at_time
        # An edited source budget must not spill into neighboring automatic
        # time through the +/-0.5 s information window. Reuse the untouched
        # automatic decision there, without re-running source detection/model
        # inference. Placement still validates both physical boundaries.
        automatic_frames = []
        for frame in frames:
            original = frame.get('_automaticDecision')
            value = {**frame, **original} if original else dict(frame)
            value.pop('_automaticDecision', None)
            value.pop('mappingFocus', None)
            automatic_frames.append(value)
        automatic = {event['id']: event for event in select_events(
            automatic_frames, events, phrases, settings, beat_times)}
        candidates = [candidate if focus_at_time(focus_regions, candidate['time']) else
                      automatic.get(candidate['id'], candidate) for candidate in candidates]
    return candidates
