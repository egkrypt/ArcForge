"""User-directed musical attention, separate from detection and hit timing.

Ranges are half-open audio seconds. Source weights express creative priority,
not instrument confidence. No directive creates a source event or a timestamp.
"""
from __future__ import annotations
import bisect
import collections
import copy
import math
from .settings import SOURCES, COMPETITORS, attention_source


def _number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a number.')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number.') from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be between {low:g} and {high:g}.')
    return value


def _legacy_regions(regions, duration):
    if not isinstance(regions, list) or len(regions) > 128:
        raise ValueError('Use at most 128 source focus regions.')
    result = []
    for index, region in enumerate(regions):
        if not isinstance(region, dict) or region.get('source') not in (*SOURCES, 'auto'):
            raise ValueError('Choose a recognized musical source or Auto.')
        start = _number(region.get('start'), 'Focus start', 0, duration)
        end = _number(region.get('end'), 'Focus end', 0, duration)
        if end <= start:
            raise ValueError('A source focus region must end after it starts.')
        policy = region.get('policy', region.get('sourcePolicy', 'only'))
        if policy not in ('only', 'primary'):
            raise ValueError('Choose Only selected sources or Primary sources.')
        result.append({'id': str(region.get('id') or f'focus-{index + 1}'),
                       'name': str(region.get('name') or 'Musical focus')[:120],
                       'start': start, 'end': end,
                       'analysisMode': 'auto' if region['source'] == 'auto' else 'custom',
                       'sources': {} if region['source'] == 'auto' else {region['source']: 1.},
                       'sourcePolicy': policy,
                       'transitionSeconds': 0.})
    result.sort(key=lambda r: (r['start'], r['end']))
    if any(b['start'] < a['end'] for a, b in zip(result, result[1:])):
        raise ValueError('Source focus regions cannot overlap.')
    return result


def normalize_focus_regions(mapping_regions=None, focus_regions=None, duration=None):
    """Make one attention view from current regions and older saved focus data.

    Explicit current Auto masks an older directive. Style-only regions do not.
    Style/NPS settings are absent from this view, avoiding unnecessary source
    reanalysis when only choreography changes.
    """
    if not mapping_regions and not focus_regions:
        return []
    from ..regions import validate_regions
    duration = _number(duration, 'Song duration', .001, 86400)
    mapped = validate_regions(mapping_regions or [], duration)
    legacy = _legacy_regions(focus_regions or [], duration)
    edges = sorted({r[key] for r in mapped + legacy for key in ('start', 'end')})
    result = []
    for start, end in zip(edges, edges[1:]):
        center = (start + end) / 2
        region = next((r for r in mapped if r['start'] <= center < r['end'] and 'analysisMode' in r), None)
        if region is None:
            region = next((r for r in legacy if r['start'] <= center < r['end']), None)
        if region is None or region.get('analysisMode', 'auto') == 'auto':
            continue
        entry = {key: copy.deepcopy(region.get(key)) for key in
                 ('id', 'name', 'analysisMode', 'sources', 'sourcePolicy')}
        entry.update(start=start, end=end, sources=entry.get('sources') or {},
                     sourcePolicy=entry.get('sourcePolicy') or 'only',
                     transitionSeconds=min(float(region.get('transitionSeconds', 0)), (end - start) / 2))
        if result and result[-1]['end'] == start and all(
                result[-1].get(key) == entry.get(key) for key in entry if key not in ('start', 'end')):
            result[-1]['end'] = end
        else:
            result.append(entry)
    return result


def focus_regions_from_config(config, duration):
    return normalize_focus_regions(config.get('_attentionMappingRegions', config.get('mappingRegions', config.get('regions', []))),
                                   config.get('sourceFocusRegions', []), duration)


def has_manual_focus(config, duration):
    return bool(focus_regions_from_config(config, duration))


def focus_at_time(regions, second):
    return next((r for r in regions if r['start'] <= second < r['end']), None)


def same_focus_region(regions, first, second):
    a, b = focus_at_time(regions, first), focus_at_time(regions, second)
    return a is b or a == b


def source_weight(region, source):
    if region is None or region['analysisMode'] == 'all':
        return 1.
    weights = region['sources']
    return max(float(weights.get(source, 0.)),
               float(weights.get('drums', 0.)) if attention_source(source) == 'drums' else 0.)


def source_allowed(region, source):
    return region is None or region['analysisMode'] == 'all' or region['sourcePolicy'] == 'primary' or source_weight(region, source) > 0


def focus_decision(frame, second, regions, event=None, event_times=None):
    """A view over a detected frame; raw strengths/confidence remain untouched."""
    original = frame.get('_automaticDecision')
    if original:
        frame = {**frame, **original}
        frame.pop('mappingFocus', None)
        frame.pop('_automaticDecision', None)
    region = focus_at_time(regions, second)
    if region is None:
        return frame
    value = dict(frame)
    value['_automaticDecision'] = {key: copy.deepcopy(frame.get(key)) for key in
        ('primarySource', 'secondarySource', 'attention', 'allowSecondary', 'secondaryPolicy',
         'confidence', 'densityTarget', 'attentionBudget', 'selectionMode', 'state')}
    mode = region['analysisMode']
    weights = region['sources']
    positive = {source: weight for source, weight in weights.items() if weight > 0}
    groups = collections.defaultdict(float)
    if mode == 'all':
        groups.update({source: max(.001, float(frame.get('attention', {}).get(source, 0.)))
                       for source in COMPETITORS})
    else:
        for source, weight in positive.items():
            groups[attention_source(source)] = max(groups[attention_source(source)], weight)
        if region['sourcePolicy'] == 'primary':
            for source, share in frame.get('attention', {}).items():
                if source not in groups:
                    groups[source] = max(0., float(share)) * .25
    transition = float(region.get('transitionSeconds', 0))
    blend = min(1., max(0., (second - region['start']) / transition),
                max(0., (region['end'] - second) / transition)) if transition else 1.
    if blend < 1. and mode == 'custom':
        # Blend priority only among allowed sources. Strict allowlists never
        # extend beyond the selected range or leak unselected accompaniment.
        auto = frame.get('attention', {})
        scale = max((float(auto.get(source, 0.)) for source in groups), default=0.)
        for source in groups:
            groups[source] = groups[source] * blend + (float(auto.get(source, 0.)) / max(scale, 1e-9)) * (1. - blend)
        if not any(groups.values()):
            for source, weight in positive.items():
                groups[attention_source(source)] = max(groups[attention_source(source)], weight)
    total = sum(groups.values())
    shares = {source: groups.get(source, 0.) / total if total else 0. for source in COMPETITORS}
    primary = max(groups, key=groups.get) if groups else None
    # Requesting a source does not increase confidence in its identity.
    confidence = max((float(frame.get('sourceConfidence', {}).get(source, 0.)) for source in positive), default=float(frame.get('confidence', .15)))
    if mode == 'all':
        confidence = float(frame.get('confidence', .15))
    rate = None
    if event_times is not None:
        active = []
        for source, times in event_times.items():
            if source_allowed(region, source) and (mode == 'all' or region['sourcePolicy'] == 'only' or source_weight(region, source) > 0):
                low, high = max(region['start'], second - .75), min(region['end'], second + .75)
                active.extend(times[bisect.bisect_left(times, low):bisect.bisect_left(times, high)])
        # Coincident components carry one unit of musical information.
        distinct = []
        for time in sorted(active):
            if not distinct or time - distinct[-1] >= .032:
                distinct.append(time)
        rate = len(distinct) / max(.05, min(region['end'], second + .75) - max(region['start'], second - .75))
    density = float(frame.get('densityTarget', 0.)) if rate is None else min(1., rate / 8.) * (.60 + .40 * confidence)
    activity = float(frame.get('rhythmDensity', 0.)) if rate is None else min(1., rate / 8.)
    budget = float(frame.get('attentionBudget', 0.)) if rate is None else ((.18 + .82 * activity) * (.55 + .45 * confidence) if rate else 0.)
    value.update(primarySource=primary, secondarySource=None, attention=shares,
                 allowSecondary=False, secondaryPolicy='manual_focus', confidence=confidence,
                 densityTarget=density, attentionBudget=budget,
                 selectionMode='manual_full_mix' if mode == 'all' else 'manual_sources',
                 mappingFocus={'regionId': region['id'], 'analysisMode': mode,
                               'sources': dict(weights), 'sourcePolicy': region['sourcePolicy'],
                               'transitionWeight': blend, 'detectorConfidenceUnchanged': True})
    if rate and frame.get('state') == 'silence' and float(frame.get('energy', 0.)) > .003:
        value['state'] = 'manual_focus'
    if event is not None:
        source = event.get('source', 'other')
        weight = source_weight(region, source)
        value['_focusAllowed'] = source_allowed(region, source)
        value['_focusPrimary'] = mode == 'all' or weight > 0
        requested = weight / max(positive.values(), default=1.) if weight else .25
        automatic_scale = max((float(frame.get('attention', {}).get(attention_source(s), 0.))
                               for s in positive), default=0.)
        automatic = float(frame.get('attention', {}).get(attention_source(source), 0.)) / max(automatic_scale, 1e-9)
        value['_focusRelativeWeight'] = requested * blend + min(1., automatic) * (1. - blend) if mode == 'custom' else 1.
    return value


def focus_event_times(events):
    times = collections.defaultdict(list)
    for event in events:
        if not event.get('sustained') and event.get('timingEvidence', 'observed_onset') == 'observed_onset':
            times[event.get('source', 'other')].append(float(event['time']))
    return {source: sorted(values) for source, values in times.items()}


def focused_frames(frames, events, regions):
    if not regions:
        return frames
    times = focus_event_times(events)
    return [focus_decision(frame, float(frame['time']), regions, event_times=times) for frame in frames]


def focus_summary(regions, events):
    rows = []
    for region in regions:
        local = [event for event in events if region['start'] <= event['time'] < region['end']]
        observed = [event for event in local if (region['analysisMode'] == 'all' or source_weight(region, event.get('source', 'other')) > 0)
                    and not event.get('sustained') and event.get('timingEvidence', 'observed_onset') == 'observed_onset']
        chosen = [event for event in local if event.get('selected')]
        eligible = [event for event in local if event.get('selected') or event.get('reason') == 'attention_budget']
        warnings = []
        if not observed:
            warnings.append('No observed attacks were detected for the requested sources. No timings were invented.')
        elif not eligible:
            warnings.append('Observed source attacks did not pass the importance/confidence policy. Lowering source confidence is not automatic.')
        rows.append({**copy.deepcopy(region), 'observedAttacks': len(observed), 'eligibleEvents': len(eligible),
                     'selectedEvents': len(chosen), 'selectedSourceCounts': dict(collections.Counter(e['source'] for e in chosen)),
                     'meanDetectedConfidence': sum(float(e.get('confidence', 0.)) for e in observed) / max(1, len(observed)),
                     'warnings': warnings})
    return {'regions': rows, 'timingPolicy': 'Observed attacks only; source instructions do not create or relabel musical events.',
            'outsideRegions': 'Automatic musical attention', 'sourceWeightsAreConfidence': False}
