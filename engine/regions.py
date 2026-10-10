"""Creative intent for time ranges, independent of musical hit detection.

All public goals use percentages (0..100). The mapper-facing normalizedGoals
uses fractions (0..1). A region inherits every goal it does not override.
"""
from __future__ import annotations
import copy
import math
from .styles import GOALS

DEFAULT_GOALS = {'musicality': 85., 'tech': 55., 'movement': 55., 'variety': 80.,
                 'contrast': 90., 'stamina': 65.}
ALIASES = {'Tech / Acc': 'Tech/Acc', 'Tech / AccStream': 'Tech/Acc',
           'Light / Easy': 'Light/easy', 'Light / easy': 'Light/easy',
           'Light/Easy': 'Light/easy', 'Light': 'Light/easy',
           'Saber Control': 'Saber control', 'Acc-Focused': 'Acc-focused'}


def _number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a number.')
    try:
        value = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{name} must be a number.') from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be between {low:g} and {high:g}.')
    return value


def canonical_style(style):
    name = ALIASES.get(str(style), str(style))
    if name not in GOALS:
        raise ValueError(f'Unknown mapping philosophy: {name}.')
    return name


def normalize_goals(goals=None, defaults=None):
    """Return validated percentages. Unknown keys are rejected, not ignored."""
    goals = goals or {}
    if not isinstance(goals, dict):
        raise ValueError('Mapping goals must be an object.')
    unknown = set(goals) - set(DEFAULT_GOALS)
    if unknown:
        raise ValueError('Unknown mapping goals: ' + ', '.join(sorted(unknown)))
    values = dict(DEFAULT_GOALS if defaults is None else defaults)
    for key, value in goals.items():
        values[key] = _number(value, key, 0, 100)
    return values


def validate_regions(regions, duration):
    """Canonicalize nonoverlapping half-open [start,end) audio-second ranges."""
    duration = _number(duration, 'Song duration', .001, 86400)
    if not isinstance(regions, list):
        raise ValueError('Mapping regions must be a list.')
    if len(regions) > 128:
        raise ValueError('Use at most 128 mapping regions.')
    result, ids = [], set()
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError('Each mapping region must be an object.')
        start = _number(region.get('start'), 'Region start', 0, duration)
        end = _number(region.get('end'), 'Region end', 0, duration)
        if end <= start:
            raise ValueError('A mapping region must end after it starts.')
        key = str(region.get('id') or f'region-{index + 1}')
        if len(key) > 128 or key in ids:
            raise ValueError('Mapping region IDs must be unique and at most 128 characters.')
        ids.add(key)
        secondary = region.get('secondaryStyles', [])
        if not isinstance(secondary, list) or len(secondary) > 2:
            raise ValueError('Choose at most two secondary philosophies.')
        primary = region.get('primaryStyle', region.get('style'))
        primary = canonical_style(primary) if primary and primary != 'Inherit' else None
        secondary = [canonical_style(s) for s in secondary]
        if len(set(secondary)) != len(secondary) or primary in secondary:
            raise ValueError('A philosophy must not be selected twice in one region.')
        overrides = normalize_goals(region.get('overrides'), defaults={})
        value = {'id': key, 'name': str(region.get('name') or f'Region {index + 1}')[:120],
                 'start': start, 'end': end, 'primaryStyle': primary,
                 'secondaryStyles': secondary, 'overrides': overrides}
        if region.get('nps') is not None:
            value['nps'] = _number(region['nps'], 'Region NPS', .25, 30)
        if region.get('npsRange') is not None:
            span = region['npsRange']
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                raise ValueError('Region NPS range needs a minimum and maximum.')
            low, high = [_number(n, 'Region NPS range', .25, 30) for n in span]
            if low > high:
                raise ValueError('Region minimum NPS cannot exceed its maximum.')
            if region.get('nps') is not None:
                raise ValueError('Choose a region NPS target or an NPS range, not both.')
            value['npsRange'] = [low, high]
        if 'analysisMode' in region:
            mode = region['analysisMode']
            if mode not in ('auto', 'all', 'custom'):
                raise ValueError('Choose Auto, Full mix or Custom for region analysis.')
            value['analysisMode'] = mode
            weights = region.get('sources', {})
            if not isinstance(weights, dict):
                raise ValueError('Region source weights must be an object.')
            from .arcaudio.settings import SOURCES
            if set(weights) - set(SOURCES):
                raise ValueError('Choose recognized musical sources for a region.')
            weights = {source: _number(weight, 'Source weight', 0, 1) for source, weight in weights.items()}
            policy = region.get('sourcePolicy', 'only')
            if policy not in ('only', 'primary'):
                raise ValueError('Choose Only selected sources or Primary sources.')
            if mode == 'custom':
                if not any(weights.values()):
                    raise ValueError('Choose at least one source with a positive weight for Custom analysis.')
                value.update(sources=weights, sourcePolicy=policy)
        if region.get('transitionSeconds') is not None:
            transition = _number(region['transitionSeconds'], 'Region transition', 0, 8)
            value['transitionSeconds'] = min(transition, (end - start) / 2)
        result.append(value)
    result.sort(key=lambda r: (r['start'], r['end']))
    for before, after in zip(result, result[1:]):
        if after['start'] < before['end']:
            raise ValueError('Mapping regions cannot overlap. Resize or split the overlapping regions.')
    return result


def style_weights(primary, secondary=()):
    """One blended objective: secondary traits never become separate generators."""
    primary = canonical_style(primary)
    secondary = [canonical_style(s) for s in secondary if s != primary]
    if not secondary:
        return [(primary, 1.)]
    secondary = list(dict.fromkeys(secondary))[:2]
    main = .8 if len(secondary) == 1 else .7
    return [(primary, main)] + [(s, (1 - main) / len(secondary)) for s in secondary]


def requested_nps(config, activity=None):
    """An activity-responsive goal, never a guarantee or a source of hit times."""
    span = config.get('npsRange')
    if span is None:
        return float(config.get('nps', 10))
    activity = .5 if activity is None else max(0., min(1., float(activity)))
    return float(span[0]) + (float(span[1]) - float(span[0])) * activity


def _region_objective(config, region, activity):
    primary = canonical_style(region.get('primaryStyle') or config.get('style', 'Modern')) if region else canonical_style(config.get('style', 'Modern'))
    secondary = region.get('secondaryStyles', []) if region else config.get('secondaryStyles', [])
    goals = normalize_goals(config.get('philosophy'))
    if region:
        goals = normalize_goals(region.get('overrides'), defaults=goals)
    density = dict(config)
    if region and region.get('nps') is not None:
        density['nps'] = region['nps']
        density.pop('npsRange', None)
    elif region and region.get('npsRange') is not None:
        density['npsRange'] = region['npsRange']
    return {'style': primary, 'weights': dict(style_weights(primary, secondary)),
            'goals': goals, 'nps': requested_nps(density, activity)}


def _mix_objective(first, second, weight):
    styles = set(first['weights']) | set(second['weights'])
    return {'style': second['style'],
            'weights': {style: first['weights'].get(style, 0.) * (1. - weight) + second['weights'].get(style, 0.) * weight
                        for style in sorted(styles)},
            'goals': {key: first['goals'][key] * (1. - weight) + second['goals'][key] * weight for key in first['goals']},
            'nps': first['nps'] * (1. - weight) + second['nps'] * weight}


def region_boundaries(regions, duration):
    """Include fade edges so a four-beat quota cannot skip a short transition."""
    boundaries = {0., float(duration)}
    for region in regions:
        start, end = float(region['start']), float(region['end'])
        boundaries.update((start, end))
        fade = min(float(region.get('transitionSeconds', 0)), (end - start) / 2)
        if fade > 0:
            boundaries.update((start + fade / 2, start + fade, end - fade, end - fade / 2))
    return sorted(boundaries)


def config_at_time(config, time, activity=None):
    """Apply creative direction without changing a hit's musical timestamp."""
    # Generation-local caches can contain hundreds of learned phrase profiles.
    # They are read-only here and share references; public creative settings
    # still receive a private copy so region application cannot mutate them.
    value = copy.deepcopy({k: v for k, v in config.items() if not str(k).startswith('_')})
    value.update({k: v for k, v in config.items() if str(k).startswith('_')})
    regions = config.get('mappingRegions', config.get('regions', []))
    region = next((r for r in regions
                   if float(r['start']) <= time < float(r['end'])), None)
    objective = _region_objective(config, region, activity)
    primary = objective['style']
    blend = 1.
    if region:
        if region.get('nps') is not None:
            value['nps'] = _number(region['nps'], 'Region NPS', .25, 30)
            value.pop('npsRange', None)
        elif region.get('npsRange') is not None:
            value['npsRange'] = list(region['npsRange'])
            value['nps'] = requested_nps(value, activity)
        transition = float(region.get('transitionSeconds', 0))
        if transition > 0:
            left = max(0., (time - region['start']) / transition)
            right = max(0., (region['end'] - time) / transition)
            blend = min(1., left, right)
            if blend < 1.:
                neighbor = next((r for r in regions if r is not region and
                    (r['end'] == region['start'] if left <= right else r['start'] == region['end'])), None)
                anchor = _region_objective(config, neighbor, activity)
                if neighbor and float(neighbor.get('transitionSeconds', 0)) > 0:
                    # Both fades meet at the same midpoint. A hard neighbor
                    # instead owns the edge, avoiding a dip to global defaults.
                    anchor = _mix_objective(anchor, objective, .5)
                objective = _mix_objective(anchor, objective, blend)
        value['regionTransitionWeight'] = blend
        value['regionAnalysisMode'] = region.get('analysisMode', 'auto')
        value['mappingRegionId'] = region.get('id')
    else:
        value.pop('mappingRegionId', None)
    weights = [(style, weight) for style, weight in objective['weights'].items() if weight > 1e-12]
    global_goals = objective['goals']
    # config_at_time has resolved the range at this activity, including a
    # transition. Downstream targets use this value rather than resolving twice.
    value['_resolvedNps'] = objective['nps']
    value['nps'] = objective['nps']
    value.update(style=primary, secondaryStyles=[s for s, _ in weights if s != primary],
                 philosophy=global_goals,
                 normalizedGoals={k: v / 100 for k, v in global_goals.items()},
                 styleGoal=[sum(GOALS[s][axis] * w for s, w in weights) for axis in range(4)],
                 styleWeights=[{'style': s, 'weight': w} for s, w in weights])
    return value


def blended_style_profile(config):
    """Blend learned human-chart archetypes once, using the same style weights."""
    import numpy as np
    from .styles import style_profile
    weights = ([(entry['style'], entry['weight']) for entry in config['styleWeights']]
               if config.get('styleWeights') else
               style_weights(config.get('style', 'Modern'), config.get('secondaryStyles', [])))
    profiles = [(style_profile(style, config), weight, style) for style, weight in weights]
    matches = []
    for profile, weight, style in profiles:
        matches += [dict(m, philosophy=style, blendWeight=weight) for m in profile['matches']]
    return {'prior': sum(p['prior'] * w for p, w, _ in profiles),
            'goal': sum(np.asarray(p['goal']) * w for p, w, _ in profiles),
            'matches': matches[:10],
            'method': 'One weighted objective over learned human-chart archetypes; secondary philosophies contribute traits.'}
