"""Boundary-aware, reversible proposals for missing map sections.

This module does not write projects or generate arbitrary beat times. It adapts
the existing mapper's audio-conditioned hits to fixed context on both sides.
Existing objects are immutable constraints. Only new placements may change.
"""
from __future__ import annotations
import collections
import copy
import math
import numpy as np
from modding import BeatClock, decode_precision, precision
from .diagnostics import ANGLES, VECTORS, minimum_interval, analyze_playability
from .regions import GOALS, validate_regions, config_at_time

NOTE_KEYS = ('colorNotes', 'sliders', 'burstSliders')
EVENT_KEYS = ('bombNotes', 'obstacles', 'basicBeatmapEvents', 'rotationEvents',
              'colorBoostBeatmapEvents', 'lightColorEventBoxGroups',
              'lightRotationEventBoxGroups', 'lightTranslationEventBoxGroups')


def _clock(chart, analysis, config):
    return BeatClock(config.get('bpm') or analysis.get('bpm', 120), chart.get('bpmEvents', []))


def _time(note, clock, offset):
    return clock.time(note['b']) + offset


def _physical(note):
    custom = note.get('customData', note.get('_customData', {})) or {}
    return not custom.get('fake', custom.get('_fake', False)) and not custom.get('uninteractable', custom.get('_uninteractable', False)) and custom.get('interactable', custom.get('_interactable', True)) is not False


def _xy(note):
    return np.array([decode_precision(note['x']), decode_precision(note['y'])], dtype=float)


def _angle(a, b):
    return abs((ANGLES[int(a)] - ANGLES[int(b)] + 180) % 360 - 180)


def _statistics(notes, seconds, window_start, window_end):
    """Measured local vocabulary, rather than a copied repeating template."""
    previous, travel, intervals, tokens, directions = {}, [], [], [], []
    groups = collections.Counter()
    for n, t in zip(notes, seconds):
        hand = n['c']; groups[round(t, 5)] += 1
        if hand in previous:
            old, old_time = previous[hand]
            travel.append(float(np.linalg.norm(_xy(n) - _xy(old))))
            intervals.append(t - old_time)
        previous[hand] = (n, t)
        tokens.append((n['c'], n['x'], n['y'], n['d']))
        directions.append(n['d'])
    nps = len(notes) / max(.01, window_end - window_start)
    movement = float(np.mean(travel)) if travel else 0.
    angular = sum(d not in (0, 1, 8) for d in directions) / max(1, len(directions))
    doubles = sum(n == 2 for n in groups.values()) / max(1, len(groups))
    feature = np.array([min(1., nps / 12), angular, min(1., movement / 2), doubles])
    inferred = min(GOALS, key=lambda s: float(np.linalg.norm(feature - np.asarray(GOALS[s])))) if notes else None
    return {'noteCount': len(notes), 'nps': round(nps, 4), 'meanGridTravel': round(movement, 4),
            'angularFraction': round(angular, 4), 'doubleFraction': round(doubles, 4),
            'medianSameHandIntervalMs': round(float(np.median(intervals)) * 1000, 2) if intervals else None,
            'inferredStyle': inferred, 'styleMethod': 'Nearest measured density/angles/travel/doubles objective; not a verified style label.',
            'vocabulary': [list(k) + [count] for k, count in collections.Counter(tokens).most_common(16)]}


def _section(analysis, second):
    music = analysis.get('musicAnalysis', {})
    section = next((s for s in music.get('sections', []) if s['start'] <= second < s['end']), None)
    return {k: section.get(k) for k in ('label', 'primaryRole', 'activity', 'motifId')} if section else {}


def plan_continuation(existing, analysis, config, ranges, options=None):
    """Inspect both context windows and produce the mapper's creative config.

    Generate with generationConfig, retaining the full audio/beat clock. Pass
    its beatmap to adapt_continuation. No project is persisted by either call.
    """
    options = dict(options or {})
    operation = options.get('operation', 'fill')
    if operation not in ('fill', 'replace', 'continue'):
        raise ValueError('Choose Fill Empty Region, Replace Selected Region, or Continue From Here.')
    ranges = validate_regions(ranges, analysis['duration'])
    if not ranges:
        raise ValueError('Select at least one region for Let AI Continue.')
    bpm = float(config.get('bpm') or analysis.get('bpm', 120))
    window = float(options.get('contextSeconds', max(4., min(16., 16 * 60 / bpm))))
    if not math.isfinite(window) or not 2 <= window <= 32:
        raise ValueError('Context window must be between 2 and 32 seconds.')
    offset = float(config.get('songTimeOffset', 0)); clock = _clock(existing, analysis, config)
    notes = sorted([n for n in existing.get('colorNotes', []) if _physical(n)], key=lambda n: (n['b'], n['c']))
    times = [_time(n, clock, offset) for n in notes]
    summaries, region_configs = [], []
    for region in ranges:
        start, end = region['start'], region['end']
        a, b = max(0., start - window), min(float(analysis['duration']), end + window)
        before = [(n, t) for n, t in zip(notes, times) if a <= t < start]
        after = [(n, t) for n, t in zip(notes, times) if end <= t < b]
        combined = before + after
        stats = _statistics([n for n, _ in combined], [t for _, t in combined], 0, start - a + b - end)
        before_stats = _statistics([n for n, _ in before], [t for _, t in before], a, start)
        after_stats = _statistics([n for n, _ in after], [t for _, t in after], end, b)
        boundary = {'before': {}, 'after': {}}
        for hand in (0, 1):
            left = [(n, t) for n, t in before if n['c'] == hand]
            right = [(n, t) for n, t in after if n['c'] == hand]
            boundary['before'][str(hand)] = [dict(n, seconds=t) for n, t in left[-12:]]
            boundary['after'][str(hand)] = [dict(n, seconds=t) for n, t in right[:12]]
        local = copy.deepcopy(region)
        if local['primaryStyle'] is None:
            local['primaryStyle'] = stats['inferredStyle'] if options.get('useSurroundingPatterns', True) and stats['noteCount'] >= 6 else config.get('style', 'Modern')
        if options.get('matchNps', True) and local.get('nps') is None and local.get('npsRange') is None and stats['noteCount'] >= 4:
            local['nps'] = max(.25, min(30., stats['nps']))
        if options.get('matchMovement', True) and 'movement' not in local['overrides'] and stats['noteCount'] >= 4:
            local['overrides']['movement'] = max(0., min(100., stats['meanGridTravel'] / 2 * 100))
        # A gap may cross several authored mapping intents. Keep their NPS,
        # style and source directives rather than replacing the whole gap with
        # one inferred objective. Explicit gap controls take precedence.
        surrounding_regions = config.get('mappingRegions', config.get('regions', []))
        edges = sorted(set([start, end] + [float(r[k]) for r in surrounding_regions
            for k in ('start', 'end') if start < float(r[k]) < end]))
        for piece_index, (left, right) in enumerate(zip(edges, edges[1:])):
            inherited = next((r for r in surrounding_regions if r['start'] <= (left + right) / 2 < r['end']), None)
            piece = copy.deepcopy(local)
            if inherited:
                if region.get('primaryStyle') is None and inherited.get('primaryStyle'):
                    piece['primaryStyle'] = inherited['primaryStyle']
                    piece['secondaryStyles'] = copy.deepcopy(inherited.get('secondaryStyles', []))
                piece['overrides'] = dict(inherited.get('overrides', {})) | piece['overrides']
                for field in ('nps', 'npsRange'):
                    if region.get('nps') is None and region.get('npsRange') is None and inherited.get(field) is not None:
                        piece.pop('nps' if field == 'npsRange' else 'npsRange', None)
                        piece[field] = copy.deepcopy(inherited[field])
                if 'analysisMode' not in region and 'analysisMode' in inherited:
                    for field in ('analysisMode', 'sources', 'sourcePolicy'):
                        if field in inherited:
                            piece[field] = copy.deepcopy(inherited[field])
                if 'transitionSeconds' not in region and inherited.get('transitionSeconds') is not None:
                    piece['transitionSeconds'] = min(float(inherited['transitionSeconds']), (right - left) / 2)
            piece.update(start=left, end=right)
            if len(edges) > 2:
                piece['id'] = f"{region['id']}-part-{piece_index + 1}"
            region_configs.append(piece)
        phase_counts = np.ones(16, dtype=float) * .25
        for note, _ in combined:
            phase_counts[int(round(float(note['b']) * 4)) % 16] += 1
        phase_counts /= max(float(phase_counts.max()), 1.)
        effective = config_at_time(dict(config, mappingRegions=[local]), (start + end) / 2)
        effective = {key: effective.get(key) for key in ('style', 'secondaryStyles', 'philosophy', 'normalizedGoals', 'styleGoal', 'nps')}
        summaries.append({'id': region['id'], 'start': start, 'end': end, 'contextStart': a, 'contextEnd': b,
                          'before': before_stats, 'after': after_stats, 'combined': stats, 'boundaries': boundary,
                          'musicBefore': _section(analysis, max(0., start - .15)),
                          'musicInside': _section(analysis, (start + end) / 2), 'musicAfter': _section(analysis, min(analysis['duration'] - .001, end + .15)),
                          'rhythmPhaseWeights': phase_counts.tolist(),
                          'effectiveIntent': effective})
    generation_config = copy.deepcopy(config)
    # Audio focus belongs to the complete song, not only the selected gap.
    # This snapshot also keeps source decisions around both boundaries stable.
    generation_config['_attentionMappingRegions'] = copy.deepcopy(config.get('_attentionMappingRegions', config.get('mappingRegions', config.get('regions', []))))
    for region in ranges:
        if 'analysisMode' not in region:
            continue
        combined = []
        for inherited in generation_config['_attentionMappingRegions']:
            if inherited['end'] <= region['start'] or inherited['start'] >= region['end']:
                combined.append(inherited)
                continue
            if inherited['start'] < region['start']:
                combined.append(dict(copy.deepcopy(inherited), end=region['start'], id=f"{inherited['id']}-before-{region['id']}"[:128]))
            if inherited['end'] > region['end']:
                combined.append(dict(copy.deepcopy(inherited), start=region['end'], id=f"{inherited['id']}-after-{region['id']}"[:128]))
        combined.append(copy.deepcopy(region))
        generation_config['_attentionMappingRegions'] = sorted(combined, key=lambda r: (r['start'], r['end']))
    generation_config['mappingRegions'] = region_configs
    generation_config['continuationContext'] = summaries
    generation_config['continuationMatchRhythm'] = bool(options.get('matchRhythm', True))
    generation_config['continuationRhythmClock'] = {'bpm': bpm, 'events': copy.deepcopy(existing.get('bpmEvents', [])), 'offset': offset}
    return {'operation': operation, 'ranges': ranges, 'contextSeconds': window,
            'options': {'useSurroundingPatterns': bool(options.get('useSurroundingPatterns', True)),
                        'matchNps': bool(options.get('matchNps', True)), 'matchMovement': bool(options.get('matchMovement', True)),
                        'matchRhythm': bool(options.get('matchRhythm', True)), 'includeEvents': bool(options.get('includeEvents', False))},
            'contextSummaries': summaries, 'generationConfig': generation_config}


def rhythm_bias(config, time, beat):
    """Candidate-score multiplier from both context windows; never a new time."""
    if not config.get('continuationMatchRhythm', False):
        return 1.
    summary = next((s for s in config.get('continuationContext', []) if s['start'] <= time < s['end']), None)
    if summary is None:
        return 1.
    native = config.get('continuationRhythmClock')
    if native:
        beat = BeatClock(native['bpm'], native['events']).beat(time - native['offset'])
    return .75 + .5 * summary['rhythmPhaseWeights'][int(round(float(beat) * 4)) % 16]


def _candidate_positions(node, config):
    original = node['note']
    if not node['new']:
        return [dict(original)]
    hand = original['c']; rank = int(config.get('rank', 9))
    directions = [8] if config.get('mode') == 'NoArrows' or original['d'] == 8 else ([0, 1] if rank <= 3 else [0, 1, 4, 5, 6, 7])
    if original['d'] in (2, 3) and rank >= 7:
        directions += [2, 3]
    encoded_x, encoded_y = abs(original['x']) >= 1000, abs(original['y']) >= 1000
    delta_x = decode_precision(original['x']) - round(decode_precision(original['x'])) if encoded_x else 0
    delta_y = decode_precision(original['y']) - round(decode_precision(original['y'])) if encoded_y else 0
    return [dict(original, x=precision(x + delta_x) if encoded_x else x,
                 y=precision(y + delta_y) if encoded_y else y, d=d, a=original.get('a', 0))
            for x in ([0, 1] if hand == 0 else [2, 3]) for y in range(3) for d in directions]


def _transition(left, right, seconds, minimum):
    """Endpoint recovery is separate from note-to-note distance."""
    gap = max(seconds, .00001)
    reach = float(np.linalg.norm(_xy(right) - _xy(left)))
    start = _xy(left) + np.asarray(VECTORS[int(left['d'])]) * .55
    target = _xy(right) - np.asarray(VECTORS[int(right['d'])]) * .55
    recovery = float(np.linalg.norm(target - start)) * .6 / gap
    cost = min(25., reach * .25 / max(.15, gap)) + max(0., recovery - 5) ** 2 * .35
    if seconds < minimum - .0005:
        cost += 1e6 * (minimum - seconds + 1)
    if gap < 1.2 and left['d'] != 8 and right['d'] != 8 and _angle(left['d'], right['d']) < 135:
        cost += 1e5 + (135 - _angle(left['d'], right['d'])) * 20
    if gap < .22 and reach > 1.6:
        cost += 5e4 * (reach - 1.6)
    return cost


def _sweep_cost(note, foreign):
    if not foreign:
        return 0.
    a = _xy(note); vector = np.asarray(VECTORS[int(note['d'])])
    sweep = np.linspace(a - vector * .5, a + vector * .75, 9)
    cost = 0.
    for other in foreign:
        b = _xy(other); v = np.asarray(VECTORS[int(other['d'])])
        other_sweep = np.linspace(b - v * .5, b + v * .75, 9)
        distance = float(np.linalg.norm(sweep[:, None] - other_sweep[None, :], axis=2).min())
        if distance < .28:
            cost += 2e4 * (.28 - distance + .1)
    return cost


def _transition_matrix(left, right, seconds, minimum):
    gap = max(seconds, .00001)
    a = np.array([_xy(n) for n in left]); b = np.array([_xy(n) for n in right])
    av = np.array([VECTORS[int(n['d'])] for n in left]); bv = np.array([VECTORS[int(n['d'])] for n in right])
    reach = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    recovery = np.linalg.norm((a + av * .55)[:, None, :] - (b - bv * .55)[None, :, :], axis=2) * .6 / gap
    costs = np.minimum(25., reach * .25 / max(.15, gap)) + np.maximum(0., recovery - 5) ** 2 * .35
    if seconds < minimum - .0005:
        costs += 1e6 * (minimum - seconds + 1)
    if gap < 1.2:
        directions_a = np.array([n['d'] for n in left]); directions_b = np.array([n['d'] for n in right])
        angles = np.abs((np.asarray(ANGLES)[directions_a, None] - np.asarray(ANGLES)[directions_b][None, :] + 180) % 360 - 180)
        conflict = (angles < 135) & (directions_a[:, None] != 8) & (directions_b[None, :] != 8)
        costs += np.where(conflict, 1e5 + (135 - angles) * 20, 0.)
    if gap < .22:
        costs += np.maximum(0., reach - 1.6) * 5e4
    return costs


def _bridge(nodes, foreign_nodes, config, vocabulary):
    """Dynamic programming considers both fixed endpoints, not greedy parity."""
    if not nodes:
        return
    minimum = minimum_interval(config); options = [_candidate_positions(n, config) for n in nodes]
    preferences = []
    for node, candidates in zip(nodes, options):
        original = node['note']; foreign = [n['note'] for n in foreign_nodes if abs(n['time'] - node['time']) <= .025]
        values = []
        for candidate in candidates:
            distance = float(np.linalg.norm(_xy(candidate) - _xy(original)))
            change = .12 * distance + (0. if candidate['d'] == original['d'] else .25)
            if node['new'] and vocabulary:
                key = (candidate['c'], candidate['x'], candidate['y'], candidate['d'])
                change -= .08 * math.log1p(vocabulary.get(key, 0))
            values.append(change + (_sweep_cost(candidate, foreign) if node['new'] else 0.))
        preferences.append(np.array(values))
    costs, pointers = preferences[0], []
    for index in range(1, len(nodes)):
        gap = nodes[index]['time'] - nodes[index - 1]['time']
        matrix = _transition_matrix(options[index - 1], options[index], gap, minimum)
        choices = costs[:, None] + matrix
        pointer = choices.argmin(axis=0)
        pointers.append(pointer)
        costs = choices[pointer, np.arange(len(pointer))] + preferences[index]
    choice = int(costs.argmin()); chosen = [choice]
    for pointer in reversed(pointers):
        choice = int(pointer[choice]); chosen.append(choice)
    for node, candidates, index in zip(nodes, options, reversed(chosen)):
        if node['new']:
            node['note'].update({k: candidates[index][k] for k in ('x', 'y', 'd', 'a')})


def adapt_continuation(existing, generated, analysis, config, plan):
    """Return a proposal. Only newly generated notes receive geometric edits.

    Fill keeps every existing object, including objects inside selected ranges.
    Replace removes only objects whose heads are selected. Cross-boundary tails
    of existing objects are preserved exactly. New tails are clipped to ranges.
    """
    result = copy.deepcopy(existing)
    clock = _clock(existing, analysis, config); source_clock = _clock(generated, analysis, config)
    offset = float(config.get('songTimeOffset', 0))
    # The mapper produces an absolute-audio-time chart. Its clock does not
    # inherit Info.dat's offset from a different, imported authored chart.
    source_offset = float(config.get('generatedSongTimeOffset', 0))
    ranges = plan['ranges']; operation = plan['operation']
    changes, skipped, nodes, original_placements = [], [], [], {}
    include_events = plan['options'].get('includeEvents', False)
    keys = NOTE_KEYS + (EVENT_KEYS if include_events else ())
    generated_refs = {}; new_sustains = set()
    for key in keys:
        original = result.get(key, [])
        def selected(item):
            second = _time(item, clock, offset)
            return next((r for r in ranges if r['start'] <= second < r['end']), None)
        kept = [n for n in original if not selected(n)] if operation == 'replace' else list(original)
        additions = []
        for item in generated.get(key, []):
            second = _time(item, source_clock, source_offset)
            region = next((r for r in ranges if r['start'] <= second < r['end']), None)
            if not region:
                continue
            if key in ('sliders', 'burstSliders'):
                head = (round(second, 5), item.get('c'), item.get('x'), item.get('y'), item.get('d'))
                if head not in generated_refs:
                    skipped.append({'time': second, 'source': key, 'reason': 'The generated sustain head was not added; authored objects were protected.'})
                    continue
            if key == 'colorNotes' and operation != 'replace':
                # An empty slot is not merely a different square: the same hand
                # still needs recovery from authored notes around that instant.
                minimum = minimum_interval(config_at_time(plan['generationConfig'], second))
                conflict = next((n for n in kept if n['c'] == item['c'] and _physical(n) and abs(_time(n, clock, offset) - second) < minimum - .0005), None)
                if conflict:
                    skipped.append({'time': second, 'source': key, 'reason': 'Authored same-hand note occupies this recovery window.'})
                    continue
            value = copy.deepcopy(item); value['b'] = clock.beat(second - offset)
            if 'tb' in value:
                tail = min(region['end'], source_clock.time(item['tb']) + source_offset)
                value['tb'] = clock.beat(tail - offset)
            if key == 'obstacles':
                tail = min(region['end'], source_clock.time(item['b'] + item.get('d', 0)) + source_offset)
                value['d'] = max(0., clock.beat(tail - offset) - value['b'])
            additions.append(value)
            if key in ('sliders', 'burstSliders'):
                new_sustains.add(id(value))
            if key == 'colorNotes':
                node = {'note': value, 'new': True, 'time': second, 'regionId': region['id']}
                nodes.append(node); original_placements[id(value)] = copy.deepcopy(value)
                generated_refs[(round(second, 5), value['c'], item['x'], item['y'], item['d'])] = value
        # Keeping original order is part of preserving authored content. Native
        # array order is not gameplay time order; processing always sorts copies.
        result[key] = kept + additions
    existing_nodes = [{'note': n, 'new': False, 'time': _time(n, clock, offset)} for n in result.get('colorNotes', []) if _physical(n) and id(n) not in original_placements]
    # The two passes include the opposing hand's latest proposal in sweep cost.
    for region, summary in zip(ranges, plan['contextSummaries']):
        local_config = config_at_time(plan['generationConfig'], (region['start'] + region['end']) / 2)
        vocabulary = {tuple(v[:4]): v[4] for v in summary['combined']['vocabulary']} if plan['options']['useSurroundingPatterns'] else {}
        new_inside = [n for n in nodes if n['regionId'] == region['id']]
        protected = [n for n in existing_nodes if region['start'] <= n['time'] < region['end']]
        for _ in range(2):
            for hand in (0, 1):
                inside = sorted([n for n in new_inside + protected if n['note']['c'] == hand], key=lambda n: n['time'])
                before = [n for n in existing_nodes + nodes if n['note']['c'] == hand and summary['contextStart'] <= n['time'] < region['start']]
                after = [n for n in existing_nodes + nodes if n['note']['c'] == hand and region['end'] <= n['time'] < summary['contextEnd']]
                # Neighboring selected gaps count as context, but this pass must
                # not mutate their placements or their different creative intent.
                left = [dict(max(before, key=lambda n: n['time']), new=False)] if before else []
                right = [dict(min(after, key=lambda n: n['time']), new=False)] if after else []
                chain = left + inside + right
                foreign = [n for n in nodes + existing_nodes if n['note']['c'] != hand and summary['contextStart'] <= n['time'] < summary['contextEnd']]
                _bridge(chain, foreign, local_config, vocabulary)
    for node in nodes:
        before = original_placements[id(node['note'])]
        now = node['note']
        if any(before[k] != now[k] for k in ('x', 'y', 'd', 'a')):
            changes.append({'time': round(node['time'], 6), 'hand': now['c'],
                            'before': {k: before[k] for k in ('x', 'y', 'd')},
                            'after': {k: now[k] for k in ('x', 'y', 'd')},
                            'reason': 'Prepared both fixed boundaries and opposing-hand sweep paths.'})
    # New arcs/chains must still agree with their reranked note endpoints.
    for key in ('sliders', 'burstSliders'):
        for sustain in result.get(key, []):
            if id(sustain) not in new_sustains:
                continue
            second = _time(sustain, clock, offset)
            lookup = (round(second, 5), sustain.get('c'), sustain.get('x'), sustain.get('y'), sustain.get('d'))
            note = generated_refs.get(lookup)
            if note is not None:
                sustain.update({k: note[k] for k in ('x', 'y', 'd')})
                if key == 'sliders' and 'tb' in sustain:
                    tail = clock.time(sustain['tb']) + offset
                    target = next((n['note'] for n in nodes if n['note']['c'] == sustain.get('c') and abs(n['time'] - tail) < .0001), None)
                    if target:
                        sustain.update(tx=target['x'], ty=target['y'], td=target['d'])
    diagnostics = analyze_playability(result, analysis, config)
    relevant = []
    new_times = [n['time'] for n in nodes]
    for issue in diagnostics['issues']:
        # A first authored note after the range can carry an exit-boundary issue.
        inside = any(r['start'] <= issue['time'] < r['end'] + 1.2 for r in ranges)
        if inside and (not new_times or min(abs(issue['time'] - t) for t in new_times) <= 1.2):
            relevant.append(issue)
    outside_unchanged = all([n for n in existing.get(key, []) if not any(r['start'] <= _time(n, clock, offset) < r['end'] for r in ranges)] ==
                            [n for n in result.get(key, []) if not any(r['start'] <= _time(n, clock, offset) < r['end'] for r in ranges)] for key in set(existing) if isinstance(existing.get(key), list) and all(isinstance(n, dict) and 'b' in n for n in existing[key]))
    if not outside_unchanged:
        raise RuntimeError('Continuation changed an object outside its selected region.')
    return {'beatmap': result, 'changes': changes,
            'report': {'operation': operation, 'newNotes': len(nodes), 'adaptedPlacements': len(changes),
                       'skippedCandidates': skipped, 'preservedOutside': True,
                       'preservedExistingInside': operation != 'replace', 'changedHitTimes': 0,
                       'contextSummaries': plan['contextSummaries'], 'boundaryIssues': relevant,
                       'playtest': diagnostics, 'needsReview': bool(relevant),
                       'limitation': 'Fixed authored timing may make a boundary impossible; unresolved diagnostics require review. New-note geometry is estimated, not a headset simulation.'}}
