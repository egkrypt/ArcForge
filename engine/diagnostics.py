"""Inspectable geometric playtest checks and hit-preserving repairs.

These checks model timing, swings, recovery and visibility. They do not simulate
tracking hardware, a person's wrists, or the complete Beat Saber physics engine.
"""
from __future__ import annotations
import bisect, collections, copy, math
import numpy as np
from modding import BeatClock

ANGLES = [90, 270, 180, 0, 135, 45, 225, 315, 0]
VECTORS = [(0, 1), (0, -1), (-1, 0), (1, 0), (-.707, .707), (.707, .707), (-.707, -.707), (.707, -.707), (0, 0)]


def _physical(note):
    cd = note.get('customData', note.get('_customData', {})) or {}
    return not cd.get('fake', cd.get('_fake', False)) and not cd.get('uninteractable',cd.get('_uninteractable',False)) and cd.get('interactable', cd.get('_interactable', True)) is not False


def _angle(a, b): return abs((ANGLES[a] - ANGLES[b] + 180) % 360 - 180)


def _xy(note):
    from modding import decode_precision
    return float(decode_precision(note['x'])), float(decode_precision(note['y']))


def _segment_distance(a, b, c, d):
    # Sample the sweep segments, which is enough to catch close opposing center
    # cuts without claiming a complete 3D saber/arm collision simulation.
    first = np.linspace(a, b, 9); second = np.linspace(c, d, 9)
    return float(np.linalg.norm(first[:, None] - second[None], axis=2).min())


def minimum_interval(config):
    if config.get('minimumHandInterval') is not None: return float(config['minimumHandInterval'])
    rank = int(config.get('rank', 9)); return .12 if rank >= 9 else .16 if rank >= 7 else .22 if rank >= 5 else .30


def analyze_playability(chart, analysis, config=None):
    config = config or {}; bpm = float(config.get('bpm') or analysis.get('bpm', 120))
    clock = BeatClock(bpm, chart.get('bpmEvents', [])); minimum = minimum_interval(config)
    def hit_time(beat):return clock.time(beat)+float(config.get('songTimeOffset',0))
    duration = float(analysis.get('duration', 1e9)); mode = config.get('mode', 'Standard')
    indexed = sorted([(i, n) for i, n in enumerate(chart.get('colorNotes', [])) if _physical(n)], key=lambda p: (p[1]['b'], p[1]['c']))
    issues, previous, hand_intervals, motion_speeds, recovery_speeds = [], {}, [], [], []
    def issue(code, severity, index, second, description, related=None, metrics=None):
        item = {'code': code, 'severity': severity, 'noteIndex': index, 'time': round(second, 5),
                'description': description, 'relatedNoteIndex': related, 'metrics': metrics or {}}
        if len(issues) < 1000: issues.append(item)
    groups = collections.defaultdict(list)
    for index, note in indexed:
        second = hit_time(note['b']); hand = note['c']; x, y = _xy(note)
        groups[round(second, 4)].append((index, note))
        if not 0 <= second <= duration: issue('audioBounds', 'error', index, second, 'Note is outside the audio timeline.')
        if hand not in [0, 1] or not 0 <= note['d'] <= 8: issue('invalidNote', 'error', index, second, 'Hand or cut direction is invalid.'); continue
        if mode == 'OneSaber' and hand != 1: issue('oneSaberHand', 'error', index, second, 'One Saber note uses the other hand.')
        if hand in previous:
            last_index, last = previous[hand]; gap = second - hit_time(last['b'])
            hand_intervals.append(gap)
            if gap < minimum - .0005: issue('handTiming', 'error', index, second, 'The same hand has insufficient recovery time.', last_index, {'intervalMs': round(gap * 1000, 2), 'guardMs': round(minimum * 1000)})
            if gap <= 1e-6: issue('sameHandStack', 'error', index, second, 'Two physical notes need the same hand at the same time.', last_index)
            elif gap < 1.2:
                lx, ly = _xy(last); reach = math.hypot(x - lx, y - ly)
                speed = reach * .6 / gap; motion_speeds.append(speed)
                if gap < .22 and reach > 1.6: issue('reach', 'warning', index, second, 'Rapid swing also makes a large lateral or vertical reach.', last_index, {'gridDistance': round(reach, 3), 'metersPerSecond': round(speed, 2)})
                a = np.array([lx, ly]) + np.array(VECTORS[last['d']]) * .55
                z = np.array([x, y]) - np.array(VECTORS[note['d']]) * .55
                recovery = float(np.linalg.norm(z - a)) * .6 / gap; recovery_speeds.append(recovery)
                if recovery > 7.5 and gap < .28: issue('recoveryReach', 'warning', index, second, 'The cut endpoints require a sharp recovery movement.', last_index, {'estimatedMetersPerSecond': round(recovery, 2)})
            if last['d'] != 8 and note['d'] != 8 and gap < 1.2 and _angle(last['d'], note['d']) < 120:
                issue('parity', 'error', index, second, 'Consecutive arrows ask for a reset or the same swing direction.', last_index, {'directionChangeDegrees': _angle(last['d'], note['d'])})
        previous[hand] = (index, note)
    for second, group in groups.items():
        reds = [p for p in group if p[1]['c'] == 0]; blues = [p for p in group if p[1]['c'] == 1]
        for ri, red in reds:
            for bi, blue in blues:
                r, b = np.array(_xy(red)), np.array(_xy(blue))
                rv, bv = np.array(VECTORS[red['d']]), np.array(VECTORS[blue['d']])
                distance = _segment_distance(r - rv * .5, r + rv * .75, b - bv * .5, b + bv * .75)
                if distance < .28: issue('saberClash', 'warning', ri, second, 'Simultaneous sweep paths come close in the central lanes.', bi, {'minimumGridDistance': round(distance, 3)})
    for pos, (index, note) in enumerate(indexed):
        second = hit_time(note['b']); x, y = _xy(note)
        for other_index, other in indexed[pos + 1:pos + 7]:
            delta = hit_time(other['b']) - second
            if delta > .09: break
            if delta > .025 and _xy(other) == (x, y): issue('noteOcclusion', 'warning', other_index, hit_time(other['b']), 'A nearby note approaches behind another note in the same lane/layer.', index, {'separationMs': round(delta * 1000, 2)})
        for obstacle_index, obstacle in enumerate(chart.get('obstacles', [])):
            if not _physical(obstacle): continue
            start, end = hit_time(obstacle['b']), hit_time(obstacle['b'] + obstacle['d'])
            if start <= second <= end and obstacle.get('x', 0) <= x < obstacle.get('x', 0) + obstacle.get('w', 1) and obstacle.get('y', 0) <= y < obstacle.get('y', 0) + obstacle.get('h', 5):
                issue('wallOcclusion', 'error', index, second, 'A scoring note is inside an interactive obstacle.', metrics={'obstacleIndex': obstacle_index})
        for bomb_index, bomb in enumerate(chart.get('bombNotes', [])):
            delta = abs(hit_time(bomb['b']) - second)
            if delta < .11 and math.hypot(bomb['x'] - x, bomb['y'] - y) < 1.05:
                issue('bombClearance', 'warning', index, second, 'A bomb is close to this swing path.', metrics={'bombIndex': bomb_index, 'separationMs': round(delta * 1000, 2)})
    times = np.array([hit_time(n['b']) for _, n in indexed])
    def peak(window):
        return float(np.max(np.arange(len(times)) + 1 - np.searchsorted(times, times - window, side='right'))) / window if len(times) else 0.
    njs = float(config.get('njs') or 20); distance = float(config.get('jumpDistance') or 24)
    reading_time = distance / (2 * njs)
    if reading_time < .38 and peak(4) > 8:
        issue('readingTime', 'warning', None, 0, 'The selected jump settings give a short reading window for this density.', metrics={'readingTimeMs': round(reading_time * 1000), 'peakFourSecondNps': round(peak(4), 2)})
    counts = dict(collections.Counter(i['code'] for i in issues)); errors = sum(i['severity'] == 'error' for i in issues)
    return {'passed': errors == 0, 'errors': errors, 'warnings': len(issues) - errors, 'issues': issues,
            'counts': counts, 'noteCount': len(indexed), 'averageNps': round(len(indexed) / max(duration, 1), 3),
            'peakOneSecondNps': round(peak(1), 2), 'peakFourSecondNps': round(peak(4), 2),
            'minimumHandIntervalMs': round(min(hand_intervals) * 1000, 2) if hand_intervals else None,
            'p95MotionMetersPerSecond': round(float(np.percentile(motion_speeds, 95)), 2) if motion_speeds else 0,
            'p95RecoveryMetersPerSecond': round(float(np.percentile(recovery_speeds, 95)), 2) if recovery_speeds else 0,
            'readingTimeMs': round(reading_time * 1000),
            'model': 'Native-grid timing, cut endpoints, sampled sweep paths, obstacles and visibility',
            'limitation': 'A geometric preview, not a headset test or a ranking certification.'}


def _opposite(direction):
    return {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5, 8: 8}[direction]


def repair_playability(chart, analysis, config=None):
    """Preserve every note's hit time/count, correcting geometry where possible."""
    config = config or {}; result = copy.deepcopy(chart); notes = result.get('colorNotes', [])
    original = copy.deepcopy(notes); clock = BeatClock(config.get('bpm') or analysis['bpm'], chart.get('bpmEvents', []))
    minimum = minimum_interval(config); changes = []; mode = config.get('mode', 'Standard')
    previous = {}
    ordered = sorted(range(len(notes)), key=lambda i: (notes[i]['b'], notes[i]['c']))
    for index in ordered:
        note = notes[index]
        if not _physical(note): continue
        hand = note['c']; second = clock.time(note['b'])
        if mode == 'OneSaber' and hand != 1:
            note['c'] = hand = 1; note['x'] = min(3, max(2, note['x'])); changes.append({'noteIndex': index, 'time': round(second, 5), 'action': 'Use the One Saber hand'})
        if hand in previous:
            previous_index = previous[hand]; last = notes[previous_index]
            gap = second - clock.time(last['b'])
            if gap < minimum and mode != 'OneSaber':
                other = 1 - hand; other_last = notes[previous[other]] if other in previous else None
                future = [notes[j] for j in ordered if clock.time(notes[j]['b']) > second + 1e-5 and notes[j]['c'] == other]
                free_before = other_last is None or second - clock.time(other_last['b']) >= minimum
                free_after = not future or clock.time(future[0]['b']) - second >= minimum
                if free_before and free_after:
                    hand = other; note['c'] = hand; note['x'] = min(1, note['x']) if hand == 0 else max(2, note['x'])
                    changes.append({'noteIndex': index, 'time': round(second, 5), 'action': 'Transfer a cramped hit to the available hand'})
                    last = notes[previous[hand]] if hand in previous else None; gap = second - clock.time(last['b']) if last else 10.
            if last:
                if last['d'] != 8 and note['d'] != 8 and gap < 1.2 and _angle(last['d'], note['d']) < 120:
                    note['d'] = _opposite(last['d']); changes.append({'noteIndex': index, 'time': round(second, 5), 'action': 'Restore alternating cut direction'})
                if gap < .22 and math.hypot(note['x'] - last['x'], note['y'] - last['y']) > 1.6:
                    note['x'] = last['x']; note['y'] = min(1, note['y']); changes.append({'noteIndex': index, 'time': round(second, 5), 'action': 'Reduce the reach between rapid swings'})
        previous[hand] = index
    groups = collections.defaultdict(list)
    for i, note in enumerate(notes): groups[round(note['b'], 5)].append(i)
    for group in groups.values():
        if len({notes[i]['c'] for i in group}) < 2: continue
        for index in group:
            note = notes[index]
            if note['d'] in [4, 5, 6, 7] and note['x'] in [1, 2]:
                note['x'] = 0 if note['c'] == 0 else 3
                changes.append({'noteIndex': index, 'time': round(clock.time(note['b']), 5), 'action': 'Separate a central double sweep'})
    # Keep interactive objects only when all scoring hits stay clear. Decorative
    # Noodle scenery is untouched. Removing an obstacle never removes a hit.
    retained = []
    for index, obstacle in enumerate(result.get('obstacles', [])):
        conflict = _physical(obstacle) and any(obstacle['b'] <= n['b'] <= obstacle['b'] + obstacle['d'] and obstacle['x'] <= n['x'] < obstacle['x'] + obstacle['w'] and obstacle.get('y', 0) <= n['y'] < obstacle.get('y', 0) + obstacle.get('h', 5) for n in notes)
        if conflict: changes.append({'obstacleIndex': index, 'time': round(clock.time(obstacle['b']), 5), 'action': 'Remove an obstacle that contains a scoring hit'})
        else: retained.append(obstacle)
    result['obstacles'] = retained
    # Reconnect arcs/chains to edited heads/tails when their hands still agree.
    lookup = {(round(n['b'], 5), n['c'], n['x'], n['y']): i for i, n in enumerate(original)}
    for name in ['sliders', 'burstSliders']:
        kept = []
        for slider in result.get(name, []):
            head_index = lookup.get((round(slider['b'], 5), slider['c'], slider['x'], slider['y']))
            tail_index = lookup.get((round(slider['tb'], 5), slider['c'], slider['tx'], slider['ty'])) if name == 'sliders' else None
            if head_index is not None:
                head = notes[head_index]
                if tail_index is not None and head['c'] != notes[tail_index]['c']:
                    changes.append({'time': round(clock.time(slider['b']), 5), 'action': 'Remove an arc whose edited endpoints use different hands'}); continue
                slider.update(c=head['c'], x=head['x'], y=head['y'], d=head['d'])
                if tail_index is not None:
                    tail = notes[tail_index]; slider.update(tx=tail['x'], ty=tail['y'], tc=tail['d'])
            kept.append(slider)
        result[name] = kept
    assert len(notes) == len(original) and [n['b'] for n in notes] == [n['b'] for n in original]
    report = analyze_playability(result, analysis, config)
    return {'beatmap': result, 'report': report, 'changes': changes, 'preservedHitCount': len(notes), 'removedHits': 0, 'changedHitTimes': 0}


def recommend_jump_settings(report, config=None):
    config = config or {}
    peak = float(report.get('peakFourSecondNps', report.get('averageNps', 4)))
    movement = float(report.get('p95RecoveryMetersPerSecond', 0))
    njs = round(min(27., max(12., 13.5 + peak * .7)), 1)
    reaction = min(.85, max(.52, .54 + .008 * peak + .009 * movement))
    if config.get('profile'):reaction=min(.95,reaction*(1+(100-float(config['profile'].get('speed',85)))*.001))
    return {'njs': njs, 'jumpDistance': round(njs * 2 * reaction, 2), 'readingTimeMs': round(reaction * 1000),
            'basis': 'Measured four-second density and recovery distance; review with the selected game environment.'}
