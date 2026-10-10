"""Audio-conditioned neural generation with explicit physical and novelty guards."""
from __future__ import annotations
import bisect, collections, hashlib, json, pathlib, functools
import numpy as np
from .features import placement_features, sample_context, untoken, audio_context
from .trainer import _softmax

ANGLES = [90, 270, 180, 0, 135, 45, 225, 315, 0]


def _angle_difference(a, b):
    return abs((ANGLES[a] - ANGLES[b] + 180) % 360 - 180)


@functools.lru_cache(maxsize=4)
def _audio_context_cached(path,modified):
    return audio_context(path)

def _contexts(analysis):
    path=pathlib.Path(analysis.get('audioPath',''))
    if path.is_file():
        # Training and inference must use exactly the same feature extraction.
        return _audio_context_cached(str(path),path.stat().st_mtime_ns)
    frames = analysis.get('frames', [])
    if not frames:
        return np.array([0., float(analysis['duration'])]), np.ones((2, 4), np.float32) * .5
    times = np.array([f['time'] for f in frames], np.float32)
    values = np.array([[f.get(k, 0) for k in ['energy', 'bass', 'onset', 'high']] for f in frames], np.float32)
    values /= np.maximum(np.quantile(values, .95, axis=0), 1e-5)
    return times, np.clip(values, 0, 3)


def _candidate_times_legacy(analysis, config, rng, model):
    bpm = float(config.get('bpm') or analysis['bpm']); spb = 60 / bpm
    duration = float(analysis['duration']); offset = float(analysis.get('offset', 0))
    nps = max(1., min(float(config.get('nps', 10)), 15.))
    # A fast expert stream uses alternating hands, not sub-80 ms same-hand bursts.
    subdivision = .25 if spb >= .29 else .5
    interval = subdivision * spb
    minimum = float(config.get('minimumHandInterval', .145))
    mode = config.get('mode', 'Standard')
    if mode == 'OneSaber': interval = max(interval, minimum)
    frame_times, contexts = _contexts(analysis)
    beat_times = analysis.get('beatTimes') or []
    expected_rate = 1 / interval
    target = min(nps, expected_rate)
    candidates = []
    count = int((duration - offset - .4) / (subdivision * spb))
    energy_reference = max(float(np.mean(contexts[:, 0])), .2)
    rhythm = model['rhythm']
    previous_energy = None
    for index in range(max(0, count)):
        beat = index * subdivision
        if beat_times and len(beat_times) > beat + 1:
            second = float(np.interp(beat, np.arange(len(beat_times)), beat_times))
        else: second = offset + beat * spb
        if second < .5 or second > duration - .35: continue
        context = sample_context(frame_times, contexts, second)
        energy = float(context[0]); onset = min(float(context[2]), 1.5)
        musical = .5 + .65 * energy / energy_reference + .18 * onset
        if energy < .05: musical *= .12
        phase = int(round(beat * 4)) % 16
        human_rhythm = float(rhythm[min(2, int(energy * 2)), phase].mean())
        probability = np.clip(target / expected_rate * musical * (.65 + .35 * human_rhythm), .02, .995)
        # Phrasing is tied to actual energy movement rather than repeating a fixed pattern.
        if previous_energy is not None and energy < previous_energy * .25: probability *= .6
        if rng.random() < probability:
            double_probability = float(model.get('doubles', np.zeros((3, 16)))[min(2, int(energy * 2)), phase])
            chord = mode != 'OneSaber' and phase % 4 == 0 and context[1] > .35 and rng.random() < min(.42, double_probability)
            if chord:
                # Clear the short lead-in to a bass accent; two hands then arrive
                # together with a full recovery interval instead of a dense collision.
                while candidates and second - candidates[-1]['time'] < minimum:
                    candidates.pop()
            candidates.append({'time': second, 'beat': second / spb, 'musicalBeat': beat, 'context': context, 'chord': chord})
        previous_energy = energy
    return candidates, minimum, target


def _mapping_clock(analysis, config):
    from modding import BeatClock
    bpm = float(config.get('bpm') or analysis['bpm']); offset = float(analysis.get('offset', 0))
    beat_times = analysis.get('beatTimes') or []
    events = []
    if len(beat_times) > 2:
        # Encode the measured local periods so notes, lighting and editor time all
        # use one clock. Offset is included as the initial empty beat interval.
        periods = np.diff(beat_times); shift = float(beat_times[0]) * bpm / 60
        # A fixed 32-beat mean smears sudden tempo changes and displaces every
        # subdivision around the boundary. Compress only genuinely equivalent
        # adjacent periods, with a bounded cumulative timing error.
        start = 0
        for index in range(1, len(periods) + 1):
            current = float(periods[start])
            close = index < len(periods) and abs(float(periods[index]) - current) <= max(.00001, current * .005)
            cumulative_error = abs(float(np.sum(periods[start:index])) - current * (index - start))
            if close and cumulative_error <= .002: continue
            period = float(np.mean(periods[start:index]))
            if period > .02: events.append({'b': round(shift + start, 6), 'm': round(60 / period, 6)})
            start = index
    return BeatClock(bpm, events), events


def _candidate_times(analysis, config, rng, model):
    """Keep the proven pipeline intact unless attention is explicitly enabled."""
    from .arcaudio.settings import canonical_mode
    mode = canonical_mode(config.get('arcAudioMode', 'legacy'))
    if mode in ('off', 'legacy'):
        selected = _candidate_times_music(analysis, config, rng, model)
    else:
        from .attention_candidates import select_attention_candidates
        clock, _ = _mapping_clock(analysis, config)
        attention_config = dict(config, arcAudioMode=mode)
        legacy = _candidate_times_music(analysis, config, rng, model)[0] if mode in ('hybrid','arcaudio3') else None
        selected = select_attention_candidates(analysis, attention_config, rng, model, clock, _contexts(analysis), legacy)
    if config.get('learnWholeLevel'):
        return _composition_budget(selected, analysis, config)
    return selected


def _composition_for(analysis, config, second, musical_beat):
    """Cache one learned phrase profile; never retain it in project settings."""
    bar = int((float(musical_beat) + 1e-7) // 4)
    cache = config.setdefault('_compositionCache', {})
    key = (bar, int(config.get('rank', 9)))
    if key not in cache:
        grid = analysis.get('beatTimes') or []
        center = bar * 4 + 2
        if len(grid) > center >= 0:
            second = float(np.interp(center, np.arange(len(grid)), grid))
        try:
            from .composition import composition_profile
            cache[key] = composition_profile(config.get('_compositionModelDir', pathlib.Path(__file__).resolve().parents[1] / 'models/style'),
                analysis, second, config)
        except ImportError:
            cache[key] = {'ready': False, 'method': 'Whole-level model is not available.'}
    return cache[key]


def _composition_budget(selected, analysis, config):
    """A learned phrase envelope can remove redundant hits, never invent times."""
    candidates, minimum, limit = selected
    groups = collections.defaultdict(list)
    for candidate in candidates: groups[int((float(candidate['musicalBeat']) + 1e-7) // 4)].append(candidate)
    chosen = []; summaries = []
    for group in groups.values():
        middle = group[len(group) // 2]
        profile = _composition_for(analysis, config, middle['time'], middle['musicalBeat'])
        if not profile.get('ready'):
            chosen.extend(group); continue
        if config.get('mappingRegions'):
            from .regions import config_at_time
            intent = config_at_time(config, middle['time'])
        else: intent = config
        confidence = max(0., min(1., float(profile.get('confidence', 0))))
        contrast = max(0., min(1., float(intent.get('philosophy', {}).get('contrast', 65)) / 100))
        ratio = max(.08, min(2.2, float(profile.get('densityRatio', 1))))
        fraction = 1 + (min(1., ratio) - 1) * confidence * (.35 + .65 * contrast)
        # Whole-map density ratios are supporting evidence, not a second hard
        # ceiling on an explicitly selected, rhythmically busy impact. Preserve
        # the section plan's controlled intensity while still shaping rests.
        shared_section=next((s for s in analysis.get('musicAnalysis',{}).get('sections',[])
            if float(s['start'])<=middle['time']<float(s['end'])),{})
        drop=(shared_section.get('label')=='Drop' and float(shared_section.get('activity',0))>=.65
            or any(p.get('section')=='Drop' and float(p.get('activity',0))>=.65 for p in group))
        # Attention candidates carry context states (e.g. rhythm_focused), not
        # song labels. Their foreground budget has already excluded background
        # layers; do not interpret a context label mismatch as a quiet passage.
        attention_active=any(p.get('attentionState') and float(p.get('targetNps',0))>=5 for p in group)
        if attention_active:
            # Foreground admission, difficulty quota and both-hand recovery
            # already spent one physical-note budget. A whole-map relative
            # density ratio is not new evidence of musical redundancy and must
            # not prune that same active budget a second time. Learned motion,
            # complexity and motif variation still condition placement below.
            fraction = 1.
        elif drop:
            fraction = max(.9, fraction)
        # A strongly selected vocal/lead phrase retains a foreground anchor.
        # Other already-sparse supporting phrases may legitimately become empty
        # when human composition evidence recommends a near-zero density.
        anchor = 1 if any(float(p.get('importance', 0)) >= .6 and p.get('source') in
            ('vocals', 'spoken', 'melody', 'lead', 'piano') for p in group) else 0
        quota = max(anchor, int(round(len(group) * fraction)))
        kept = sorted(sorted(group, key=lambda p: p.get('importance', p.get('score', 0)), reverse=True)[:quota], key=lambda p: p['time'])
        chords = [p for p in kept if p.get('chord')]
        if chords and not attention_active:
            old_fraction = len(chords) / len(kept)
            learned_fraction = max(0., min(1., float(profile.get('doubleFraction', old_fraction))))
            cap = max(0, min(len(chords), int(round(len(kept) * (old_fraction * (1 - confidence) + learned_fraction * confidence)))))
            allow = {id(p) for p in sorted(chords, key=lambda p: p.get('score', 0), reverse=True)[:cap]}
            kept = [dict(p, chord=False) if p.get('chord') and id(p) not in allow else p for p in kept]
        chosen.extend(kept)
        summaries.append({'start': group[0]['time'], 'end': group[-1]['time'], 'candidatesBefore': len(group),
            'candidatesAfter': len(kept), 'densityRatio': ratio, 'confidence': confidence,
            'hitsBefore':sum(2 if p.get('chord') else 1 for p in group),
            'hitsAfter':sum(2 if p.get('chord') else 1 for p in kept),
            'movement': profile.get('movement'), 'complexity': profile.get('complexity'),
            'patternVariety': profile.get('patternVariety'), 'repeatVariation': profile.get('repeatVariation'),
            'matches': profile.get('matches', [])[:3]})
    analysis['_arcForgeCompositionReport'] = {'ready': bool(summaries), 'phrases': summaries,
        'candidatesBefore': len(candidates), 'candidatesAfter': len(chosen),
        'timingPolicy': 'Learned phrase density only selects among already supported mapper candidates; no new timings or RNG calls.'}
    return sorted(chosen, key=lambda p: p['time']), minimum, limit


def _regional_target(intent, activity, section, observed_rate, limit, expected_rate):
    """Creative density modifies supported hits, never their musical times."""
    from .styles import GOALS
    requested = max(.25, min(float(intent.get('_resolvedNps', intent.get('nps', 10))), 30.))
    philosophy = intent.get('philosophy', {})
    contrast = float(philosophy.get('contrast', 65)) / 100
    stamina = float(philosophy.get('stamina', intent.get('profile', {}).get('stamina', 75))) / 100
    style = intent.get('style', 'Modern')
    goal = intent.get('styleGoal', GOALS.get(style, GOALS['Modern']))
    if intent.get('npsRange') is not None:
        # Activity already resolved this explicit range; available opportunities
        # and hand recovery can limit output without attenuating its goal twice.
        return min(limit, expected_rate, requested)
    density = float(section.get('densityFactor', .08 + .9 * activity)) if section else .08 + .9 * activity
    density = max(.04, density) ** (.65 + contrast * .65)
    if intent.get('dynamicDensity', True) is False and activity >= .45: density = 1.
    label = section.get('label', 'Drive') if section else 'Drive'
    target = requested * min(1.15, density)
    if label not in ['Drop', 'Build']: target *= .78 + .25 * goal[0]
    if label == 'Drive': target *= .85 + .15 * stamina
    if label in ['Intro', 'Break', 'Outro']:
        target = min(target, max(.6, observed_rate * (.70 if int(intent.get('rank', 9)) < 7 else .95)))
    elif label == 'Drop': target = min(limit, max(target, requested * min(1.1, activity)))
    if style == 'Expressive' and label != 'Drop': target *= .82
    if style == 'Flow' and label == 'Build': target *= .90
    return min(limit, expected_rate, target)


def _fit_note_budget(points, quota):
    """An accent double costs two notes, not one independent timing event."""
    points=list(points);excess=sum(2 if p.get('chord') else 1 for p in points)-max(0,int(quota))
    for point in sorted((p for p in points if not p.get('chord')),key=lambda p:p['score']):
        if excess<=0:break
        points.remove(point);excess-=1
    if excess>0:
        for point in sorted((p for p in points if p.get('chord')),key=lambda p:p['score']):
            if excess<=0:break
            point['chord']=False;excess-=1
    return sorted(points,key=lambda p:p['time'])


def _candidate_times_music(analysis, config, rng, model):
    """Pick phrase hits from observed attacks and rhythm, never amplitude alone."""
    from .music import ensure_music_analysis
    analysis = ensure_music_analysis(analysis)
    music = analysis['musicAnalysis']; clock, _ = _mapping_clock(analysis, config)
    bpm = float(config.get('bpm') or analysis['bpm']); spb = 60 / bpm
    requested = max(.25, min(float(config.get('nps', 10)), 30.))
    rank = int(config.get('rank', 9 if requested >= 8 else 7 if requested >= 5 else 5 if requested >= 3 else 3))
    minimum = float(config.get('minimumHandInterval', .12 if rank >= 9 else .16 if rank >= 7 else .22 if rank >= 5 else .30))
    if config.get('profile'):minimum*=1+(100-float(config['profile'].get('speed',85)))*.002
    mode = config.get('mode', 'Standard'); style = config.get('style', 'Technical')
    philosophy = config.get('philosophy', {})
    from .styles import GOALS
    style_goal=GOALS.get(style,GOALS['Modern'])
    profile=config.get('profile',{})
    stamina=float(philosophy.get('stamina',profile.get('stamina',75)))/100
    contrast = float(philosophy.get('contrast', 65)) / 100
    musicality = float(philosophy.get('musicality', 85)) / 100
    grid_times = analysis.get('beatTimes') or []
    offset = float(analysis.get('offset', 0)); duration = float(analysis['duration'])
    sections = music.get('sections', [])
    frame_times, contexts = _contexts(analysis)
    accents = music.get('accents', [])
    attack_times = np.array([a['time'] for a in accents], np.float32)
    rhythm_frames = music.get('rhythmFrames', [])
    rt = np.array([f['time'] for f in rhythm_frames]); activities = np.array([f.get('activity', 0) for f in rhythm_frames])
    motifs = music.get('motifs', [])
    guided = bool(config.get('mappingRegions'))
    if guided:
        from .regions import config_at_time, region_boundaries
    if config.get('continuationMatchRhythm'):
        from .continuation import rhythm_bias
    from .scheduling import select_supported
    candidates = []; density_rows = []; effective_limit = min(requested, (1 if mode == 'OneSaber' else 2) / minimum)
    hands = 1 if mode == 'OneSaber' else 2
    beats = len(grid_times) - 1 if len(grid_times) > 2 else int((duration - offset) / spb)
    for bar in range(0, max(1, beats), 4):
        start = float(np.interp(bar, np.arange(len(grid_times)), grid_times)) if len(grid_times) > bar else offset + bar * spb
        end = float(np.interp(bar + 4, np.arange(len(grid_times)), grid_times)) if len(grid_times) > bar + 4 else start + 4 * spb
        end = min(end, duration - .35)
        if end <= max(start, .5): continue
        section = next((s for s in sections if s['start'] <= (start + end) / 2 < s['end']), None)
        activity = float(np.interp((start + end) / 2, rt, activities)) if len(rt) else .3
        density = float(section.get('densityFactor', .08 + .9 * activity)) if section else .08 + .9 * activity
        density = max(.04, density) ** (.65 + contrast * .65)
        if config.get('dynamicDensity', True) is False and activity >= .45: density = 1.
        label = section.get('label', 'Drive') if section else 'Drive'
        target = requested * min(1.15, density)
        if label not in ['Drop','Build']:target*=.78+.25*style_goal[0]
        # Style and stamina goals regulate sustained load; they never create
        # events in a pause or override the measured rhythmic activity.
        if label=='Drive':target*=.85+.15*stamina
        # Explicitly resist a requested 12 NPS in a slow piano/voice passage.
        # Drops may reach or exceed the user's target when the audio supports it.
        attack_ids = np.flatnonzero((attack_times >= start) & (attack_times < end))
        observed_rate = len(attack_ids) / max(end - start, .1)
        if label in ['Intro', 'Break', 'Outro']:
            target = min(target, max(.6, observed_rate * (.70 if rank < 7 else .95)))
        elif label == 'Drop': target = min(effective_limit, max(target, requested * min(1.1, activity)))
        if style == 'Expressive' and label != 'Drop': target *= .82
        if style == 'Flow' and label == 'Build': target *= .90
        local_spb = (end - start) / 4
        subdivision = .25
        expected_rate = 1 / (local_spb * subdivision)
        # Keep the musical grid at high BPM. Hand scheduling can select a safe
        # subset; halving the entire grid discarded every other opportunity.
        # At low BPM, observed faster attacks can legitimately exceed 1/4 beats.
        available_rate = max(expected_rate, observed_rate)
        target = min(target, available_rate)
        desired = max(0, int(round(target * (end - start))))
        pool = []
        motif = next((m for m in motifs if abs(m.get('beat', -1) - bar) < .1), {})
        for step in np.arange(0, 4, subdivision):
            musical_beat = bar + float(step)
            second = float(np.interp(musical_beat, np.arange(len(grid_times)), grid_times)) if len(grid_times) > musical_beat else offset + musical_beat * spb
            if second < .5 or second > duration - .35: continue
            intent = config_at_time(config, second) if guided else config
            local_musicality = float(intent.get('philosophy', {}).get('musicality', 85)) / 100 if guided else musicality
            local_style = intent.get('style', style) if guided else style
            context = sample_context(frame_times, contexts, second)
            nearest = None; error = 1.
            if len(attack_times):
                i = int(np.searchsorted(attack_times, second))
                indices = [v for v in [i - 1, i] if 0 <= v < len(attack_times)]
                nearest = min(indices, key=lambda j: abs(float(attack_times[j]) - second)) if indices else None
                if nearest is not None: error = abs(float(attack_times[nearest]) - second)
            supported = error <= min(.055 - local_musicality * .015, local_spb * .17)
            # Sustained sections get actual musical attacks only. Busy passages
            # may fill a coherent fast subdivision between those attacks.
            if not supported and (activity < .52 or observed_rate < 3.5): continue
            phase = int(round(musical_beat * 4)) % 16
            prior = float(model['rhythm'][min(2, int(context[0] * 2)), phase].mean())
            support = np.exp(-.5 * (error / .023) ** 2)
            accent = accents[nearest] if supported and nearest is not None else None
            role = (accent['roles'][0] if accent and accent.get('roles') else section.get('primaryRole', 'midTransient') if section else 'midTransient')
            score = 1.45 * support + .35 * prior + .12 * context[2] + (.15 if phase % 4 == 0 else 0) + rng.uniform(0, .08)
            if local_style == 'Expressive' and role == 'tonalChange': score += .28
            if local_style == 'Technical' and role in ['midTransient', 'highPercussion']: score += .16
            if local_style == 'Flow' and phase % 2 == 0: score += .10
            if config.get('continuationMatchRhythm'): score *= rhythm_bias(config, second, musical_beat)
            pool.append({'time': second, 'beat': clock.beat(second), 'musicalBeat': musical_beat, 'context': context,
                         'chord': False, 'role': role, 'section': label, 'activity': activity,
                         'audioMotif': motif.get('motifId'), 'timingError': error if supported else None,
                         'score': score, 'targetNps': round(target, 3)})
        opportunity_target = max([target] + [float(r.get('nps') or max(r.get('npsRange') or [0.]))
            for r in config.get('mappingRegions', []) if r['start'] < end and r['end'] > start]) if guided else target
        if observed_rate > expected_rate * 1.08 and opportunity_target > expected_rate * .95:
            for attack_id in attack_ids:
                accent = accents[int(attack_id)]; second = float(accent['time'])
                if second < .5 or second > duration - .35 or any(abs(p['time'] - second) < .018 for p in pool): continue
                context = sample_context(frame_times, contexts, second)
                musical_beat = float(np.interp(second, grid_times, np.arange(len(grid_times)))) if grid_times else (second - offset) / spb
                role = accent.get('roles', ['midTransient'])[0]
                intent = config_at_time(config, second) if guided else config
                score = 1.45 + min(1., float(accent.get('strength', .5))) * .28 + (.15 if role == 'lowPercussion' else 0)
                if config.get('continuationMatchRhythm'): score *= rhythm_bias(config, second, musical_beat)
                pool.append({'time':second,'beat':clock.beat(second),'musicalBeat':musical_beat,'context':context,
                    'chord':False,'role':role,'section':label,'activity':activity,'audioMotif':motif.get('motifId'),
                    'timingError':0.,'score':score,'targetNps':round(target,3),'timingOrigin':'observed_attack'})
        if guided:
            boundaries = sorted(set([start, end] + [t for t in region_boundaries(config['mappingRegions'], duration) if start < t < end]))
            chosen = []
            for a, b in zip(boundaries, boundaries[1:]):
                intent = config_at_time(config, (a + b) / 2, activity=activity)
                local_target = _regional_target(intent, activity, section, observed_rate,
                    hands / minimum, available_rate)
                effective_limit = max(effective_limit, local_target)
                quota = max(0, int(round(local_target * (b - a))))
                local_pool = [p for p in pool if a <= p['time'] < b]
                kept, selection = select_supported(local_pool, quota, minimum, hands,
                    [p['time'] for p in candidates[-hands:] + chosen[-hands:]])
                for point in kept:
                    point['targetNps'] = round(local_target, 3)
                    chosen.append(point)
                density_rows.append({'start':a,'end':b,'label':label,'requestedNps':float(intent.get('_resolvedNps',intent.get('nps',10))),
                    **({'npsRange':list(intent['npsRange'])} if intent.get('npsRange') is not None else {}),
                    'targetNps':round(local_target,3),**selection})
            chosen.sort(key=lambda p: p['time'])
        else:
            chosen, selection = select_supported(pool, desired, minimum, hands,
                [p['time'] for p in candidates[-hands:]])
            density_rows.append({'start':start,'end':end,'label':label,'requestedNps':requested,
                'targetNps':round(target,3),**selection})
        for point in chosen:
            phase = int(round(point['musicalBeat'] * 4)) % 16
            double_rate = float(model.get('doubles', np.zeros((3, 16)))[min(2, int(point['context'][0] * 2)), phase])
            loud_accent = point['role'] == 'lowPercussion' and point['timingError'] is not None
            chord = mode != 'OneSaber' and phase % 4 == 0 and loud_accent and rng.random() < min(.35, double_rate + .05)
            if chord and all(abs(other['time'] - point['time']) >= minimum for other in chosen if other is not point) and all(
                point['time']-other['time']>=minimum for other in candidates[-2:]):point['chord'] = True
        if guided:
            for a,b in zip(boundaries,boundaries[1:]):
                row=next(r for r in reversed(density_rows) if r['start']==a and r['end']==b)
                local=[p for p in chosen if a<=p['time']<b]
                retained=_fit_note_budget(local,row['quota'])
                row['selectedEvents']=len(retained);row['selectedHits']=sum(2 if p.get('chord') else 1 for p in retained)
                candidates.extend(retained)
        else:
            retained=_fit_note_budget(chosen,desired)
            density_rows[-1]['selectedEvents']=len(retained)
            density_rows[-1]['selectedHits']=sum(2 if p.get('chord') else 1 for p in retained)
            candidates.extend(retained)
    analysis['_arcForgeCandidateReport'] = {'densityCurve':density_rows,
        'selectionPolicy':'Salient supported hits, two-hand recovery scheduling, then reachable-density backfill; no new hit times.'}
    return sorted(candidates, key=lambda p: p['time']), minimum, effective_limit


def _valid_tokens(history, hand, seconds, minimum, allow_dots, mode, config=None, other_history=()):
    from .diagnostics import VECTORS
    config=config or {};rank=int(config.get('rank',9));style=config.get('style','Modern')
    result = np.ones(108, dtype=bool)
    previous = history[-1] if history else None
    gap = seconds - previous['seconds'] if previous else 10
    goal = config.get('_measuredStyleGoal', (0, 0, 0, 0))
    movement_goal = max(float(config.get('philosophy', {}).get('movement', 50)) / 100, float(goal[2]))
    expressive = style in ('Dance','Jump','Tech','Technical','Challenge','Experimental','Tech-heavy','Powerhouse','Expressive')
    crossing = rank >= 7 and expressive and gap >= .38 and movement_goal >= .65
    lanes = ([0, 1, 2] if hand == 0 else [1, 2, 3]) if crossing else ([0, 1] if hand == 0 else [2, 3])
    for value in range(108):
        note = untoken(value)
        x, y, direction = note['x'], note['y'], note['d']
        # Own side, vertical/diagonal cuts and bounded movement keep difficulty in
        # speed and phrasing instead of forced horizontal wrist inversions.
        if x not in lanes: result[value] = False
        if direction in [2,3] and (rank<7 or gap<.24 or style in ['Speed','Stream','OST','Light/easy','Accuracy','Acc-focused']):result[value]=False
        if rank<=3 and direction not in [0,1,8]:result[value]=False
        if direction == 8 and not allow_dots: result[value] = False
        if gap < .22 and y == 2: result[value] = False
        if previous:
            distance = np.hypot(x - previous['x'], y - previous['y'])
            if distance > (1.55 if gap < .22 else 2.5): result[value] = False
            if previous['d'] != 8 and direction != 8 and _angle_difference(previous['d'], direction) < 135: result[value] = False
            # Cross-lane movement must remain slow enough for its travel and
            # endpoint recovery, not merely pass an arbitrary lane whitelist.
            if crossing or gap < .28:
                endpoint = np.array([previous['x'], previous['y']]) + np.asarray(VECTORS[previous['d']]) * .55
                entry = np.array([x, y]) - np.asarray(VECTORS[direction]) * .55
                max_recovery = 5.5 if crossing else 7.25
                if np.linalg.norm(entry - endpoint) * .6 / max(gap, .01) > max_recovery: result[value] = False
        if other_history and seconds - other_history[-1]['seconds'] < .12:
            other = other_history[-1]
            if abs(x - other['x']) < .5 and abs(y - other['y']) < .5: result[value] = False
        # Inner down diagonals aim away from the opposite saber when swung alone;
        # simultaneous crosses are rejected separately by keeping both lanes apart.
    return result


def generate_map(audio_analysis, config, model_dir):
    from .music import ensure_music_analysis
    audio_analysis = ensure_music_analysis(audio_analysis)
    from .arcaudio.settings import canonical_mode
    attention_mode = canonical_mode(config.get('arcAudioMode', 'legacy'))
    raw_regions = config.get('mappingRegions', config.get('regions', []))
    guided = bool(raw_regions)
    whole_level = bool(config.get('learnWholeLevel'))
    if whole_level:
        config = dict(config, _compositionModelDir=str(model_dir), _compositionCache={})
        from .variety import VarietyState, variety_multiplier, analyze_variety
        variety_state = VarietyState()
    style_cache = {}
    if guided:
        from .regions import validate_regions, config_at_time, blended_style_profile
        config = dict(config, mappingRegions=validate_regions(raw_regions, audio_analysis['duration']))
    from .arcaudio.focus import has_manual_focus
    if attention_mode in ('off', 'legacy') and has_manual_focus(config, audio_analysis['duration']):
        raise ValueError('Manual source attention needs ArcAudio 2.0, Hybrid or ArcAudio 3.0. Select an attention engine or use Auto regions.')
    from .styles import style_profile
    learned_style=style_profile(config.get('style','Modern'),config)
    def learned_for(intent):
        if not guided: return learned_style
        key = (intent.get('style', 'Modern'), tuple(intent.get('secondaryStyles', [])), int(intent.get('rank', 9)),
               tuple((entry['style'], round(float(entry['weight']), 3)) for entry in intent.get('styleWeights', [])))
        if key not in style_cache: style_cache[key] = blended_style_profile(intent)
        return style_cache[key]
    model_path = pathlib.Path(model_dir) / 'model.npz'
    if not model_path.exists(): raise ValueError('Train the local model before generating a level')
    with np.load(model_path) as loaded:
        model = {key: loaded[key].copy() for key in loaded.files}
    seed = config.get('seed')
    if seed is None:
        seed = int.from_bytes(hashlib.sha256((str(audio_analysis.get('audioPath')) + str(config.get('label')) + str(config.get('mode'))).encode()).digest()[:8], 'little')
    rng = np.random.default_rng(int(seed)); candidates, minimum, effective_target = _candidate_times(audio_analysis, config, rng, model)
    from .attention_viability import require_viable_attention
    attention_viability = require_viable_attention(audio_analysis, config, candidates)
    clock, tempo_events = _mapping_clock(audio_analysis, config)
    from .planning import build_map_plan, critique_generation
    map_plan = build_map_plan(audio_analysis, config, candidates, minimum, clock)
    plan_edges = [p['start'] for p in map_plan['sections']]
    stereo=audio_analysis['musicAnalysis'].get('stereo',{}).get('windows',[])
    stereo_times=np.array([w['time'] for w in stereo]);stereo_pan=np.array([w.get('pan') or 0 for w in stereo])
    mode = config.get('mode', 'Standard'); allow_dots = bool(config.get('events', {}).get('dotBlocks', config.get('dotBlocks', False)))
    backend = config.get('backend', config.get('model', 'hybrid'))
    if backend not in ('local','inferno','hybrid','auto'):
        raise ValueError('Unknown placement engine. Choose Local, InfernoSaber, or Hybrid.')
    inferno, inferno_metadata, warnings = None, None, []
    if backend == 'inferno' and not pathlib.Path(audio_analysis.get('audioPath','')).is_file():
        raise ValueError('InfernoSaber requires the original decoded audio. Choose Local for an audio-free fixture.')
    if backend in ['inferno', 'hybrid', 'auto'] and audio_analysis.get('audioPath'):
        from .inferno_adapter import predict, status
        if status()['weightsAvailable']:
            try:
                if candidates: inferno, inferno_metadata = predict(audio_analysis['audioPath'], [c['time'] for c in candidates])
            except Exception as e:
                if backend == 'inferno': raise RuntimeError('The selected InfernoSaber engine failed: ' + str(e)) from e
                warnings.append('InfernoSaber failed; local learned model used: ' + str(e))
        else:
            if backend == 'inferno': raise ValueError('The selected InfernoSaber engine is unavailable. Install its weights or choose Local / Hybrid.')
            warnings.append('InfernoSaber weights unavailable; local learned model used')
    elif backend in ('hybrid','auto'):
        warnings.append('Original audio unavailable; local learned model used')
    from .scheduling import HandAssignment
    hand_assignment = HandAssignment()
    histories = {0: [], 1: []}; notes = []; next_hand = 0
    recent_tokens = {0: collections.deque(maxlen=12), 1: collections.deque(maxlen=12)}
    recent_ngrams = collections.Counter(); bars = collections.defaultdict(list)
    substitutions = 0;motif_memory={};motif_returns=0
    expanded = []
    for index, candidate in enumerate(candidates):
        candidate['_emittedHits'] = 0
        if candidate.get('chord'):
            expanded.extend([candidate | {'forcedHand': hand, 'sourceIndex': index} for hand in [0, 1]])
        else: expanded.append(candidate | {'sourceIndex': index})
    for candidate in expanded:
        index = candidate['sourceIndex']
        active_config = config_at_time(config, candidate['time']) if guided else config
        active_learned_style = learned_for(active_config)
        plan_section = map_plan['sections'][max(0, bisect.bisect_right(plan_edges,candidate['time'])-1)] if plan_edges else {}
        composition = _composition_for(audio_analysis, config, candidate['time'], candidate['musicalBeat']) if whole_level else {}
        hand = hand_assignment.preferred(candidate, histories, minimum, mode)
        second, beat, context = candidate['time'], candidate['beat'], candidate['context']
        pan=float(np.interp(second,stereo_times,stereo_pan)) if len(stereo_times) else 0.
        if histories[hand] and second - histories[hand][-1]['seconds'] < minimum - .0001:
            other = 1 - hand
            if mode == 'OneSaber' or candidate.get('chord') or (histories[other] and second - histories[other][-1]['seconds'] < minimum - .0001): continue
            hand = other
        history = histories[hand]
        gap = second - history[-1]['seconds'] if history else 1
        feature = placement_features(history, histories[1 - hand], hand, candidate['musicalBeat'], gap,
                                     float(plan_section.get('localBpm') or active_config.get('bpm') or audio_analysis['bpm']),
                                     float(candidate.get('targetNps') or active_config.get('nps', 10)), context)
        feature = np.clip((feature - model['mean']) / model['std'], -6, 6)
        probability = _softmax((np.maximum(feature @ model['w1'] + model['b1'], 0) @ model['w2'] + model['b2'])[None])[0]
        probability*=np.maximum(active_learned_style['prior']*108,1e-5)**.28
        if inferno is not None and index < len(inferno):
            # Product of experts: real pretrained recommendations meet uploaded-map adaptation.
            probability *= (np.maximum(inferno[index, hand], 1e-6) * 108) ** .45
        allowed = _valid_tokens(history, hand, second, minimum, allow_dots, mode,
            dict(active_config, _measuredStyleGoal=active_learned_style['goal']), histories[1-hand])
        style = active_config.get('style', 'Technical'); philosophy = active_config.get('philosophy', {})
        rank_factor=min(1.,.2+int(active_config.get('rank',9))*.085)
        movement = (.5*float(philosophy.get('movement', 50))/100+.5*active_learned_style['goal'][2])*rank_factor
        technique = (.5*float(philosophy.get('tech', 60))/100+.5*active_learned_style['goal'][1])*rank_factor
        if composition.get('ready'):
            evidence_weight = .30 * max(0., min(1., float(composition.get('confidence', 0))))
            movement = movement * (1 - evidence_weight) + float(composition.get('movement', movement)) * rank_factor * evidence_weight
            technique = technique * (1 - evidence_weight) + float(composition.get('complexity', technique)) * rank_factor * evidence_weight
        if active_config.get('profile'):
            movement*=.75+.25*float(active_config['profile'].get('jumping',65))/100
            technique*=.75+.25*float(active_config['profile'].get('tech',72))/100
        for value in np.flatnonzero(allowed):
            position = untoken(value)
            if history:
                reach = float(np.hypot(position['x'] - history[-1]['x'], position['y'] - history[-1]['y']))
                # Travel is an objective, not an unconditional distance penalty.
                # The old penalty made almost every style prefer the same small
                # inline wrist pattern. Planned density limits rapid travel;
                # expressive slower phrases can use their measured style range.
                desired_travel = (.3 + 2.1 * movement) * min(1., gap / .34)
                if float(plan_section.get('targetNps',0)) >= 10: desired_travel = min(1.2, desired_travel)
                probability[value] *= np.exp(-abs(reach - desired_travel) * (.9 + .4 * movement))
            if position['d'] in [4, 5, 6, 7]: probability[value] *= .7 + technique
            if candidate.get('role') == 'highPercussion' and position['y'] == 0: probability[value] *= 1.65
            if candidate.get('role') == 'tonalChange' and gap > .28 and position['d'] in [4, 5, 6, 7]: probability[value] *= 1.35
            # Measured stereo balance is a modest spatial cue. Physical masks
            # still enforce hand recovery and allowed lanes.
            if abs(pan)>.1:probability[value]*=np.exp(-.35*abs((position['x']-1.5)/1.5-pan))
            if style == 'Flow' and history and position['x'] == history[-1]['x']: probability[value] *= 1.5
            if style == 'Expressive' and gap > .35 and position['y'] == 1: probability[value] *= 1.6
        if candidate.get('chord'):
            for value in range(108):
                position = untoken(value)
                if position['d'] in [4, 5, 6, 7] and position['x'] != (0 if hand == 0 else 3):
                    allowed[value] = False
        probability[~allowed] = 0
        motif_key=(candidate.get('audioMotif'),int(round(candidate['musicalBeat']*4))%16,hand)
        remembered=motif_memory.get(motif_key)
        if motif_key[0] is not None and remembered is not None:
            # Recall a phrase's movement family while allowing height and angle
            # variation. Allowed geometry and neural probabilities still decide.
            prior_note=untoken(remembered)
            for v in np.flatnonzero(allowed):
                n=untoken(v)
                if n['x']==prior_note['x'] and (n['d']==prior_note['d'] or _angle_difference(n['d'],prior_note['d'])<=45):probability[v]*=1.7
            motif_returns+=1
        # Penalize reused four-swing sequences and recently overused locations/cuts.
        repeat_allowance = 1.
        if whole_level and candidate.get('audioMotif') is not None:
            identity = str(candidate['audioMotif']); current_bar = int((candidate['musicalBeat'] + 1e-7) // 4)
            if variety_state.motif_first_bar.get(identity, current_bar) < current_bar:
                repeat_allowance = .25 + .55 * max(0., min(1., float(composition.get('repeatVariation', .15))))
        for value in np.flatnonzero(allowed):
            seq = tuple(list(recent_tokens[hand])[-3:] + [int(value)])
            variety = float(active_config.get('philosophy', {}).get('variety', 65)) / 100
            probability[value] /= (1 + recent_ngrams[(hand, seq)]) ** ((.8 + 1.5 * variety) * repeat_allowance)
            probability[value] /= 1 + .3 * repeat_allowance * list(recent_tokens[hand]).count(value)
            if whole_level:
                confidence = max(0., min(1., float(composition.get('confidence', 0)))) if composition.get('ready') else 0.
                learned_variety = float(composition.get('patternVariety', variety))
                target_variety = variety * (1 - .3 * confidence) + learned_variety * .3 * confidence
                probability[value] *= variety_multiplier(history, untoken(value), hand, candidate['musicalBeat'],
                    state=variety_state, phrase_id=candidate.get('audioMotif'), variety=target_variety,
                    contrast=float(philosophy.get('contrast', 65)) / 100,
                    repeat_variation=float(composition.get('repeatVariation', .15)))
        if probability.sum() <= 0:
            probability = allowed.astype(np.float64); substitutions += 1
        probability = probability ** (1 / float(active_config.get('temperature', .9)))
        probability /= probability.sum()
        value = int(rng.choice(108, p=probability))
        placement = untoken(value)
        if motif_key[0] is not None:motif_memory.setdefault(motif_key,value)
        note = {'b': round(beat, 6), 'c': hand, **placement, 'a': 0,
                'customData': {'arcforgeRole': candidate.get('role'), 'arcforgeSection': candidate.get('section'),
                    'arcforgePlanSection':plan_section.get('id')}}
        if attention_mode in ('arcaudio', 'hybrid', 'arcaudio3'):
            note['customData'].update(arcforgeSource=candidate.get('source'),
                arcforgeAttentionConfidence=candidate.get('attentionConfidence'))
            evidence=candidate.get('audioEvidence')
            if evidence:
                note['customData'].update(arcforgeEventId=evidence.get('eventId'),
                    arcforgeObservedTime=evidence['eventTime'],
                    arcforgeImportance=round(float(candidate.get('importance',0)),5),
                    arcforgeSourceConfidence=evidence.get('confidence',0),
                    arcforgeAccent={'double':bool(candidate.get('chord')),
                        'decision':candidate.get('accentDecision',{}).get('reason','single_foreground_event')})
        if guided and active_config.get('mappingRegionId'):
            note['customData']['arcforgeRegion'] = active_config['mappingRegionId']
        if mode == 'NoArrows': note['d'] = 8
        if whole_level:
            variety_state.record(history, note, candidate['musicalBeat'], candidate.get('audioMotif'))
        notes.append(note); history.append(note | ({'seconds': second, 'musicalBeat': candidate['musicalBeat']} if whole_level else {'seconds': second}))
        candidates[index]['_emittedHits'] += 1
        hand_assignment.record(hand)
        sequence = tuple(list(recent_tokens[hand])[-3:] + [value]); recent_ngrams[(hand, sequence)] += 1
        recent_tokens[hand].append(value); next_hand = 1 - hand
        bars[int(candidate['musicalBeat'] // 4)].append((round(candidate['musicalBeat'] % 4, 2), hand, value))
    beatmap = {'version': '3.3.0', 'bpmEvents': tempo_events, 'rotationEvents': [], 'colorNotes': notes, 'bombNotes': [],
               'obstacles': [], 'sliders': [], 'burstSliders': [], 'waypoints': [], 'basicBeatmapEvents': [],
               'colorBoostBeatmapEvents': [], 'lightColorEventBoxGroups': [], 'lightRotationEventBoxGroups': [],
               'lightTranslationEventBoxGroups': [], 'basicEventTypesWithKeywords': {'d': []}, 'useNormalEventsAsCompatibleEvents': True,
               'customData': {'arcforge': {'seed': int(seed), 'backend': inferno_metadata['backend'] if inferno_metadata else 'ArcForge local neural model'}}}
    if mode in ['90Degree', '90Degrees', '360Degree', '360Degrees']:
        duration_beats = float(audio_analysis['duration']) * float(audio_analysis['bpm']) / 60
        # Spawn rotations use Beat Saber v3 event semantics, not rotated note arrows.
        current = 0
        for beat in np.arange(16, duration_beats - 8, 16):
            amount = 15 if rng.random() > .5 else -15
            if mode.startswith('90') and abs(current + amount) > 45: amount = -amount
            current += amount
            beatmap['rotationEvents'].append({'b': clock.beat(float(beat) * 60 / audio_analysis['bpm']), 'e': 1, 'r': amount})
    report = audit_map(beatmap, float(config.get('bpm') or audio_analysis['bpm']), float(audio_analysis['duration']), minimum, mode)
    signatures = [tuple(v) for _, v in sorted(bars.items()) if v]
    report.update({'backend': inferno_metadata['backend'] + ' + local learned adaptation' if inferno_metadata else 'ArcForge local audio-conditioned neural placement',
                   'inferno': inferno_metadata, 'requestedNps': config.get('nps', 10), 'effectiveGridNpsLimit': round(effective_target, 2),
                   'seed': int(seed), 'distinctBars': len(set(signatures)), 'totalBars': len(signatures),
                   'consecutiveIdenticalBars': sum(a == b for a, b in zip(signatures, signatures[1:])),
                   'guardFallbacks': substitutions, 'warnings': warnings,
                   'doubleAccents': sum(len(v) == 2 for v in _group_notes(notes).values()),
                   'mappingStyle': config.get('style', 'Technical'), 'musicAnalysisVersion': audio_analysis['musicAnalysis']['version'],
                   'styleConditioning':{'method':learned_style['method'],'nearestHumanCharts':learned_style['matches'],'motifRecallHits':motif_returns},
                   'musicalSections': _section_report(notes, audio_analysis, config, clock, map_plan),
                   'timingErrorsMs': _timing_report([p for p in candidates for _ in range(p['_emittedHits'])]), 'minimumHandIntervalGuardMs': round(minimum * 1000, 2),
                   'limitations': ['Static checks cannot certify comfort in a headset or ranking eligibility.',
                                   'Upstream network is frozen; uploaded ZIP training updates the local placement network.']})
    effective_audio = 'legacy' if attention_mode == 'off' else attention_mode
    recognition = audio_analysis.get('_arcAudioAttention', {}).get('recognition') or audio_analysis.get('_arcAudioMapperReport',{}).get('recognition',{})
    report['engineSelection'] = {'audioAnalysis': {'requestedMode':attention_mode,'effectiveMode':effective_audio,
        'recognition':recognition.get('model', recognition.get('method'))},
        'placement': {'requestedBackend':backend,'effectiveBackend':'inferno+local' if inferno is not None else 'local',
            'fallbackReason': next((w for w in warnings if 'local learned model used' in w), None)}}
    if attention_mode in ('arcaudio', 'hybrid', 'arcaudio3'):
        report['attention'] = audio_analysis.get('_arcAudioMapperReport', {'mode': attention_mode, 'version': '2.0'})
        report['attentionViability'] = attention_viability
    if guided:
        region_rows = []
        for region in config['mappingRegions']:
            intent = config_at_time(config, (region['start'] + region['end']) / 2)
            count = int(sum(region['start'] <= clock.time(n['b']) < region['end'] for n in notes))
            region_rows.append({'id': region['id'], 'name': region['name'], 'start': region['start'], 'end': region['end'],
                'style': intent['style'], 'secondaryStyles': intent['secondaryStyles'], 'goals': intent['philosophy'],
                'requestedNps': intent.get('_resolvedNps', intent.get('nps', 10)),
                **({'npsRange':list(region['npsRange'])} if region.get('npsRange') is not None else {}),
                'analysisMode':region.get('analysisMode','auto'),
                'sources':region.get('sources',{}),'sourcePolicy':region.get('sourcePolicy','only'),
                'transitionSeconds':region.get('transitionSeconds',0), 'noteCount': count,
                'actualNps': round(count / (region['end'] - region['start']), 3)})
        report['regionReport'] = {'regions': region_rows, 'styleArchetypes': len(style_cache),
            'timingPolicy': 'Region intent changes candidate priorities and placements; it does not invent musical hit times.'}
    if whole_level:
        report['wholeLevelLearning'] = audio_analysis.get('_arcForgeCompositionReport', {'ready': False})
        report['variety'] = analyze_variety(beatmap, audio_analysis, config)
    report['mapPlan'] = map_plan
    report['critic'] = critique_generation(beatmap, audio_analysis, config, report['mapPlan'], candidates, clock)
    return {'beatmap': beatmap, 'report': report}


def _timing_report(candidates):
    errors = [c['timingError'] * 1000 for c in candidates if c.get('timingError') is not None]
    return {'supportedHits': len(errors), 'interpolatedRhythmHits': len(candidates) - len(errors),
            'medianOnsetDistance': round(float(np.median(errors)), 2) if errors else None,
            'p95OnsetDistance': round(float(np.percentile(errors, 95)), 2) if errors else None,
            'interpretation': 'Distance to a detected spectral attack; not a guarantee of perceptual synchronization.'}


def _section_report(notes, analysis, config, clock, plan=None):
    times = np.array([clock.time(n['b']) for n in notes])
    rows = []
    for section in analysis.get('musicAnalysis', {}).get('sections', []):
        duration = section['end'] - section['start']
        count = int(((times >= section['start']) & (times < section['end'])).sum())
        planned=(sum(max(0.,min(p['end'],section['end'])-max(p['start'],section['start']))*p['targetNps']
            for p in (plan or {}).get('sections',[]))/max(duration,.01)) if plan else float(config.get('nps',10))*section.get('densityFactor',1)
        rows.append({k: section.get(k) for k in ['id', 'start', 'end', 'label', 'activity', 'primaryRole', 'attacksPerSecond']} |
                    {'noteCount': count, 'actualNps': round(count / max(duration, .01), 3),
                     'targetNps': round(planned, 3)})
    return rows


def _group_notes(notes):
    groups = collections.defaultdict(list)
    for note in notes: groups[round(note['b'], 5)].append(note)
    return groups


def audit_map(beatmap, bpm, duration, minimum=.145, mode='Standard'):
    from modding import BeatClock
    clock = BeatClock(bpm, beatmap.get('bpmEvents', []))
    issues, warnings, previous = [], [], {}
    notes = sorted(beatmap.get('colorNotes', []), key=lambda n: (n['b'], n['c']))
    smallest = float('inf'); beats = collections.Counter()
    for note in notes:
        hand = note['c']; sec = clock.time(note['b'])
        if note['x'] not in range(4) or note['y'] not in range(3): issues.append('Note outside native grid')
        if sec < 0 or sec > duration: issues.append('Note outside audio bounds')
        key = (round(note['b'], 5), hand); beats[key] += 1
        if beats[key] > 1: issues.append('Same-hand duplicate time')
        if hand in previous:
            last = previous[hand]; interval = sec - clock.time(last['b'])
            smallest = min(smallest, interval)
            if interval < minimum - .001: issues.append('Same-hand interval below configured guard')
            if last['d'] != 8 and note['d'] != 8 and _angle_difference(last['d'], note['d']) < 135: issues.append('Cut parity conflict')
        previous[hand] = note
    if mode == 'OneSaber' and any(n['c'] != 1 for n in notes): issues.append('One Saber contains the wrong hand')
    times=np.array([clock.time(n['b']) for n in notes])
    peak=lambda window:round(float(np.max(np.arange(len(times))+1-np.searchsorted(times,times-window,side='right')))/window,2) if len(times) else 0
    return {'passed': not issues, 'issues': sorted(set(issues)), 'noteCount': len(notes), 'averageNps': round(len(notes) / max(duration, 1), 2),
            'peakOneSecondNps':peak(1),'peakFourSecondNps':peak(4),
            'minimumHandIntervalMs': round(smallest * 1000, 2) if np.isfinite(smallest) else None, 'warnings': warnings}
