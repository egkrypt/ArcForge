"""Human-map composition and lighting learning across complete song phrases.

This is a small, auditable statistical learner: archive-held-out ridge regression
and nearest phrase evidence, with audio-conditioned lighting distributions. It
does not copy authored layouts or claim to learn v3 event-box filter semantics.
"""
from __future__ import annotations
import bisect, collections, functools, hashlib, json, math, pathlib, tempfile, time, zipfile
import numpy as np
from training_runtime import check_cancel
from .dataset import read_archive, file_digest
from .features import audio_context, sample_context, note_seconds
from modding import BeatClock

VERSION = '1.0'
# Preserve caches when only archive-import plumbing changes. Bump this key when
# phrase/lighting feature extraction semantics change, not when UI/cache code changes.
FEATURE_SCHEMA_VERSION = '347c8d22876979b9'
FEATURES = ['energy', 'bass', 'onset', 'high', 'previousEnergy', 'nextEnergy',
            'audioNovelty', 'onsetActivity', 'difficulty', 'songPosition', 'tempoRatio', 'rhythmRegularity']
TARGETS = ['densityRatio', 'movement', 'complexity', 'doubleFraction', 'patternVariety', 'repeatVariation',
           'lightingEventsPerSecond', 'fadeFraction', 'boostFraction', 'meanBrightness', 'bombRate', 'wallRate', 'arcRate', 'chainRate']
RANK = {'Easy': 1, 'Normal': 3, 'Hard': 5, 'Expert': 7, 'ExpertPlus': 9, 'Expert+': 9}
EVENT_BOXES = ('lightColorEventBoxGroups', 'lightRotationEventBoxGroups', 'lightTranslationEventBoxGroups', 'fxEventBoxGroups')
ANGLES = [90, 270, 180, 0, 135, 45, 225, 315, 0]


def _digest(path):
    return file_digest(path)


def _extractor_version():
    return FEATURE_SCHEMA_VERSION


def _valid_events(chart):
    return [e for e in chart.get('bpmEvents', []) if isinstance(e, dict) and isinstance(e.get('b'), (int, float))
            and isinstance(e.get('m'), (int, float)) and e['b'] >= 0 and 20 <= e['m'] <= 1000]


def _lighting_bin(context):
    return int(np.searchsorted([.40, 1.0], max(0., float(context[0])))) * 3 + int(np.searchsorted([.35, 1.0], max(0., float(context[2]))))


def _json_entry(z, name):
    info = z.getinfo(name)
    if info.file_size > 30 * 1024 * 1024: raise ValueError('Map JSON exceeds the 30 MB limit')
    from .formats import read_native_json
    return read_native_json(z.read(name))


def _read_contexts(path, archive, folder):
    cache = pathlib.Path(folder) / 'audio-cache'; cache.mkdir(parents=True, exist_ok=True)
    target = cache / (archive['sha256'][:16] + '.npz')
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            times, contexts = saved['times'].copy(), saved['values'].copy()
        if contexts.ndim == 2 and contexts.shape[1] == 4 and len(times) == len(contexts): return times, contexts
    if not archive.get('audioEntry'): return np.array([], dtype=np.float32), np.empty((0, 4), dtype=np.float32)
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory(dir=cache) as temp:
        source = pathlib.Path(temp) / 'source.audio'
        source.write_bytes(z.read(archive['audioEntry']))
        times, contexts = audio_context(source)
    np.savez_compressed(target, times=times, values=contexts)
    return times, contexts


def _mean_context(times, contexts, start, end):
    if not len(times): return np.zeros(4, dtype=np.float32)
    indices = (times >= start) & (times < end)
    return contexts[indices].mean(0) if indices.any() else sample_context(times, contexts, (start + end) / 2)


def _phrase_feature(times, contexts, start, end, rank, duration, bpm, previous=None, following=None):
    c = _mean_context(times, contexts, start, end)
    previous = previous if previous is not None else _mean_context(times, contexts, max(0., start - (end - start)), start)
    following = following if following is not None else _mean_context(times, contexts, end, end + (end - start))
    inside = (times >= start) & (times < end)
    flux = contexts[inside, 2] if inside.any() else np.array([c[2]])
    regularity = float(1 / (1 + np.std(flux))) if len(flux) else 0.
    activity = float(np.mean(flux > .65)) if len(flux) else 0.
    return [*map(float, c), float(previous[0]), float(following[0]), float(np.linalg.norm(following - previous) / 6),
            activity, rank / 9, (start + end) / max(duration * 2, .1), float(bpm / 200), regularity]


def _color(event):
    cd = event.get('customData', event.get('_customData', {})) or {}
    color = cd.get('color', cd.get('_color'))
    if not isinstance(color, list) or len(color) < 3: return None
    try: rgb = tuple(round(min(8., max(0., float(v))), 2) for v in color[:3])
    except (TypeError, ValueError): return None
    return rgb if all(math.isfinite(v) for v in rgb) else None


def _light_data(raw, clock, offset, times, contexts):
    types = np.zeros((9, 32), dtype=np.int64)
    values = np.zeros((9, 32, 16), dtype=np.int64)
    phases = np.zeros((9, 32, 16), dtype=np.int64)
    brightness = np.zeros((9, 32, 2), dtype=np.float64)
    boosts = np.zeros((9, 2), dtype=np.int64)
    colors = collections.Counter(); conditional_colors = [collections.Counter() for _ in range(9)]
    transitions = collections.Counter(); previous_colors = {}; events = []
    malformed = 0
    for event in raw.get('basicBeatmapEvents', raw.get('_events', [])):
        try:
            beat = float(event.get('b', event.get('_time', 0))); kind = int(event.get('et', event.get('_type', 0)))
            value = int(event.get('i', event.get('_value', 0))); level = float(event.get('f', event.get('_floatValue', 1)))
            if not math.isfinite(beat) or not math.isfinite(level) or not 0 <= kind < 32 or not 0 <= value < 16: raise ValueError()
        except (TypeError, ValueError): malformed += 1; continue
        second = clock.time(beat) + offset
        context = sample_context(times, contexts, second); category = _lighting_bin(context)
        types[category, kind] += 1; values[category, kind, value] += 1; phases[category, kind, int(round(beat * 4)) % 16] += 1
        brightness[category, kind, 0] += max(0., min(10., level)); brightness[category, kind, 1] += 1
        rgb = _color(event)
        if rgb is not None:
            colors[rgb] += 1; conditional_colors[category][rgb] += 1
            old = previous_colors.get(kind)
            if old and old != rgb: transitions[(old, rgb)] += 1
            previous_colors[kind] = rgb
        events.append({'time': second, 'type': kind, 'value': value, 'brightness': max(0., min(10., level)), 'color': rgb})
        if kind == 5: boosts[category, int(bool(value))] += 1
    for event in raw.get('colorBoostBeatmapEvents', []):
        try:
            second = clock.time(float(event.get('b', 0))) + offset
            category = _lighting_bin(sample_context(times, contexts, second)); enabled = bool(event.get('o', False))
        except (TypeError, ValueError): malformed += 1; continue
        boosts[category, int(enabled)] += 1
        events.append({'time': second, 'type': 5, 'value': int(enabled), 'brightness': 1., 'color': None})
    unsupported = {key: len(raw.get(key, [])) for key in EVENT_BOXES}
    unsupported['customEvents'] = len((raw.get('customData', raw.get('_customData', {})) or {}).get('customEvents',
                                      (raw.get('_customData', {}) or {}).get('_customEvents', [])))
    return {'types': types, 'values': values, 'phases': phases, 'brightness': brightness, 'boosts': boosts,
            'events': sorted(events, key=lambda e: e['time']), 'colors': colors, 'conditionalColors': conditional_colors,
            'colorTransitions': transitions, 'unsupported': unsupported, 'malformed': malformed}


def _movement_target(notes, previous, start, end, mean_nps):
    if not notes: return [0., 0., 0., 0., 0.], np.zeros(43)
    travels, angular, histories, positions, direction_hist = [], [], {}, collections.Counter(), np.zeros(9)
    for note in notes:
        hand = note['c']; old = histories.get(hand, previous.get(hand))
        if old:
            travels.append(float(np.hypot(note['x'] - old['x'], note['y'] - old['y'])) / 3.)
            angular.append(abs((ANGLES[note['d']] - ANGLES[old['d']] + 180) % 360 - 180) / 180)
        histories[hand] = note; positions[round(note['b'], 5)] += 1; direction_hist[note['d']] += 1
    probabilities = direction_hist / max(direction_hist.sum(), 1)
    entropy = -float(np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))) / math.log(9)
    complex_cuts = sum(n['d'] in [2, 3, 4, 5, 6, 7] for n in notes) / len(notes)
    complexity = .60 * complex_cuts + .40 * float(np.mean(angular)) if angular else complex_cuts
    double_fraction = sum(v == 2 for v in positions.values()) / max(1, len(positions))
    targets = [len(notes) / max(end - start, .1) / max(mean_nps, .01), float(np.mean(travels)) if travels else 0., complexity, double_fraction, entropy]
    signature = np.zeros(43, dtype=np.float64)
    for note in notes:
        pos = min(15, max(0, int((note['seconds'] - start) / max(end - start, .1) * 16)))
        signature[pos] += 1
        signature[16 + note['c'] * 12 + note['x'] * 3 + note['y']] += 1
    signature[40:] = [targets[1], complexity, double_fraction]
    signature /= max(float(np.linalg.norm(signature)), 1e-8)
    return targets, signature


def _archive_features(path, folder, cancel_callback=None):
    archive = read_archive(path); times, contexts = _read_contexts(path, archive, folder)
    X, Y, origins = [], [], []
    totals = {'types': np.zeros((9, 32), np.int64), 'values': np.zeros((9, 32, 16), np.int64),
              'phases': np.zeros((9, 32, 16), np.int64), 'brightness': np.zeros((9, 32, 2)), 'boosts': np.zeros((9, 2), np.int64)}
    colors, transitions, conditional_colors, unsupported = collections.Counter(), collections.Counter(), [collections.Counter() for _ in range(9)], collections.Counter()
    event_count, malformed, phrase_boundaries, repeated_phrases = 0, 0, 0, 0
    native_objects, excluded_objects = collections.Counter(), collections.Counter()
    lighting_shows = set()
    light_cache, phrase_audio_cache = {}, {}
    with zipfile.ZipFile(path) as z:
        entries = {pathlib.PurePosixPath(n.replace('\\', '/')).name.lower(): n for n in z.namelist()}
        for chart_index, chart in enumerate(archive['maps']):
            check_cancel(cancel_callback)
            filename = entries.get(pathlib.PurePosixPath(chart['name']).name.lower())
            if not filename: continue
            raw = _json_entry(z, filename)
            if chart.get('nativeV4'):
                from .formats import decode_v4
                raw=decode_v4(raw,_json_entry(z,chart['lightshowEntry']))
            bpm_events = _valid_events(chart); clock = BeatClock(chart['bpm'], bpm_events); offset = chart.get('offset', 0.)
            clock_key = (chart['bpm'], offset, tuple((e['b'], e['m']) for e in bpm_events))
            light_signature = hashlib.sha256(json.dumps([raw.get('basicBeatmapEvents', raw.get('_events', [])),
                                                          raw.get('colorBoostBeatmapEvents', []), [raw.get(k, []) for k in EVENT_BOXES],
                                                          (raw.get('customData', raw.get('_customData', {})) or {}).get('customEvents',
                                                           (raw.get('_customData', {}) or {}).get('_customEvents', []))], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            light_key = (clock_key, light_signature)
            if light_key not in light_cache: light_cache[light_key] = _light_data(raw, clock, offset, times, contexts)
            light = light_cache[light_key]
            if light_signature not in lighting_shows:
                for key in totals: totals[key] += light[key]
                colors.update(light['colors']); transitions.update(light['colorTransitions']); unsupported.update(light['unsupported'])
                for i in range(9): conditional_colors[i].update(light['conditionalColors'][i])
                event_count += len(light['events']); malformed += light['malformed']; lighting_shows.add(light_signature)
            object_times, object_excluded = _object_times(raw, clock, offset)
            native_objects.update({k: len(v) for k, v in object_times.items()}); excluded_objects.update(object_excluded)
            from .diagnostics import _physical
            nonphysical = set()
            for note in raw.get('colorNotes', raw.get('_notes', [])):
                if not _physical(note):
                    try: nonphysical.add((round(float(note.get('b', note.get('_time', 0))), 5), int(note.get('c', note.get('_type', 0)))))
                    except (TypeError, ValueError): pass
            notes = [{**n, 'seconds': clock.time(n['b']) + offset} for n in chart['notes'] if (round(n['b'], 5), n['c']) not in nonphysical]
            if not notes: continue
            duration = max(float(times[-1]) if len(times) else 0., notes[-1]['seconds'], 1.)
            mean_nps = len(notes) / duration; rank = RANK.get(chart.get('difficulty'), 9)
            note_times = [n['seconds'] for n in notes]; light_times = [e['time'] for e in light['events']]
            end_beat = max(notes[-1]['b'], clock.beat(duration - offset))
            previous = {}; previous_signature = None; audio_signatures, map_signatures = [], []
            for beat in range(0, max(1, math.ceil(end_beat)), 4):
                if beat % 256 == 0: check_cancel(cancel_callback)
                start, end = max(0., clock.time(beat) + offset), min(duration, clock.time(beat + 4) + offset)
                if end <= start: continue
                lo, hi = bisect.bisect_left(note_times, start), bisect.bisect_left(note_times, end)
                phrase_notes = notes[lo:hi]
                targets, signature = _movement_target(phrase_notes, previous, start, end, mean_nps)
                for n in phrase_notes: previous[n['c']] = n
                phrase_key = (clock_key, round(duration, 4), beat)
                if phrase_key not in phrase_audio_cache:
                    feature = _phrase_feature(times, contexts, start, end, rank, duration, chart['bpm'])
                    query = np.stack([sample_context(times, contexts, t)[[0, 2]] for t in np.linspace(start, end, 16, endpoint=False)]).ravel()
                    query /= max(float(np.linalg.norm(query)), 1e-8)
                    phrase_audio_cache[phrase_key] = (feature, query)
                feature, query = phrase_audio_cache[phrase_key]
                feature = feature.copy(); feature[8] = rank / 9
                # Repeat evidence compares measured audio attack/energy shape,
                # while variation compares the human movement/rhythm signature.
                similarities = np.asarray(audio_signatures) @ query if audio_signatures else np.array([])
                match = int(np.argmax(similarities)) if len(similarities) and similarities.max() > .97 and query.any() else None
                variation = 1 - float(signature @ map_signatures[match]) if match is not None and phrase_notes else 0.
                if match is not None: repeated_phrases += 1
                audio_signatures.append(query); map_signatures.append(signature)
                # Every phrase participates, including breaks with no notes.
                a, b = bisect.bisect_left(light_times, start), bisect.bisect_left(light_times, end)
                lighting = light['events'][a:b]; count = len(lighting)
                fades = sum(e['value'] in [3, 7] and e['type'] < 5 for e in lighting) / max(count, 1)
                boosts = sum(e['type'] == 5 for e in lighting) / max(count, 1)
                level = float(np.mean([e['brightness'] for e in lighting])) if lighting else 0.
                object_rates = [(bisect.bisect_left(object_times[k], end) - bisect.bisect_left(object_times[k], start)) / (end - start)
                                for k in ['bomb', 'wall', 'arc', 'chain']]
                Y.append(targets + [variation, count / (end - start), fades, boosts, level] + object_rates); X.append(feature)
                origins.append({'chart': chart['name'], 'difficulty': chart['difficulty'], 'start': round(start, 4), 'end': round(end, 4)})
                if previous_signature is not None: phrase_boundaries += 1
                previous_signature = signature
    metadata = {'archive': pathlib.Path(path).name, 'sha256': archive['sha256'], 'song': archive['songName'],
                'maps': len(archive['maps']), 'physicalNotes': sum(len(c['notes']) for c in archive['maps']), 'excluded': archive['excluded'],
                'audioAvailable': bool(len(times)), 'lightingEvents': event_count, 'malformedLightingEvents': malformed,
                'uniqueLightingShows': len(lighting_shows), 'nativeObjects': dict(native_objects), 'excludedObjects': dict(excluded_objects),
                'phraseBoundaries': phrase_boundaries, 'repeatedAudioPhrases': repeated_phrases, 'unsupported': dict(unsupported),
                'colors': [[list(rgb), count] for rgb, count in colors.most_common(64)],
                'conditionalColors': [[[list(rgb), count] for rgb, count in values.most_common(24)] for values in conditional_colors],
                'colorTransitions': [[list(old), list(new), count] for (old, new), count in transitions.most_common(48)], 'origins': origins}
    return np.asarray(X, np.float32).reshape(-1, len(FEATURES)), np.asarray(Y, np.float32).reshape(-1, len(TARGETS)), totals, metadata


def _object_times(raw, clock, offset):
    """Count native, physical object heads; never label animated decorations valid."""
    from .diagnostics import _physical
    kinds = {'bomb': raw.get('bombNotes', [n for n in raw.get('_notes', []) if n.get('_type') == 3]),
             'wall': raw.get('obstacles', raw.get('_obstacles', [])), 'arc': raw.get('sliders', raw.get('_sliders', [])),
             'chain': raw.get('burstSliders', [])}
    result = {k: [] for k in kinds}; exclusions = collections.Counter()
    for kind, objects in kinds.items():
        for obj in objects:
            cd = obj.get('customData', obj.get('_customData', {})) or {}
            if not _physical(obj) or any(k in cd for k in ['animation', '_animation', 'position', '_position', 'localRotation', '_localRotation']):
                exclusions['nonPhysicalOrTransformed'] += 1; continue
            try:
                beat = float(obj.get('b', obj.get('_time', obj.get('_headTime', 0))))
                if not math.isfinite(beat): raise ValueError()
                for field in ['x', 'tx']:
                    if field in obj and not 0 <= float(obj[field]) <= 3: raise ValueError()
                for field in ['y', 'ty']:
                    if field in obj and not 0 <= float(obj[field]) <= 2: raise ValueError()
                result[kind].append(clock.time(beat) + offset)
            except (TypeError, ValueError): exclusions['outsideNativeGridOrMalformed'] += 1
    return {k: sorted(v) for k, v in result.items()}, dict(exclusions)


def train_composition(dataset_folder, model_dir, progress=None, cancel_callback=None, max_samples=24000):
    """Learn complete-song phrase targets and lighting; checkpoint only on success."""
    started = time.perf_counter(); folder = pathlib.Path(model_dir); folder.mkdir(parents=True, exist_ok=True)
    cache = folder / 'composition-cache'; cache.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(dataset_folder).glob('*.zip')); extractor = _extractor_version()
    X_parts, Y_parts, sources, provenance, errors, all_origins = [], [], [], [], [], []
    totals = {'types': np.zeros((9, 32), np.int64), 'values': np.zeros((9, 32, 16), np.int64),
              'phases': np.zeros((9, 32, 16), np.int64), 'brightness': np.zeros((9, 32, 2)), 'boosts': np.zeros((9, 2), np.int64)}
    colors, transitions, conditional_colors = collections.Counter(), collections.Counter(), [collections.Counter() for _ in range(9)]
    unsupported, native_objects, exclusions = collections.Counter(), collections.Counter(), collections.Counter()
    cache_hits = 0
    for index, path in enumerate(files):
        check_cancel(cancel_callback)
        if progress: progress({'phase': 'composition', 'progress': index / max(1, len(files)) * .85,
                               'message': f'Learning whole-song phrases and lighting: {path.name}'})
        try:
            sha = _digest(path); cached = cache / (sha[:24] + '-' + extractor + '.npz')
            if cached.exists():
                with np.load(cached, allow_pickle=False) as data:
                    x, y = data['X'].copy(), data['Y'].copy(); meta = json.loads(str(data['metadata']))
                    values = {k: data[k].copy() for k in totals}
                cache_hits += 1; status = 'hit'
            else:
                x, y, values, meta = _archive_features(path, folder, cancel_callback)
                check_cancel(cancel_callback)
                pending = cached.with_suffix('.pending.npz')
                np.savez_compressed(pending, X=x, Y=y, metadata=json.dumps(meta), **values); pending.replace(cached)
                status = 'created'
            if x.shape != (len(y), len(FEATURES)) or y.shape[1] != len(TARGETS): raise ValueError('Invalid composition feature-cache shape')
            keep = np.isfinite(x).all(1) & np.isfinite(y).all(1)
            x, y = x[keep], y[keep]
            if not len(x): raise ValueError('No complete-song phrase examples')
            # Archive-held-out splitting avoids leakage across difficulties in
            # one song; SHA membership is stable if the dataset order changes.
            source_index = len(provenance)
            X_parts.append(x); Y_parts.append(y); sources.extend([source_index] * len(x))
            all_origins.extend([origin for origin, yes in zip(meta.get('origins', [{}] * len(keep)), keep) if yes])
            for key in totals: totals[key] += values[key]
            colors.update({tuple(rgb): count for rgb, count in meta.get('colors', [])})
            transitions.update({(tuple(old), tuple(new)): count for old, new, count in meta.get('colorTransitions', [])})
            for i, palette in enumerate(meta.get('conditionalColors', [])): conditional_colors[i].update({tuple(rgb): count for rgb, count in palette})
            unsupported.update(meta.get('unsupported', {})); native_objects.update(meta.get('nativeObjects', {})); exclusions.update(meta.get('excludedObjects', {}))
            provenance.append({k: v for k, v in meta.items() if k not in ['origins', 'colors', 'conditionalColors', 'colorTransitions']} |
                              {'archive': path.name, 'featureCache': status, 'phraseExamples': len(x), 'validation': int(sha[:8], 16) % 5 == 0})
        except Exception as exc:
            from training_runtime import TrainingCancelled
            if isinstance(exc, TrainingCancelled): raise
            errors.append({'archive': path.name, 'error': str(exc)})
    if not X_parts: raise ValueError('No usable whole-song human-map examples')
    check_cancel(cancel_callback)
    X = np.concatenate(X_parts); Y = np.concatenate(Y_parts); source_ids = np.asarray(sources)
    train = np.array([not provenance[i]['validation'] for i in source_ids])
    if not train.any(): train[:] = True
    validation = ~train
    lo, hi = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), np.array([4., 1., 1., 1., 1., 1., 200., 1., 1., 10., 100., 100., 100., 100.])
    unclipped = Y.copy(); Y = np.clip(Y, lo, hi)
    mean, std = X[train].mean(0), np.maximum(X[train].std(0), .05)
    standardized = (X - mean) / std
    design = np.c_[np.ones(len(X)), standardized]
    regularization = np.eye(design.shape[1]) * max(2., len(X[train]) * .005); regularization[0, 0] = .001
    coefficients = np.linalg.solve(design[train].T @ design[train] + regularization, design[train].T @ Y[train])
    validation_metrics = {}
    if validation.any():
        prediction = design[validation] @ coefficients
        baseline = Y[train].mean(0)
        for i, target in enumerate(TARGETS):
            validation_metrics[target] = {'rmse': round(float(np.sqrt(np.mean((prediction[:, i] - Y[validation, i]) ** 2))), 5),
                                          'constantBaselineRmse': round(float(np.sqrt(np.mean((baseline[i] - Y[validation, i]) ** 2))), 5), 'units': 'target units'}
    if progress: progress({'phase': 'composition', 'progress': .92, 'message': f'Fitting {len(X):,} phrase examples; collecting lighting distributions'})
    # Nearest-phrase evidence is a balanced, deterministic subset. Store goals,
    # never actual note sequences or a chart's reusable block layout.
    rng = np.random.default_rng(29017); selected = []
    per_source = max(1, int(max_samples) // max(1, len(provenance)))
    for source_index in range(len(provenance)):
        ids = np.flatnonzero(source_ids == source_index)
        selected.extend(rng.choice(ids, min(per_source, len(ids)), replace=False).tolist())
    remaining = min(max(0, int(max_samples) - len(selected)), len(X) - len(selected))
    if remaining:
        ids = np.setdiff1d(np.arange(len(X)), np.array(selected), assume_unique=True)
        selected.extend(rng.choice(ids, remaining, replace=False).tolist())
    selected = np.asarray(sorted(selected), dtype=int)
    report = {'state': 'trained', 'version': VERSION, 'model': 'ArcForge whole-song composition and lighting learner',
              'method': 'Archive-held-out ridge regression + nearest human phrase goals + audio-conditioned event distributions',
              'archiveCount': len(provenance), 'mapCount': sum(p['maps'] for p in provenance), 'phraseExamples': len(X),
              'audioArchives': sum(p['audioAvailable'] for p in provenance), 'lightingEvents': sum(p['lightingEvents'] for p in provenance),
              'uniqueLightingShows': sum(p.get('uniqueLightingShows', 0) for p in provenance),
              'nativeObjects': dict(native_objects), 'excludedObjects': dict(exclusions), 'unsupportedObjects': dict(unsupported),
              'phraseBoundaries': sum(p['phraseBoundaries'] for p in provenance), 'repeatedAudioPhrases': sum(p['repeatedAudioPhrases'] for p in provenance),
              'featureCacheHits': cache_hits, 'trainingArchives': sum(not p['validation'] for p in provenance),
              'validationArchives': sum(p['validation'] for p in provenance), 'validationPhrases': int(validation.sum()),
              'heldOutMetrics': validation_metrics, 'clippedExtremeTargets': int(np.any(Y != unclipped, axis=1).sum()),
              'nearestPhraseSamples': len(selected), 'trainingSeconds': round(time.perf_counter() - started, 3),
              'provenance': provenance, 'errors': errors,
              'limitations': ['Learned targets and event distributions do not certify physical gameplay or musical quality.',
                              'Native event-box groups and custom animation events are counted but their filters/transforms are not learned.',
                              'Style labels are not provided by the human corpus; named styles remain user goals.',
                              'Source instrument identity and vocal semantics are not training labels. This model uses measured audio context.',
                              'RGB statistics keep the 64 most common colors per archive; they do not reproduce complete authored lightshows.']}
    model = {'version': VERSION, 'extractorFingerprint': extractor, 'features': FEATURES, 'targets': TARGETS,
             'featureMean': mean.tolist(), 'featureStd': std.tolist(), 'coefficients': coefficients.tolist(),
             'samples': {'features': X[selected].tolist(), 'targets': Y[selected].tolist(), 'sources': source_ids[selected].tolist(),
                         'origins': [all_origins[i] for i in selected]},
             'lighting': {**{k: v.tolist() for k, v in totals.items()},
                          'commonColors': [{'rgb': list(rgb), 'count': count} for rgb, count in colors.most_common(48)],
                          'conditionalColors': [[{'rgb': list(rgb), 'count': count} for rgb, count in values.most_common(24)] for values in conditional_colors],
                          'colorTransitions': [{'from': list(old), 'to': list(new), 'count': count} for (old, new), count in transitions.most_common(48)]},
             'report': report}
    check_cancel(cancel_callback)
    pending = folder / 'composition.pending.json'; pending.write_text(json.dumps(model, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    compact = {k: v for k, v in report.items() if k != 'provenance'} | {'provenanceFile': 'composition.json'}
    report_pending = folder / 'composition-report.pending.json'; report_pending.write_text(json.dumps(compact, indent=2, allow_nan=False), encoding='utf-8')
    pending.replace(folder / 'composition.json')
    report_pending.replace(folder / 'composition-report.json')
    if progress: progress({'phase': 'composition', 'progress': 1., 'message': 'Whole-song composition and lighting model saved'})
    return report


@functools.lru_cache(maxsize=3)
def _load_model(path, modified):
    raw = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    if raw.get('version') != VERSION: raise ValueError('Unsupported composition model version')
    raw['_samples'] = np.asarray(raw['samples']['features'], dtype=np.float32)
    raw['_targets'] = np.asarray(raw['samples']['targets'], dtype=np.float32)
    raw['_mean'] = np.asarray(raw['featureMean']); raw['_std'] = np.asarray(raw['featureStd'])
    raw['_normalized'] = (raw['_samples'] - raw['_mean']) / raw['_std']
    raw['_coefficients'] = np.asarray(raw['coefficients'])
    return raw


def composition_status(model_dir):
    path = pathlib.Path(model_dir) / 'composition.json'
    if not path.exists(): return {'ready': False, 'state': 'untrained', 'model': 'Whole-song composition and lighting learner'}
    try:
        model = _load_model(str(path.resolve()), path.stat().st_mtime_ns)
        return model['report'] | {'ready': True, 'extractorFingerprint': model['extractorFingerprint']}
    except (ValueError, KeyError, OSError) as exc: return {'ready': False, 'state': 'invalid', 'error': str(exc)}


@functools.lru_cache(maxsize=4)
def _audio_context(path, modified): return audio_context(path)


def _contexts(analysis):
    path = pathlib.Path(analysis.get('audioPath', ''))
    if path.is_file(): return _audio_context(str(path.resolve()), path.stat().st_mtime_ns)
    frames = analysis.get('frames', [])
    return np.asarray([f['time'] for f in frames]), np.asarray([[f.get(k, 0) for k in ['energy', 'bass', 'onset', 'high']] for f in frames], dtype=np.float32).reshape(-1, 4)


def _distribution(counts):
    total = float(np.sum(counts))
    return [float(v / total) for v in counts] if total > 0 else [0.] * len(counts)


def composition_profile(model_dir, analysis, second, config=None):
    """Predict phrase goals and lighting choices; caller keeps its timing/guards."""
    config = config or {}; path = pathlib.Path(model_dir) / 'composition.json'
    if not path.exists(): return {'ready': False, 'method': 'No whole-song composition model trained'}
    try: model = _load_model(str(path.resolve()), path.stat().st_mtime_ns)
    except (ValueError, KeyError, OSError): return {'ready': False, 'method': 'Composition model is invalid; preserve legacy generation'}
    duration = float(analysis.get('duration', 1)); bpm = float(config.get('bpm') or analysis.get('bpm', 120)); grid = analysis.get('beatTimes') or []
    if len(grid) > 4:
        index = max(0, bisect.bisect_right(grid, second) - 1); bar = index // 4 * 4
        start = grid[min(bar, len(grid) - 1)]; end = grid[bar + 4] if bar + 4 < len(grid) else min(duration, start + 240 / bpm)
    else:
        span = 240 / bpm; start = max(0., math.floor((second - float(analysis.get('offset', 0))) / span) * span + float(analysis.get('offset', 0))); end = min(duration, start + span)
    if end <= start: end = start + .10
    times, contexts = _contexts(analysis)
    feature = np.asarray(_phrase_feature(times, contexts, start, end, int(config.get('rank', 9)), duration, bpm))
    normalized = (feature - model['_mean']) / model['_std']
    prediction = np.r_[1., normalized] @ model['_coefficients']
    distance = np.sqrt(np.mean((model['_normalized'] - normalized) ** 2, axis=1))
    matches = np.argsort(distance)[:min(16, len(distance))]; weights = np.exp(-np.minimum(distance[matches], 30) * 2); weights /= max(float(weights.sum()), 1e-8)
    nearest = (model['_targets'][matches] * weights[:, None]).sum(0)
    goals = .60 * prediction + .40 * nearest
    caps = [(0., 2.2), (0., 1.), (0., 1.), (0., .7), (0., 1.), (0., 1.), (0., 24.), (0., 1.), (0., 1.), (0., 3.), (0., 16.), (0., 2.), (0., 8.), (0., 8.)]
    goals = {k: round(float(np.clip(v, lo, hi)), 5) for k, v, (lo, hi) in zip(TARGETS, goals, caps)}
    category = _lighting_bin(sample_context(times, contexts, second)); lighting = model['lighting']
    type_counts = np.asarray(lighting['types'])[category]
    if not type_counts.any(): type_counts = np.asarray(lighting['types']).sum(0)
    type_probs = _distribution(type_counts)
    event_values, phases, level = {}, {}, {}
    for kind, probability in enumerate(type_probs):
        if probability <= 0: continue
        value_counts = np.asarray(lighting['values'])[category, kind]
        if not value_counts.any(): value_counts = np.asarray(lighting['values'])[:, kind].sum(0)
        phase_counts = np.asarray(lighting['phases'])[category, kind]
        if not phase_counts.any(): phase_counts = np.asarray(lighting['phases'])[:, kind].sum(0)
        event_values[str(kind)] = {str(v): p for v, p in enumerate(_distribution(value_counts)) if p > 0}
        phases[str(kind)] = _distribution(phase_counts)
        levels = np.asarray(lighting['brightness'])[category, kind]
        level[str(kind)] = float(levels[0] / max(levels[1], 1))
    palette = lighting['conditionalColors'][category] or lighting['commonColors']
    provenance = model['report']['provenance']; sources = model['samples']['sources']
    return {'ready': True, 'modelVersion': VERSION, 'method': model['report']['method'],
            **{k: goals[k] for k in TARGETS[:6]}, 'confidence': round(float(1 / (1 + np.mean(distance[matches]))), 4),
            'confidenceMeaning': 'Nearest human phrase evidence closeness; not a calibrated quality score',
            'phraseStart': round(start, 5), 'phraseEnd': round(end, 5),
            'matches': [{'song': provenance[sources[i]]['song'], 'archive': provenance[sources[i]]['archive'],
                         **model['samples'].get('origins', [{}] * len(model['samples']['sources']))[i],
                         'distance': round(float(distance[i]), 4)} for i in matches[:5]],
            'objectRates': {kind: goals[target] for kind, target in [('bomb', 'bombRate'), ('wall', 'wallRate'), ('arc', 'arcRate'), ('chain', 'chainRate')]},
            'lighting': {'eventsPerSecond': goals['lightingEventsPerSecond'], 'fadeFraction': goals['fadeFraction'],
                         'boostFraction': goals['boostFraction'], 'meanBrightness': goals['meanBrightness'],
                         'contextBin': category, 'eventTypes': {str(k): p for k, p in enumerate(type_probs) if p > 0},
                         'eventValues': event_values, 'beatPhases': phases, 'brightnessByType': level,
                         'commonColors': palette, 'colorTransitions': lighting['colorTransitions']}}


__all__ = ['train_composition', 'composition_status', 'composition_profile']
