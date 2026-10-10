"""Read native map archives without executing supplied code or extracting paths."""
from __future__ import annotations
import collections, copy, gzip, hashlib, json, math, pathlib, shutil, threading, time, uuid, zipfile
from concurrent.futures import ThreadPoolExecutor

MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_ENTRY_BYTES = 150 * 1024 * 1024
ARCHIVE_PARSE_VERSION = '1.0'  # Bump only when native-note extraction semantics change.
CACHE_ROOT = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'archive-cache'
_CACHE_LOCK = threading.RLock()
_MEMORY = collections.OrderedDict()
_DIGESTS = {}


def _fingerprint(path):
    path = pathlib.Path(path).resolve(); stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime': stat.st_mtime_ns,
            'ctime': stat.st_ctime_ns, 'version': ARCHIVE_PARSE_VERSION}


def _cache_key(fingerprint):
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (uuid.uuid4().hex + '.pending.json')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')), 'utf-8'); temporary.replace(path)
    finally: temporary.unlink(missing_ok=True)


def file_digest(path):
    """Rehash only changed files; never reread every OGG inside every warm ZIP."""
    fingerprint = _fingerprint(path); key = _cache_key(fingerprint)
    with _CACHE_LOCK:
        if key in _DIGESTS: return _DIGESTS[key]
    record = CACHE_ROOT / (key + '.summary.json')
    if record.is_file():
        try:
            data = json.loads(record.read_text('utf-8'))
            if data['fingerprint'] == fingerprint:
                digest = data['summary']['sha256']
                if isinstance(digest, str) and len(digest) == 64:
                    with _CACHE_LOCK: _DIGESTS[key] = digest
                    return digest
        except (ValueError, OSError, KeyError): pass
    hasher = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): hasher.update(block)
    if _fingerprint(path) != fingerprint: raise ValueError('Archive changed while its content hash was being read')
    digest = hasher.hexdigest()
    with _CACHE_LOCK: _DIGESTS[key] = digest
    return digest


def _summary(data):
    return {key: value for key, value in data.items() if key != 'maps'} | {
        'mapCount': len(data['maps']), 'noteCount': sum(len(chart['notes']) for chart in data['maps']),
        'difficulties': [chart['difficulty'] for chart in data['maps']]}


def _json(z, name):
    entry = z.getinfo(name)
    if entry.file_size > 30 * 1024 * 1024:
        raise ValueError('Map JSON exceeds the 30 MB limit')
    from .formats import read_native_json
    return read_native_json(z.read(name))


def _parse_archive(path):
    path = pathlib.Path(path)
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError('Archive exceeds the 200 MB limit')
    with zipfile.ZipFile(path) as z:
        if len(z.infolist())>2000 or sum(e.file_size for e in z.infolist())>1024*1024*1024:
            raise ValueError('Archive exceeds the expanded size or entry limit')
        entries = {}
        for e in z.infolist():
            p = pathlib.PurePosixPath(e.filename.replace('\\', '/'))
            if p.is_absolute() or '..' in p.parts or e.file_size > MAX_ENTRY_BYTES:
                raise ValueError('Unsafe archive path or oversized entry')
            if not e.is_dir():
                entries.setdefault(p.name.lower(), e.filename)
        if 'info.dat' not in entries:
            raise ValueError('No Info.dat found')
        info = _json(z, entries['info.dat'])
        version = str(info.get('_version', info.get('version', '2')))
        native_audio=None
        if version.startswith('4'):
            from .formats import info_v4_to_v2,audio_data_to_tempo
            original_info=info;info=info_v4_to_v2(original_info)
            audio_name=entries.get(pathlib.PurePosixPath(original_info['audio']['audioDataFilename']).name.lower())
            if not audio_name:raise ValueError('V4 archive is missing its shared AudioData file')
            native_audio=_json(z,audio_name);tempo=audio_data_to_tempo(native_audio)
            info['_beatsPerMinute']=tempo['bpm'];info['_songTimeOffset']=tempo['songTimeOffset']
        bpm = float(info.get('_beatsPerMinute', info.get('beatsPerMinute', 120)))
        if not 20 <= bpm <= 1000:
            raise ValueError('Invalid BPM')
        offset = float(info.get('_songTimeOffset', 0))
        maps, exclusions = [], collections.Counter()
        light_rates, colors, event_types = [], collections.Counter(), collections.Counter()
        for group in info.get('_difficultyBeatmapSets', []):
            mode = group.get('_beatmapCharacteristicName', 'Standard')
            if mode != 'Standard':
                continue
            for diff in group.get('_difficultyBeatmaps', []):
                filename = str(diff.get('_beatmapFilename', ''))
                name = entries.get(pathlib.PurePosixPath(filename).name.lower())
                if not name:
                    continue
                raw = _json(z, name)
                lightshow_name=None
                if str(raw.get('version','')).startswith('4'):
                    from .formats import decode_v4
                    if native_audio is None:raise ValueError('Native V4 difficulty needs Info4 and AudioData')
                    lightshow_name=entries.get(pathlib.PurePosixPath(diff.get('_lightshowFilename','')).name.lower())
                    if not lightshow_name:raise ValueError('V4 archive is missing its lightshow file')
                    raw=decode_v4(raw,_json(z,lightshow_name),native_audio,bpm)
                ns = raw.get('colorNotes')
                if ns is None:
                    ns = [{'b': n.get('_time'), 'c': n.get('_type'),
                           'x': n.get('_lineIndex'), 'y': n.get('_lineLayer'),
                           'd': n.get('_cutDirection'), 'customData': n.get('_customData', {})}
                          for n in raw.get('_notes', []) if n.get('_type') in [0, 1]]
                clean, seen = [], set()
                for n in sorted(ns, key=lambda n: (float(n.get('b', 0)), int(n.get('c', 0)))):
                    cd = n.get('customData', n.get('_customData', {})) or {}
                    if n.get('r',0):
                        exclusions['rotatedGameplay']+=1;continue
                    if cd.get('fake', cd.get('_fake', False)) or cd.get('interactable', cd.get('_interactable', True)) is False:
                        exclusions['fakeOrNonInteractable'] += 1; continue
                    try:
                        note = {k: float(n[k]) if k == 'b' else int(n[k]) for k in ['b', 'c', 'x', 'y', 'd']}
                    except (TypeError, ValueError, KeyError):
                        exclusions['malformed'] += 1; continue
                    if not math.isfinite(note['b']) or note['b'] < 0 or note['c'] not in [0, 1] or not 0 <= note['x'] <= 3 or not 0 <= note['y'] <= 2 or not 0 <= note['d'] <= 8:
                        exclusions['outsideNativeGrid'] += 1; continue
                    # Noodle coordinates override the ordinary grid. Such objects are not reliable
                    # physical pattern examples without evaluating the map's animation system.
                    if any(k in cd for k in ['position', '_position', 'localRotation', '_localRotation']):
                        exclusions['transformedGameplay'] += 1; continue
                    key = (round(note['b'], 5), note['c'])
                    if key in seen:
                        exclusions['sameHandStacksOrDuplicates'] += 1; continue
                    seen.add(key); clean.append(note)
                if len(clean) < 24:
                    exclusions['insufficientNotes'] += len(clean); continue
                light_events = raw.get('basicBeatmapEvents', raw.get('_events', []))
                active_end = max(clean[-1]['b'], 1) * 60 / bpm
                light_rates.append(len(light_events) / active_end)
                for event in light_events:
                    event_types[str(event.get('et', event.get('_type', 0)))] += 1
                    color = (event.get('customData', event.get('_customData', {})) or {}).get('color')
                    if color is None: color = (event.get('customData', event.get('_customData', {})) or {}).get('_color')
                    if isinstance(color, list) and len(color) >= 3 and all(isinstance(v, (int, float)) for v in color[:3]):
                        colors[tuple(round(float(v), 2) for v in color[:3])] += 1
                maps.append({'name': filename, 'difficulty': diff.get('_difficulty', 'ExpertPlus'),
                             'notes': clean, 'bpmEvents': raw.get('bpmEvents', []), 'bpm': bpm,
                             **({'lightshowEntry':lightshow_name,'nativeV4':True} if lightshow_name else {}),
                             'offset': offset, 'source': str(path), 'songName': info.get('_songName', path.stem)})
        song_name = str(info.get('_songFilename', 'song.ogg'))
        song_entry = entries.get(pathlib.PurePosixPath(song_name).name.lower())
        return {'path': str(path), 'sha256': file_digest(path),
                'songName': info.get('_songName', path.stem), 'bpm': bpm, 'maps': maps,
                'audioEntry': song_entry, 'excluded': dict(exclusions),
                'lighting': {'eventRatesPerSecond': light_rates, 'eventTypes': dict(event_types),
                             'commonColors': [{'rgb': list(k), 'count': v} for k, v in colors.most_common(12)]}}


def read_archive(path):
    """Parse once per file revision; reuse safe local JSON caches afterwards."""
    fingerprint = _fingerprint(path); key = _cache_key(fingerprint)
    with _CACHE_LOCK:
        if key in _MEMORY:
            _MEMORY.move_to_end(key); return copy.deepcopy(_MEMORY[key])
    cache = CACHE_ROOT / (key + '.maps.json.gz')
    data = None
    if cache.is_file():
        try:
            with gzip.open(cache, 'rt', encoding='utf-8') as source: stored = json.load(source)
            if stored['fingerprint'] == fingerprint and isinstance(stored['data'].get('maps'), list): data = stored['data']
        except (ValueError, OSError, KeyError, TypeError): pass
    if data is None:
        data = _parse_archive(path)
        # A source modified while parsing is not committed as a valid cache.
        if _fingerprint(path) != fingerprint: raise ValueError('Archive changed while it was being read; import it again')
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = CACHE_ROOT / (uuid.uuid4().hex + '.pending.gz')
        try:
            with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as target:
                json.dump({'fingerprint': fingerprint, 'data': data}, target, ensure_ascii=False, separators=(',', ':'))
            temporary.replace(cache)
        finally: temporary.unlink(missing_ok=True)
    data['path'] = str(pathlib.Path(path))
    for chart in data['maps']: chart['source'] = data['path']
    _atomic_json(CACHE_ROOT / (key + '.summary.json'), {'fingerprint': fingerprint, 'summary': _summary(data)})
    with _CACHE_LOCK:
        _MEMORY[key] = data
        while len(_MEMORY) > 6: _MEMORY.popitem(last=False)
    return copy.deepcopy(data)


def archive_summary(path):
    """Warm dataset screens read tiny summaries, never chart JSON or ZIP audio."""
    fingerprint = _fingerprint(path); key = _cache_key(fingerprint)
    cache = CACHE_ROOT / (key + '.summary.json')
    if cache.is_file():
        try:
            saved = json.loads(cache.read_text('utf-8'))
            if saved['fingerprint'] == fingerprint:
                return saved['summary'] | {'path': str(pathlib.Path(path)), 'metadataCache': 'hit'}
        except (ValueError, OSError, KeyError, TypeError): pass
    # Library/training screens need metadata only. Do not compress and copy the
    # complete note arrays merely to display an archive count.
    data = _parse_archive(path)
    if _fingerprint(path) != fingerprint: raise ValueError('Archive changed while it was being read; import it again')
    summary = _summary(data)
    _atomic_json(cache, {'fingerprint': fingerprint, 'summary': summary})
    return summary | {'metadataCache': 'created'}


def scan_dataset(folder):
    folder = pathlib.Path(folder)
    archives, errors = [], []
    if not folder.is_dir():
        return {'folder': str(folder), 'archives': [], 'archiveCount': 0, 'mapCount': 0, 'noteCount': 0, 'errors': ['Dataset folder does not exist']}
    for path in sorted(folder.glob('*.zip')):
        try:
            archives.append(archive_summary(path))
        except Exception as e:
            errors.append({'archive': path.name, 'error': str(e)})
    return {'folder': str(folder), 'archives': archives, 'archiveCount': len(archives),
            'mapCount': sum(a['mapCount'] for a in archives), 'noteCount': sum(a['noteCount'] for a in archives), 'errors': errors}


def import_archive(source, folder):
    """Validate then copy a ZIP. Hash naming prevents traversal and silent overwrite."""
    folder = pathlib.Path(folder); folder.mkdir(parents=True, exist_ok=True)
    source_fingerprint = _fingerprint(source)
    digest = file_digest(source); destination = folder / (digest[:16] + '.zip')
    if destination.is_symlink(): raise ValueError('Training archive destination must not be a symbolic link')
    if destination.is_file() and file_digest(destination) == digest:
        # Equal content can reuse the already validated destination summary,
        # without reopening the source or materializing thousands of notes.
        data = archive_summary(destination)
    else:
        data = archive_summary(source)
        if data['sha256'] != digest: raise ValueError('Archive changed before it could be imported')
        if not data['mapCount']: raise ValueError('Archive contains no usable Standard training maps')
        temporary = folder / (uuid.uuid4().hex + '.pending.zip')
        try:
            shutil.copyfile(source, temporary)
            if _fingerprint(source) != source_fingerprint: raise ValueError('Archive changed while it was being copied')
            temporary.replace(destination)
        finally: temporary.unlink(missing_ok=True)
    if not data['mapCount']: raise ValueError('Archive contains no usable Standard training maps')
    # Seed the destination summary without reparsing the same map in corpus().
    fingerprint = _fingerprint(destination); key = _cache_key(fingerprint)
    cloned = data | {'path': str(destination)}
    _atomic_json(CACHE_ROOT / (key + '.summary.json'), {'fingerprint': fingerprint, 'summary': cloned})
    return {'path': str(destination), 'songName': data['songName'], 'noteCount': data['noteCount'], 'excluded': data['excluded']}


def import_references_fast(source_folder, destination_folder, progress=None, cancel_callback=None, workers=2):
    """Incremental import: unchanged source files bypass ZIP validation/copy.

    A persistent manifest records both source and destination file revisions.
    Modified/missing destinations are revalidated. Content-validation failures
    are reused until the file changes; temporary I/O failures are retried on the
    next import. Two workers bound CPU and disk contention.
    """
    from training_runtime import check_cancel
    started = time.perf_counter(); source_folder = pathlib.Path(source_folder)
    destination_folder = pathlib.Path(destination_folder); destination_folder.mkdir(parents=True, exist_ok=True)
    manifest_path = destination_folder / '.reference-imports.json'
    try: manifest = json.loads(manifest_path.read_text('utf-8'))
    except (ValueError, OSError): manifest = {'version': ARCHIVE_PARSE_VERSION, 'sources': {}}
    if manifest.get('version') != ARCHIVE_PARSE_VERSION: manifest = {'version': ARCHIVE_PARSE_VERSION, 'sources': {}}
    records = manifest.get('sources', {}); accepted, rejected, pending = [], [], []
    sources = sorted(source_folder.glob('*.zip')); skipped = 0
    for source in sources:
        check_cancel(cancel_callback); fingerprint = _fingerprint(source); name = str(source.resolve())
        old = records.get(name)
        if old and old.get('fingerprint') == fingerprint:
            # Earlier manifests treated every failure as permanent, including
            # sharing violations or sandbox/antivirus access failures. Require
            # an explicit permanent marker, so old failures get one fresh try.
            if 'rejected' in old and old.get('retryable') is False:
                rejected.append(old['rejected']); skipped += 1; continue
            result = old.get('accepted', {}); destination = pathlib.Path(result.get('path', ''))
            if destination.is_file() and _fingerprint(destination) == old.get('destinationFingerprint'):
                accepted.append(result | {'unchanged': True}); skipped += 1; continue
        pending.append((source, fingerprint, name))
    completed = 0
    def process(item):
        source, fingerprint, name = item; check_cancel(cancel_callback)
        try:
            result = import_archive(source, destination_folder)
            record = {'fingerprint': fingerprint, 'accepted': result, 'destinationFingerprint': _fingerprint(result['path'])}
            return name, record, result, None
        except Exception as error:
            from training_runtime import TrainingCancelled
            if isinstance(error, TrainingCancelled): raise
            # Invalid ZIP/JSON/map contents cannot improve without a revision.
            # OS access, sharing and disk failures can improve immediately.
            retryable = not isinstance(error, (ValueError, zipfile.BadZipFile, zipfile.LargeZipFile))
            rejection = {'name': source.name, 'reason': str(error), 'retryable': retryable}
            return name, {'fingerprint': fingerprint, 'rejected': rejection, 'retryable': retryable}, None, rejection
    with ThreadPoolExecutor(max_workers=max(1, min(4, int(workers)))) as pool:
        for name, record, result, rejection in pool.map(process, pending):
            check_cancel(cancel_callback); records[name] = record; completed += 1
            if result: accepted.append(result)
            if rejection: rejected.append(rejection)
            if progress: progress((skipped + completed) * 100 / max(len(sources), 1), f'Imported {completed} changed archives; skipped {skipped} unchanged')
    _atomic_json(manifest_path, {'version': ARCHIVE_PARSE_VERSION, 'sources': records})
    return {'accepted': accepted, 'rejected': rejected, 'unchanged': skipped, 'processed': completed,
            'sourceCount': len(sources), 'seconds': round(time.perf_counter() - started, 4),
            'message': f'{completed} changed archives processed; {skipped} unchanged archives reused.'}
