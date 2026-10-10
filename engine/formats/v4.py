"""Indexed native Beatmap 4.0/4.1 and Lightshow 4.0 conversion.

ArcForge's editing representation remains V3-shaped. V4-only lane rotation,
NJS events and easing are explicit extensions of that representation; a V3
export refuses unsupported downgrades. Timing is in the shared AudioData file,
never a made-up `bpmEvents` field in the V4 difficulty.

Implemented from the documented native schema and independently checked with
bsmap's published schema validator. No third-party conversion code is copied.
See https://bsmg.wiki/mapping/map-format.html and its linked format references.
"""
from __future__ import annotations

import copy
import gzip
import io
import json
import math
import pathlib
import re
import subprocess


class FormatError(ValueError):
    """A native file is malformed or cannot be converted without losing meaning."""

_MISSING=object()


def read_native_json(contents, max_bytes=30 * 1024 * 1024):
    """Read native plain/gzip JSON with a bounded expanded size, including BOM."""
    if len(contents) > max_bytes: raise FormatError('Map JSON exceeds the 30 MB limit')
    if contents.startswith(b'\x1f\x8b'):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(contents)) as stream:
                contents = stream.read(max_bytes + 1)
        except (OSError, EOFError) as exc: raise FormatError('Invalid gzip-compressed map JSON') from exc
        if len(contents) > max_bytes: raise FormatError('Expanded map JSON exceeds the 30 MB limit')
    try: result = json.loads(contents.decode('utf-8-sig'))
    except (ValueError, UnicodeError) as exc: raise FormatError('Invalid native map JSON') from exc
    if not isinstance(result, dict): raise FormatError('A native map file must contain a JSON object')
    _json(result)
    return result


def measure_loudness(audio_path):
    """Cached FFmpeg EBU R128 integrated LUFS, not an RMS proxy or guessed value."""
    import imageio_ffmpeg
    path=pathlib.Path(audio_path);stat=path.stat()
    fingerprint={'size':stat.st_size,'mtimeNs':stat.st_mtime_ns,'method':'ffmpeg-ebur128-integrated-v1'}
    cache=path.with_name('native-loudness.json')
    if cache.is_file():
        try:
            stored=json.loads(cache.read_text('utf-8'))
            if stored.get('fingerprint')==fingerprint:
                return float(_number(stored['integratedLufs'],'integrated LUFS'))
        except (ValueError,OSError,KeyError):pass
    result=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-nostats','-i',str(path),
                           '-af','ebur128','-f','null','-'],capture_output=True,timeout=240)
    values=re.findall(r'\bI:\s*(-?\d+(?:\.\d+)?)\s*LUFS',result.stderr.decode('utf-8','replace'))
    if result.returncode or not values:raise FormatError('Could not measure audio loudness for native V4 metadata')
    lufs=float(values[-1]);_number(lufs,'integrated LUFS')
    temporary=cache.with_suffix('.pending.json')
    temporary.write_text(json.dumps({'fingerprint':fingerprint,'integratedLufs':lufs},allow_nan=False),encoding='utf-8')
    temporary.replace(cache)
    return lufs


def _number(value, where, *, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FormatError(f'{where}: expected a finite number')
    if integer and value != int(value):
        raise FormatError(f'{where}: expected an integer')
    return value


def _json(value, where='document'):
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FormatError(f'{where}: invalid JSON value') from exc


def _array(document, key):
    value = document.get(key, [])
    if not isinstance(value, list) or any(not isinstance(v, dict) for v in value):
        raise FormatError(f'{key}: expected an array of objects')
    return value


def _lookup(document, key, index, where):
    if index is _MISSING:
        # Native V4 default serialization permits an omitted metadata array AND
        # omitted reference to resolve to the default metadata object.
        if key not in document:return {}
        index=0
    _number(index, where, integer=True)
    values = _array(document, key)
    if index < 0 or index >= len(values):
        raise FormatError(f'{where}: index {index} is outside {key} ({len(values)} entries)')
    return values[int(index)]


def _copy_custom(source, target):
    if 'customData' in source:
        if not isinstance(source['customData'], dict):
            raise FormatError('customData must be an object')
        target['customData'] = copy.deepcopy(source['customData'])
    return target


def _object_custom(reference, target):
    # Data and occurrence have independent ownership in native V4. Preserve both.
    if reference.get('customData'):
        target['_v4ObjectCustomData'] = copy.deepcopy(reference['customData'])
    return target


def _fields(value, numeric=(), integers=(), where='object'):
    for key in numeric:
        if key in value: _number(value[key], f'{where}.{key}')
    for key in integers:
        if key in value: _number(value[key], f'{where}.{key}', integer=True)
    if 'customData' in value and not isinstance(value['customData'], dict):
        raise FormatError(f'{where}.customData: expected an object')


def _ref_custom(source, target):
    if source.get('_v4ObjectCustomData'):
        target['customData'] = copy.deepcopy(source['_v4ObjectCustomData'])
    return target


def _empty_chart():
    return {'version': '3.3.0', 'bpmEvents': [], 'rotationEvents': [],
            'colorNotes': [], 'bombNotes': [], 'obstacles': [], 'sliders': [],
            'burstSliders': [], 'waypoints': [], 'basicBeatmapEvents': [],
            'colorBoostBeatmapEvents': [], 'lightColorEventBoxGroups': [],
            'lightRotationEventBoxGroups': [], 'lightTranslationEventBoxGroups': [],
            'vfxEventBoxGroups': [], '_fxEventsCollection': {'_fl': [], '_il': []},
            'basicEventTypesWithKeywords': {'d': []},
            'useNormalEventsAsCompatibleEvents': False}


def _untranslated_fields(difficulty, lightshow):
    """Unknown extension fields may be preserved unchanged, never guessed away."""
    known_difficulty={'version','customData','colorNotes','colorNotesData','bombNotes','bombNotesData',
        'obstacles','obstaclesData','chains','chainsData','arcs','arcsData','spawnRotations',
        'spawnRotationsData','njsEvents','njsEventData'}
    known_lightshow={'version','customData','waypoints','waypointsData','basicEvents','basicEventsData',
        'colorBoostEvents','colorBoostEventsData','eventBoxGroups','indexFilters','lightColorEventBoxes',
        'lightColorEvents','lightRotationEventBoxes','lightRotationEvents','lightTranslationEventBoxes',
        'lightTranslationEvents','fxEventBoxes','floatFxEvents','basicEventTypesWithKeywords','useNormalEventsAsCompatibleEvents'}
    paths=[]
    for name,document,known in [('beatmap',difficulty,known_difficulty),('lightshow',lightshow or {},known_lightshow)]:
        paths.extend(f'{name}.{key}' for key in document if key not in known)
    object_fields={
        'colorNotes':{'b','i','r'},'bombNotes':{'b','i','r'},'obstacles':{'b','i','r'},
        'chains':{'hb','hr','tb','tr','i','ci'},'arcs':{'hb','hr','hi','tb','tr','ti','ai'},
        'spawnRotations':{'b','i'},'njsEvents':{'b','i'},
        'colorNotesData':{'x','y','c','d','a'},'bombNotesData':{'x','y'},'obstaclesData':{'x','y','d','w','h'},
        'chainsData':{'tx','ty','c','s'},'arcsData':{'m','tm','a'},'spawnRotationsData':{'t','e','r'},'njsEventData':{'d','p','e'},
        'waypoints':{'b','i','r'},'waypointsData':{'x','y','d'},'basicEvents':{'b','i'},'basicEventsData':{'t','i','f'},
        'colorBoostEvents':{'b','i'},'colorBoostEventsData':{'b'},
        'indexFilters':{'f','p','t','r','c','n','s','l','d'},
        'lightColorEventBoxes':{'w','d','s','t','b','e'},'lightColorEvents':{'p','e','c','b','f','sb','sf'},
        'lightRotationEventBoxes':{'w','d','s','t','b','e','a','f'},'lightRotationEvents':{'p','e','l','r','d'},
        'lightTranslationEventBoxes':{'w','d','s','t','b','e','a','f'},'lightTranslationEvents':{'p','e','t'},
        'fxEventBoxes':{'w','d','s','t','b','e'},'floatFxEvents':{'p','e','v'},
        'eventBoxGroups':{'b','g','t','e'}}
    for name,document in [('beatmap',difficulty),('lightshow',lightshow or {})]:
        for key,known in object_fields.items():
            for index,item in enumerate(document.get(key,[])):
                paths.extend(f'{name}.{key}[{index}].{field}' for field in item if field not in known|{'customData'})
    return paths


def validate_v4(difficulty, lightshow=None, audio_data=None):
    """Validate shape and ALL metadata references, including nested light boxes.

    Native defaults are accepted. A missing array is empty; a reference into a
    missing metadata array is an error rather than silently becoming a zero note.
    Returns counts for diagnostics; raises FormatError on invalid input.
    """
    if not isinstance(difficulty, dict) or difficulty.get('version') not in ('4.0.0', '4.1.0'):
        raise FormatError('Expected native Beatmap 4.0.0 or 4.1.0')
    _json(difficulty)
    for key, numeric, integers in (
        ('colorNotesData', (), ('x', 'y', 'c', 'd', 'a')),
        ('bombNotesData', (), ('x', 'y')),
        ('obstaclesData', ('d',), ('x', 'y', 'w', 'h')),
        ('arcsData', ('m', 'tm'), ('a',)),
        ('chainsData', ('s',), ('tx', 'ty', 'c')),
        ('spawnRotationsData', ('r',), ('t','e')),
        ('njsEventData', ('d',), ('p', 'e')),
    ):
        for index, data in enumerate(_array(difficulty, key)):
            _fields(data, numeric, integers, f'{key}[{index}]')
            if key == 'colorNotesData' and data.get('c', 0) not in (-1, 0, 1):
                raise FormatError('Color metadata must use native color -1, 0 or 1')
    for objects, metadata, fields in (
        ('colorNotes', 'colorNotesData', ('b', 'r')),
        ('bombNotes', 'bombNotesData', ('b', 'r')),
        ('obstacles', 'obstaclesData', ('b', 'r')),
        ('spawnRotations', 'spawnRotationsData', ('b',)),
        ('njsEvents', 'njsEventData', ('b',)),
    ):
        _array(difficulty, metadata)
        for k, obj in enumerate(_array(difficulty, objects)):
            for field in fields: _number(obj.get(field, 0), f'{objects}[{k}].{field}')
            data = _lookup(difficulty, metadata, obj.get('i', _MISSING), f'{objects}[{k}].i')
            _fields(obj, where=f'{objects}[{k}]')
            if objects == 'colorNotes' and data.get('c', 0) not in (0, 1):
                raise FormatError('A color note must use hand 0 or 1')
            if objects == 'njsEvents' and difficulty['version'] != '4.1.0':
                raise FormatError('Native NJS events require Beatmap 4.1.0')
            if objects=='spawnRotations':
                if data.get('t',data.get('e',0)) not in (0,1):raise FormatError('Spawn rotation execution type must be early or late')
                if 't' in data and 'e' in data and data['t']!=data['e']:raise FormatError('Conflicting native/compatibility spawn rotation execution type')
            if objects == 'obstacles' and data.get('d', 0) < 0:
                raise FormatError('An obstacle duration cannot be negative')
    for k, obj in enumerate(_array(difficulty, 'arcs')):
        for field in ('hb', 'tb', 'hr', 'tr'): _number(obj.get(field, 0), f'arcs[{k}].{field}')
        if obj.get('tb', 0) < obj.get('hb', 0): raise FormatError('An arc ends before it starts')
        for key, field in (('arcsData', 'ai'), ('colorNotesData', 'hi'), ('colorNotesData', 'ti')):
            _lookup(difficulty, key, obj.get(field, _MISSING), f'arcs[{k}].{field}')
    for k, obj in enumerate(_array(difficulty, 'chains')):
        for field in ('hb', 'tb', 'hr', 'tr'): _number(obj.get(field, 0), f'chains[{k}].{field}')
        if obj.get('tb', 0) < obj.get('hb', 0): raise FormatError('A chain ends before it starts')
        _lookup(difficulty, 'colorNotesData', obj.get('i', _MISSING), f'chains[{k}].i')
        data = _lookup(difficulty, 'chainsData', obj.get('ci', _MISSING), f'chains[{k}].ci')
        if data.get('c', 0) < 1 or data.get('s', 0) <= 0:
            raise FormatError('A chain needs a positive slice count and squish')
    if lightshow is not None:
        if not isinstance(lightshow, dict) or lightshow.get('version') != '4.0.0':
            raise FormatError('Expected native Lightshow 4.0.0')
        _json(lightshow)
        for key, numeric, integers in (
            ('waypointsData', (), ('x', 'y', 'd')),
            ('basicEventsData', ('t', 'f'), ('i',)), ('colorBoostEventsData', (), ('b',)),
            ('indexFilters', ('l',), ('f', 'p', 't', 'r', 'c', 'n', 's', 'd')),
            ('lightColorEventBoxes', ('w', 's'), ('d', 't', 'b', 'e')),
            ('lightRotationEventBoxes', ('w', 's'), ('d', 't', 'b', 'e', 'a', 'f')),
            ('lightTranslationEventBoxes', ('w', 's'), ('d', 't', 'b', 'e', 'a', 'f')),
            ('fxEventBoxes', ('w', 's'), ('d', 't', 'b', 'e')),
            ('lightColorEvents', ('b', 'sb'), ('p', 'e', 'c', 'f', 'sf')),
            ('lightRotationEvents', ('r',), ('p', 'e', 'l', 'd')),
            ('lightTranslationEvents', ('t',), ('p', 'e')),
            ('floatFxEvents', ('v',), ('p', 'e')),
        ):
            for index, data in enumerate(_array(lightshow, key)):
                _fields(data, numeric, integers, f'{key}[{index}]')
        for objects, metadata in (('waypoints', 'waypointsData'), ('basicEvents', 'basicEventsData'),
                                  ('colorBoostEvents', 'colorBoostEventsData')):
            for k, obj in enumerate(_array(lightshow, objects)):
                _number(obj.get('b', 0), f'{objects}[{k}].b')
                _lookup(lightshow, metadata, obj.get('i', _MISSING), f'{objects}[{k}].i')
        for k, group in enumerate(_array(lightshow, 'eventBoxGroups')):
            _number(group.get('b', 0), f'eventBoxGroups[{k}].b')
            kind = group.get('t', 0)
            if kind==0 and not group.get('e'):continue
            if kind not in (1, 2, 3, 4): raise FormatError(f'Unsupported light event-box type {kind}')
            boxes, events = _BOX_TYPES[kind][1:3]
            for j, box in enumerate(_array(group, 'e')):
                _lookup(lightshow, 'indexFilters', box.get('f', _MISSING), f'eventBoxGroups[{k}].e[{j}].f')
                _lookup(lightshow, boxes, box.get('e', _MISSING), f'eventBoxGroups[{k}].e[{j}].e')
                for ref in _array(box, 'l'):
                    _number(ref.get('b', 0), 'light event relative beat')
                    _lookup(lightshow, events, ref.get('i', _MISSING), 'light event metadata reference')
    if audio_data is not None: audio_data_to_tempo(audio_data)
    return {key: len(difficulty.get(key, [])) for key in ('colorNotes', 'bombNotes', 'obstacles', 'arcs', 'chains', 'njsEvents')}


# V4 group discriminator -> (V3 field, V4 box-data field, V4 event-data field).
_BOX_TYPES = {
    1: ('lightColorEventBoxGroups', 'lightColorEventBoxes', 'lightColorEvents'),
    2: ('lightRotationEventBoxGroups', 'lightRotationEventBoxes', 'lightRotationEvents'),
    3: ('lightTranslationEventBoxGroups', 'lightTranslationEventBoxes', 'lightTranslationEvents'),
    4: ('vfxEventBoxGroups', 'fxEventBoxes', 'floatFxEvents'),
}


def decode_v4(difficulty, lightshow=None, audio_data=None, bpm=120):
    """Resolve native indexed objects into ArcForge's editable representation."""
    validate_v4(difficulty, lightshow, audio_data)
    result = _empty_chart()
    _copy_custom(difficulty, result)
    unknown=_untranslated_fields(difficulty,lightshow)
    if unknown:result['_v4UntranslatedFields']=unknown
    for objects, metadata, defaults in (
        ('colorNotes', 'colorNotesData', {'x': 0, 'y': 0, 'c': 0, 'd': 0, 'a': 0}),
        ('bombNotes', 'bombNotesData', {'x': 0, 'y': 0}),
        ('obstacles', 'obstaclesData', {'x': 0, 'y': 0, 'd': 0, 'w': 0, 'h': 0}),
    ):
        for obj in difficulty.get(objects, []):
            data = _lookup(difficulty, metadata, obj.get('i', _MISSING), objects)
            item = {**defaults, **copy.deepcopy(data), 'b': obj.get('b', 0)}
            if obj.get('r', 0): item['r'] = obj['r']
            result[objects].append(_object_custom(obj, item))
    for obj in difficulty.get('spawnRotations', []):
        data = _lookup(difficulty, 'spawnRotationsData', obj.get('i', _MISSING), 'spawn rotation')
        result['rotationEvents'].append(_object_custom(obj, _copy_custom(data, {
            'b': obj.get('b', 0), 'e': data.get('t',data.get('e', 0)), 'r': data.get('r', 0)})))
    for obj in difficulty.get('arcs', []):
        head = _lookup(difficulty, 'colorNotesData', obj.get('hi', _MISSING), 'arc head')
        tail = _lookup(difficulty, 'colorNotesData', obj.get('ti', _MISSING), 'arc tail')
        data = _lookup(difficulty, 'arcsData', obj.get('ai', _MISSING), 'arc metadata')
        item = {'b': obj.get('hb', 0), 'tb': obj.get('tb', 0), 'c': head.get('c', 0),
                'x': head.get('x', 0), 'y': head.get('y', 0), 'd': head.get('d', 0),
                'tx': tail.get('x', 0), 'ty': tail.get('y', 0), 'tc': tail.get('d', 0),
                'mu': data.get('m', 0), 'tmu': data.get('tm', 0),
                'm': data.get('a', 0)}
        if tail.get('c', 0) != head.get('c', 0): item['_v4TailColor'] = tail.get('c', 0)
        for field in ('hr', 'tr'):
            if obj.get(field, 0): item[field] = obj[field]
        if head.get('customData'): item['_v4HeadCustomData'] = copy.deepcopy(head['customData'])
        if tail.get('customData'): item['_v4TailCustomData'] = copy.deepcopy(tail['customData'])
        if head.get('a',0):item['_v4HeadAngle']=head['a']
        if tail.get('a',0):item['_v4TailAngle']=tail['a']
        result['sliders'].append(_object_custom(obj, _copy_custom(data, item)))
    for obj in difficulty.get('chains', []):
        head = _lookup(difficulty, 'colorNotesData', obj.get('i', _MISSING), 'chain head')
        data = _lookup(difficulty, 'chainsData', obj.get('ci', _MISSING), 'chain metadata')
        item = {'b': obj.get('hb', 0), 'tb': obj.get('tb', 0), 'c': head.get('c', 0),
                'x': head.get('x', 0), 'y': head.get('y', 0), 'd': head.get('d', 0),
                'tx': data.get('tx', 0), 'ty': data.get('ty', 0),
                'sc': data.get('c', 0), 's': data.get('s', 0)}
        for field in ('hr', 'tr'):
            if obj.get(field, 0): item[field] = obj[field]
        if head.get('customData'): item['_v4HeadCustomData'] = copy.deepcopy(head['customData'])
        if head.get('a',0):item['_v4HeadAngle']=head['a']
        result['burstSliders'].append(_object_custom(obj, _copy_custom(data, item)))
    if difficulty.get('njsEvents'):
        result['njsEvents'] = []
        for obj in difficulty['njsEvents']:
            data = _lookup(difficulty, 'njsEventData', obj.get('i', _MISSING), 'NJS metadata')
            result['njsEvents'].append(_object_custom(obj, _copy_custom(data, {
                'b': obj.get('b', 0), 'd': data.get('d', 0),
                'p': data.get('p', 0), 'e': data.get('e', 0)})))
    if lightshow is not None: _decode_lights(lightshow, result)
    if audio_data is not None: result['bpmEvents'] = audio_data_to_tempo(audio_data)['bpmEvents']
    return result


def _decode_lights(source, result):
    if 'customData' in source:
        result['_v4LightshowCustomData'] = copy.deepcopy(source['customData'])
    for objects, metadata, destination in (
        ('waypoints', 'waypointsData', 'waypoints'),
        ('basicEvents', 'basicEventsData', 'basicBeatmapEvents'),
        ('colorBoostEvents', 'colorBoostEventsData', 'colorBoostBeatmapEvents'),
    ):
        for obj in source.get(objects, []):
            data = _lookup(source, metadata, obj.get('i', _MISSING), objects)
            if objects == 'basicEvents': item = {'b': obj.get('b', 0), 'et': data.get('t', 0), 'i': data.get('i', 0), 'f': data.get('f', 0)}
            elif objects == 'colorBoostEvents': item = {'b': obj.get('b', 0), 'o': bool(data.get('b', 0))}
            else:
                item = {'b': obj.get('b', 0), 'x': data.get('x', 0), 'y': data.get('y', 0), 'd': data.get('d', 0)}
                if obj.get('r', 0): item['r'] = obj['r']
            result[destination].append(_object_custom(obj, _copy_custom(data, item)))
    for key in ('basicEventTypesWithKeywords', 'useNormalEventsAsCompatibleEvents'):
        if key in source: result[key] = copy.deepcopy(source[key])
    for group in source.get('eventBoxGroups', []):
        if group.get('t',0)==0:
            result.setdefault('_v4EmptyEventBoxGroups',[]).append(copy.deepcopy(group));continue
        kind = group.get('t', 0); destination, boxes, events = _BOX_TYPES[kind]
        out = _copy_custom(group, {'b': group.get('b', 0), 'g': group.get('g', 0), 'e': []})
        if kind == 4: out['t'] = 1  # V3 FxType.FLOAT, not the V4 box discriminator.
        for reference in group.get('e', []):
            data = _lookup(source, boxes, reference.get('e', _MISSING), 'light box')
            box = {'f': copy.deepcopy(_lookup(source, 'indexFilters', reference.get('f', _MISSING), 'light filter')),
                   'w': data.get('w', 0), 'd': data.get('d', 1), 't': data.get('t', 1),
                   'b': data.get('b', 0), 'i': data.get('e', 0)}
            box['r' if kind == 1 else 's'] = data.get('s', 0)
            if kind in (2, 3): box.update(a=data.get('a', 0), r=data.get('f', 0))
            _copy_custom(data, box); _object_custom(reference, box)
            sequence = []
            for event_reference in reference.get('l', []):
                event_data = _lookup(source, events, event_reference.get('i', _MISSING), 'light event')
                event = {'b': event_reference.get('b', 0)}
                if kind == 1:
                    event.update(c=event_data.get('c', 0), f=event_data.get('f', 0),
                                 s=event_data.get('b', 0), sb=event_data.get('sb', 0), sf=event_data.get('sf', 0),
                                 i=2 if event_data.get('p', 0) else 0 if event_data.get('e', 0) == -1 else 1)
                    # V3 color events only have instant/linear/extend transitions.
                    if event_data.get('e', 0) not in (-1, 0) or event_data.get('p', 0):
                        event['_v4Easing'] = event_data.get('e', 0)
                elif kind == 2: event.update(e=event_data.get('e', 0), l=event_data.get('l', 0), o=event_data.get('d', 0), p=event_data.get('p', 0), r=event_data.get('r', 0))
                elif kind == 3: event.update(e=event_data.get('e', 0), p=event_data.get('p', 0), t=event_data.get('t', 0))
                else: event.update(i=event_data.get('e', 0), p=event_data.get('p', 0), v=event_data.get('v', 0))
                _copy_custom(event_data, event); _object_custom(event_reference, event)
                if kind == 4:
                    sequence.append(len(result['_fxEventsCollection']['_fl']))
                    result['_fxEventsCollection']['_fl'].append(event)
                else: sequence.append(event)
            box['e' if kind == 1 else 'l'] = sequence
            out['e'].append(box)
        result[destination].append(out)


class _Tables:
    def __init__(self, document): self.document = document; self.indices = {}
    def add(self, key, value):
        serialized = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
        indices = self.indices.setdefault(key, {})
        if serialized not in indices:
            values = self.document.setdefault(key, [])
            indices[serialized] = len(values); values.append(value)
        return indices[serialized]


def _metadata(source, fields):
    return _copy_custom(source, {key: source.get(key, default) for key, default in fields.items()})


def encode_v4(chart):
    """Serialize an editable chart into REAL, separate indexed V4 native files."""
    if not isinstance(chart, dict) or not str(chart.get('version', '')).startswith('3'):
        raise FormatError('V4 export expects ArcForge editable V3-shaped chart data')
    if chart.get('_v4UntranslatedFields'):
        raise FormatError('This imported V4 chart contains unknown extension fields. Export it unchanged to preserve them; edited conversion is not supported: '+', '.join(chart['_v4UntranslatedFields'][:4]))
    _json(chart)
    difficulty = {'version': '4.1.0', 'colorNotes': [], 'colorNotesData': [],
                  'bombNotes': [], 'bombNotesData': [], 'obstacles': [], 'obstaclesData': [],
                  'arcs': [], 'arcsData': [], 'chains': [], 'chainsData': [],
                  'njsEvents': [], 'njsEventData': []}
    _copy_custom(chart, difficulty)
    tables = _Tables(difficulty)
    rotations = sorted(chart.get('rotationEvents', []), key=lambda e: (e.get('b', 0), e.get('e', 0)))
    def lane_rotation(beat):
        return int(round(sum(event.get('r', 0) for event in rotations if event.get('b', 0) < beat or event.get('b', 0) == beat and event.get('e', 0) == 0)))
    for objects, metadata, fields in (
        ('colorNotes', 'colorNotesData', {'x': 0, 'y': 0, 'c': 0, 'd': 0, 'a': 0}),
        ('bombNotes', 'bombNotesData', {'x': 0, 'y': 0}),
        ('obstacles', 'obstaclesData', {'x': 0, 'y': 0, 'd': 0, 'w': 0, 'h': 0}),
        ('njsEvents', 'njsEventData', {'d': 0, 'p': 0, 'e': 0}),
    ):
        for source in chart.get(objects, []):
            obj = {'b': source.get('b', 0), 'i': tables.add(metadata, _metadata(source, fields))}
            if objects != 'njsEvents': obj['r'] = source.get('r', lane_rotation(obj['b']))
            difficulty[objects].append(_ref_custom(source, obj))
    for source in chart.get('sliders', []):
        head = {'x': source.get('x', 0), 'y': source.get('y', 0), 'c': source.get('c', 0), 'd': source.get('d', 0), 'a': source.get('_v4HeadAngle',0)}
        # V3 `tc` means tail cut direction, not tail color. Native V4 can
        # independently address endpoint colors; retain that extra information
        # separately instead of overloading a standard V3 field.
        tail = {'x': source.get('tx', 0), 'y': source.get('ty', 0), 'c': source.get('_v4TailColor', source.get('c', 0)), 'd': source.get('tc', source.get('td', 0)), 'a': source.get('_v4TailAngle',0)}
        if source.get('_v4HeadCustomData'): head['customData'] = copy.deepcopy(source['_v4HeadCustomData'])
        if source.get('_v4TailCustomData'): tail['customData'] = copy.deepcopy(source['_v4TailCustomData'])
        data = _copy_custom(source, {'m': source.get('mu', 0), 'tm': source.get('tmu', 0), 'a': source.get('m', 0)})
        difficulty['arcs'].append(_ref_custom(source, {
            'hb': source.get('b', 0), 'tb': source.get('tb', 0),
            'hr': source.get('hr', lane_rotation(source.get('b', 0))),
            'tr': source.get('tr', lane_rotation(source.get('tb', 0))),
            'hi': tables.add('colorNotesData', head), 'ti': tables.add('colorNotesData', tail),
            'ai': tables.add('arcsData', data)}))
    for source in chart.get('burstSliders', []):
        head = {'x': source.get('x', 0), 'y': source.get('y', 0), 'c': source.get('c', 0), 'd': source.get('d', 0), 'a': source.get('_v4HeadAngle',0)}
        if source.get('_v4HeadCustomData'): head['customData'] = copy.deepcopy(source['_v4HeadCustomData'])
        data = _copy_custom(source, {'tx': source.get('tx', 0), 'ty': source.get('ty', 0), 'c': source.get('sc', 0), 's': source.get('s', 0)})
        difficulty['chains'].append(_ref_custom(source, {
            'hb': source.get('b', 0), 'tb': source.get('tb', 0),
            'hr': source.get('hr', lane_rotation(source.get('b', 0))),
            'tr': source.get('tr', lane_rotation(source.get('tb', 0))),
            'i': tables.add('colorNotesData', head), 'ci': tables.add('chainsData', data)}))
    # Preserve deprecated V4.0 spawn-rotation events as well as object lanes.
    if rotations:
        difficulty['spawnRotations'] = []; difficulty['spawnRotationsData'] = []
        for source in rotations:
            difficulty['spawnRotations'].append(_ref_custom(source, {
                'b': source.get('b', 0), 'i': tables.add('spawnRotationsData', _copy_custom(source,{'t':source.get('e',0),'r':source.get('r',0)}))}))
    lightshow = _encode_lights(chart, lane_rotation)
    validate_v4(difficulty, lightshow)
    return {'difficulty': difficulty, 'lightshow': lightshow}


def _encode_lights(chart, lane_rotation):
    out = {'version': '4.0.0', 'waypoints': [], 'waypointsData': [], 'basicEvents': [],
           'basicEventsData': [], 'colorBoostEvents': [], 'colorBoostEventsData': [],
           'eventBoxGroups': [], 'indexFilters': [],
           'basicEventTypesWithKeywords': copy.deepcopy(chart.get('basicEventTypesWithKeywords', {'d': []})),
           'useNormalEventsAsCompatibleEvents': bool(chart.get('useNormalEventsAsCompatibleEvents', False))}
    if '_v4LightshowCustomData' in chart: out['customData'] = copy.deepcopy(chart['_v4LightshowCustomData'])
    out['eventBoxGroups'].extend(copy.deepcopy(chart.get('_v4EmptyEventBoxGroups',[])))
    tables = _Tables(out)
    for objects, metadata, source_field in (('waypoints', 'waypointsData', 'waypoints'),
                                           ('basicEvents', 'basicEventsData', 'basicBeatmapEvents'),
                                           ('colorBoostEvents', 'colorBoostEventsData', 'colorBoostBeatmapEvents')):
        for source in chart.get(source_field, []):
            if objects == 'basicEvents': data = {'t': source.get('et', 0), 'i': source.get('i', 0), 'f': source.get('f', 0)}
            elif objects == 'colorBoostEvents': data = {'b': int(bool(source.get('o', False)))}
            else: data = {'x': source.get('x', 0), 'y': source.get('y', 0), 'd': source.get('d', 0)}
            obj = {'b': source.get('b', 0), 'i': tables.add(metadata, _copy_custom(source, data))}
            if objects == 'waypoints': obj['r'] = source.get('r', lane_rotation(obj['b']))
            out[objects].append(_ref_custom(source, obj))
    for kind, (source_field, boxes, events) in _BOX_TYPES.items():
        out[boxes] = []; out[events] = []
        for group in chart.get(source_field, []):
            if kind == 4 and group.get('t', 1) != 1:
                raise FormatError('Native V4 only supports float FX; integer FX conversion is not supported')
            target = _copy_custom(group, {'b': group.get('b', 0), 'g': group.get('g', 0), 't': kind, 'e': []})
            for box in group.get('e', []):
                data = {'w': box.get('w', 0), 'd': box.get('d', 1),
                        's': box.get('r' if kind == 1 else 's', 0), 't': box.get('t', 1),
                        'b': box.get('b', 0), 'e': box.get('i', 0)}
                if kind in (2, 3): data.update(a=box.get('a', 0), f=box.get('r', 0))
                reference = {'e': tables.add(boxes, _copy_custom(box, data)),
                             'f': tables.add('indexFilters', copy.deepcopy(box.get('f', {'f': 1}))), 'l': []}
                _ref_custom(box, reference)
                for event in box.get('e' if kind == 1 else 'l', []):
                    if kind == 4:
                        collection = chart.get('_fxEventsCollection', {}).get('_fl', [])
                        if not isinstance(event, int) or event < 0 or event >= len(collection): raise FormatError('Invalid V3 float FX index')
                        event = collection[event]
                    if kind == 1:
                        data = {'p': 1 if event.get('i', 0) == 2 else 0,
                                'e': event.get('_v4Easing', 0 if event.get('i', 0) == 1 else -1),
                                'c': event.get('c', 0), 'b': event.get('s', 0), 'f': event.get('f', 0),
                                'sb': event.get('sb', 0), 'sf': event.get('sf', 0)}
                    elif kind == 2: data = {'e': event.get('e', 0), 'l': event.get('l', 0), 'd': event.get('o', 0), 'p': event.get('p', 0), 'r': event.get('r', 0)}
                    elif kind == 3: data = {'e': event.get('e', 0), 'p': event.get('p', 0), 't': event.get('t', 0)}
                    else: data = {'e': event.get('i', 0), 'p': event.get('p', 0), 'v': event.get('v', 0)}
                    reference['l'].append(_ref_custom(event, {'b': event.get('b', 0), 'i': tables.add(events, _copy_custom(event, data))}))
                target['e'].append(reference)
            out['eventBoxGroups'].append(target)
    return out


def ensure_v3_compatible(chart):
    """Reject a downgrade that would export valid JSON with incorrect gameplay."""
    if chart.get('njsEvents'): raise FormatError('This chart has native V4.1 NJS events; export as V4 to preserve them')
    def visit(value, path='chart'):
        if isinstance(value, dict):
            for key, item in value.items():
                meaningful_tail_color=key=='_v4TailColor' and item!=value.get('c',0)
                if meaningful_tail_color or key.startswith('_v4') and key!='_v4TailColor' and item:
                    raise FormatError(f'{path}: V4-only metadata requires V4 export')
                visit(item, path + '.' + key)
        elif isinstance(value, list):
            for item in value: visit(item, path)
    visit(chart)
    for field in ('colorNotes', 'bombNotes', 'obstacles', 'waypoints'):
        if any(obj.get('r', 0) for obj in chart.get(field, [])):
            raise FormatError('Per-object lane rotation cannot be silently converted to V3; export as V4')
    for field in ('sliders', 'burstSliders'):
        if any(obj.get('hr', 0) or obj.get('tr', 0) for obj in chart.get(field, [])):
            raise FormatError('Per-endpoint lane rotation requires V4 export')
    return chart


def audio_data_to_tempo(audio_data):
    """Convert sample-addressed AudioData into the existing piecewise beat clock."""
    if not isinstance(audio_data, dict) or audio_data.get('version') != '4.0.0':
        raise FormatError('Expected AudioData 4.0.0')
    _json(audio_data)
    frequency = _number(audio_data.get('songFrequency', 0), 'songFrequency', integer=True)
    count = _number(audio_data.get('songSampleCount', 0), 'songSampleCount', integer=True)
    if frequency <= 0 or count < 0: raise FormatError('AudioData needs a positive sample frequency and nonnegative sample count')
    regions = _array(audio_data, 'bpmData')
    if not regions: raise FormatError('AudioData must contain at least one tempo region')
    if not isinstance(audio_data.get('songChecksum',''),str):raise FormatError('AudioData songChecksum must be a string')
    for region in _array(audio_data,'lufsData'):
        si=_number(region.get('si'),'lufsData.si',integer=True);ei=_number(region.get('ei'),'lufsData.ei',integer=True)
        _number(region.get('l'),'lufsData.l')
        if si<0 or ei<=si or ei>count:raise FormatError('Invalid AudioData loudness segment range')
    events = []; previous = None
    for index, region in enumerate(regions):
        si = _number(region.get('si'), 'bpmData.si', integer=True)
        ei = _number(region.get('ei'), 'bpmData.ei', integer=True)
        sb = _number(region.get('sb'), 'bpmData.sb'); eb = _number(region.get('eb'), 'bpmData.eb')
        if si < 0 or ei <= si or ei > count or eb <= sb: raise FormatError('Invalid AudioData tempo segment range')
        if previous and (si != previous['ei'] or abs(sb - previous['eb']) > 1e-5):
            raise FormatError('AudioData tempo segments must be ordered, contiguous and beat-continuous')
        if not previous and si != 0: raise FormatError('AudioData tempo timeline must start at sample zero')
        bpm = (eb - sb) / ((ei - si) / frequency) * 60
        if not 20 <= bpm <= 1000: raise FormatError('AudioData tempo is outside the supported 20–1000 BPM range')
        if not previous:
            initial = bpm; offset = -sb * 60 / bpm
        elif sb < 0:
            raise FormatError('A tempo change before beat zero cannot be represented by the current editor clock')
        if not events or abs(events[-1]['m'] - bpm) > 1e-7:
            events.append({'b': max(0, sb), 'm': bpm})
        previous = region
    if previous['ei'] != count: raise FormatError('AudioData tempo timeline must cover the complete song')
    return {'bpm': initial, 'bpmEvents': events, 'songTimeOffset': offset,
            'duration': count / frequency, 'sampleRate': int(frequency)}


def build_audio_data(bpm, duration, bpm_events=None, song_offset=0, sample_rate=48000, lufs=None):
    """Build exact shared tempo segments; sample rounding is bounded to one sample."""
    from modding import BeatClock
    _number(bpm, 'bpm'); _number(duration, 'duration'); _number(song_offset, 'song offset')
    _number(sample_rate, 'sample rate', integer=True)
    if not 20 <= bpm <= 1000 or duration <= 0 or sample_rate < 8000: raise FormatError('Invalid audio clock')
    markers = sorted(copy.deepcopy(bpm_events or []), key=lambda e: e.get('b', 0))
    for marker in markers:
        _number(marker.get('b'), 'BPM marker beat'); _number(marker.get('m'), 'BPM marker value')
        if marker['b'] < 0 or not 20 <= marker['m'] <= 1000: raise FormatError('Invalid BPM marker')
    clock = BeatClock(bpm, markers); count = int(round(duration * sample_rate))
    boundaries = {0, count}
    for marker in markers:
        time = clock.time(marker['b']) + song_offset
        if 0 < time < duration: boundaries.add(int(round(time * sample_rate)))
    regions = []; ordered = sorted(boundaries)
    for si, ei in zip(ordered, ordered[1:]):
        regions.append({'si': si, 'ei': ei,
                        'sb': clock.beat(si / sample_rate - song_offset),
                        'eb': clock.beat(ei / sample_rate - song_offset)})
    result = {'version': '4.0.0', 'songChecksum': '', 'songSampleCount': count,
              'songFrequency': int(sample_rate), 'bpmData': regions, 'lufsData': []}
    if lufs is not None:
        _number(lufs, 'LUFS'); result['lufsData'] = [{'si': 0, 'ei': count, 'l': lufs}]
    audio_data_to_tempo(result)
    return result


def info_v4_to_v2(info):
    """Adapt Info4 metadata for existing importer/editor while retaining originals."""
    if info.get('version') not in ('4.0.0', '4.0.1'): raise FormatError('Unsupported Info4 version')
    song = info.get('song', {}); audio = info.get('audio', {}); groups = {}
    if not isinstance(song,dict) or not isinstance(audio,dict):raise FormatError('Info4 song and audio must be objects')
    for key in ('title','subTitle','author'):
        if not isinstance(song.get(key,''),str):raise FormatError(f'Info4 song.{key} must be a string')
    for key in ('songFilename','audioDataFilename'):
        if not isinstance(audio.get(key),str) or not audio[key]:raise FormatError(f'Info4 audio.{key} is required')
    for key in ('bpm','songDuration','lufs','previewStartTime','previewDuration'):
        _number(audio.get(key),f'Info4 audio.{key}')
    if not 20<=audio['bpm']<=1000 or audio['songDuration']<0:raise FormatError('Info4 has invalid BPM or duration')
    if not isinstance(info.get('environmentNames',[]),list) or any(not isinstance(value,str) for value in info.get('environmentNames',[])):
        raise FormatError('Info4 environmentNames must be an array of strings')
    for key in ('difficultyBeatmaps','colorSchemes'):_array(info,key)
    for diff in info.get('difficultyBeatmaps', []):
        mode = diff.get('characteristic', 'Standard'); difficulty = diff.get('difficulty', 'ExpertPlus')
        rank = {'Easy': 1, 'Normal': 3, 'Hard': 5, 'Expert': 7, 'ExpertPlus': 9}.get(difficulty)
        if rank is None: raise FormatError('Unknown native difficulty slot')
        custom = copy.deepcopy(diff.get('customData', {}))
        if 'difficultyLabel' in custom and '_difficultyLabel' not in custom: custom['_difficultyLabel'] = custom['difficultyLabel']
        for key in ('requirements', 'suggestions'):
            if key in custom and '_' + key not in custom: custom['_' + key] = custom[key]
        groups.setdefault(mode, []).append({
            '_difficulty': difficulty, '_difficultyRank': rank,
            '_beatmapFilename': diff.get('beatmapDataFilename', ''),
            '_lightshowFilename': diff.get('lightshowDataFilename', ''),
            '_noteJumpMovementSpeed': diff.get('noteJumpMovementSpeed', 10),
            '_noteJumpStartBeatOffset': diff.get('noteJumpStartBeatOffset', 0),
            '_environmentNameIdx': diff.get('environmentNameIdx', 0),
            '_beatmapColorSchemeIdx': diff.get('beatmapColorSchemeIdx', -1),
            '_customData': custom, '_v4Authors': copy.deepcopy(diff.get('beatmapAuthors', {}))})
    environments = copy.deepcopy(info.get('environmentNames', ['DefaultEnvironment']))
    def rgba(value):
        text=str(value).lstrip('#')
        if len(text)==6:text+='FF'
        if len(text)!=8 or any(char not in '0123456789abcdefABCDEF' for char in text):
            raise FormatError('Native V4 color schemes must contain RGBA hex strings')
        return {key:int(text[index:index+2],16)/255 for key,index in zip(('r','g','b','a'),(0,2,4,6))}
    schemes=[]
    for source in info.get('colorSchemes',[]):
        scheme={'colorSchemeId':source.get('colorSchemeName','Imported')}
        for key in ('saberAColor','saberBColor','obstaclesColor','environmentColor0','environmentColor1',
                    'environmentColor0Boost','environmentColor1Boost','environmentColorW','environmentColorWBoost'):
            if key in source:scheme[key]=rgba(source[key])
        schemes.append({'useOverride':source.get('overrideNotes',source.get('useOverride',True)),
                        'colorScheme':scheme})
    return {'_version': '2.1.0', '_songName': song.get('title', ''), '_songSubName': song.get('subTitle', ''),
            '_songAuthorName': song.get('author', ''), '_levelAuthorName': '',
            '_beatsPerMinute': audio.get('bpm', 120), '_songTimeOffset': 0,
            '_songFilename': audio.get('songFilename', 'song.ogg'),
            '_previewStartTime': audio.get('previewStartTime', 0), '_previewDuration': audio.get('previewDuration', 12),
            '_coverImageFilename': info.get('coverImageFilename', 'cover.png'),
            '_environmentName': environments[0] if environments else 'DefaultEnvironment',
            '_environmentNames': environments, '_colorSchemes': schemes,
            '_difficultyBeatmapSets': [{'_beatmapCharacteristicName': mode, '_difficultyBeatmaps': diffs} for mode, diffs in groups.items()],
            '_customData': copy.deepcopy(info.get('customData', {}))}


def info_v2_to_v4(info, duration, *, lufs=0, sample_rate=48000, color_schemes=None):
    """Build a valid native Info4. Explicit filenames reference split chart/light files."""
    schemes = copy.deepcopy(color_schemes or [])
    for scheme in schemes:
        scheme.setdefault('overrideNotes',bool(scheme.get('useOverride',True)))
        scheme.setdefault('overrideLights',bool(scheme.get('useOverride',True)))
    # V2 object RGB colors and V4 hex colors have different shapes. Convert known
    # fields rather than putting incompatible V2 objects in an otherwise valid Info4.
    if not schemes:
        def hex_color(color, default):
            if isinstance(color, str):
                value=color.lstrip('#')
                return value+'FF' if len(value)==6 else value
            if not isinstance(color, dict): return default
            return ''.join(f'{max(0, min(255, round(float(color.get(k, 1 if k=="a" else 0)) * 255))):02X}' for k in ('r', 'g', 'b', 'a'))
        for wrapper in info.get('_colorSchemes', []):
            scheme = wrapper.get('colorScheme', wrapper.get('_colorScheme', wrapper))
            schemes.append({'colorSchemeName': scheme.get('colorSchemeId', scheme.get('_colorSchemeId', 'Imported')),
                            'overrideNotes': bool(wrapper.get('useOverride', wrapper.get('_useOverride', True))),
                            'overrideLights': bool(wrapper.get('useOverride', wrapper.get('_useOverride', True))),
                            **{target: hex_color(scheme.get(source, scheme.get('_' + source)), default) for target, source, default in (
                                ('saberAColor', 'saberAColor', 'FF243CFF'), ('saberBColor', 'saberBColor', '2666FFFF'),
                                ('obstaclesColor', 'obstaclesColor', 'FFFFFFFF'),
                                ('environmentColor0', 'environmentColor0', 'FF243CFF'), ('environmentColor1', 'environmentColor1', '2666FFFF'),
                                ('environmentColor0Boost', 'environmentColor0Boost', 'FF243CFF'), ('environmentColor1Boost', 'environmentColor1Boost', '2666FFFF'))}})
    result = {'version': '4.0.1',
              'song': {'title': info.get('_songName', ''), 'subTitle': info.get('_songSubName', ''), 'author': info.get('_songAuthorName', '')},
              'audio': {'songFilename': info.get('_songFilename', 'song.ogg'), 'songDuration': duration,
                        'audioDataFilename': 'AudioData.dat', 'bpm': info.get('_beatsPerMinute', 120), 'lufs': lufs,
                        'previewStartTime': info.get('_previewStartTime', 0), 'previewDuration': info.get('_previewDuration', 12)},
              'songPreviewFilename': info.get('_songFilename', 'song.ogg'), 'coverImageFilename': info.get('_coverImageFilename', 'cover.png'),
              'environmentNames': copy.deepcopy(info.get('_environmentNames', [info.get('_environmentName', 'DefaultEnvironment')])),
              'colorSchemes': schemes, 'difficultyBeatmaps': [], 'customData': copy.deepcopy(info.get('_customData', {}))}
    for group in info.get('_difficultyBeatmapSets', []):
        for diff in group.get('_difficultyBeatmaps', []):
            filename = diff['_beatmapFilename']; custom = copy.deepcopy(diff.get('_customData', {}))
            for key in ('difficultyLabel', 'requirements', 'suggestions'):
                if '_' + key in custom: custom[key] = custom.pop('_' + key)
            result['difficultyBeatmaps'].append({
                'characteristic': group.get('_beatmapCharacteristicName', 'Standard'), 'difficulty': diff['_difficulty'],
                'beatmapAuthors': copy.deepcopy(diff.get('_v4Authors', {'mappers': [info.get('_levelAuthorName', 'ArcForge')], 'lighters': []})),
                'environmentNameIdx': diff.get('_environmentNameIdx', 0), 'beatmapColorSchemeIdx': diff.get('_beatmapColorSchemeIdx', -1),
                'noteJumpMovementSpeed': diff.get('_noteJumpMovementSpeed', 10), 'noteJumpStartBeatOffset': diff.get('_noteJumpStartBeatOffset', 0),
                'beatmapDataFilename': filename, 'lightshowDataFilename': diff.get('_lightshowFilename', filename.rsplit('.', 1)[0] + '.lightshow.dat'),
                'customData': custom})
    return result
