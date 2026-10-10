"""Persistent projects and complete, root-level Beat Saber ZIP export."""
import bootstrap
import collections, copy, hashlib, io, json, math, pathlib, shutil, zipfile, threading
from PIL import Image, ImageDraw
from modding import BeatClock, decode_precision
from engine.formats import (FormatError, encode_v4, ensure_v3_compatible,
                            build_audio_data, info_v2_to_v4)

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / 'data'
RANKS = {1:'Easy',3:'Normal',5:'Hard',7:'Expert',9:'ExpertPlus'}
MODES = {'Standard','90Degree','360Degree','OneSaber','NoArrows'}
_EXPORT_LOCKS={}
_PROJECT_LOCKS=collections.defaultdict(threading.RLock)
def project_lock(identity):return _PROJECT_LOCKS[identity]

def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def read_project(identity):
    path = DATA / 'projects' / identity / 'project.json'
    if not path.is_file():
        raise FileNotFoundError('Project not found')
    return json.loads(path.read_text(encoding='utf-8'))

def _canonical_tempo(bpm, events):
    """Compare musical clocks rather than unused marker metadata or duplicates."""
    result=[];last=float(bpm)
    for event in sorted(events,key=lambda e:e['b']):
        beat=float(event['b']);value=float(event['m'])
        if beat==0:last=value;result=[{'b':0.,'m':value}]
        elif abs(value-last)>1e-8:result.append({'b':beat,'m':value});last=value
    if result and result[0]['b']==0 and abs(result[0]['m']-bpm)<1e-8:result.pop(0)
    return result

def jump_settings(bpm, njs, desired_distance=None):
    half_beats = 4.
    while njs * 60 / bpm * half_beats > 17.999:
        half_beats /= 2
    half_beats = max(.25, half_beats)
    offset = 0. if desired_distance is None else desired_distance/(2*njs*60/bpm)-half_beats
    offset = max(.25-half_beats, offset)
    distance = 2*njs*(half_beats+offset)*60/bpm
    return round(offset, 6), round(distance, 3)

def validate_project(project):
    errors=[];warnings=[];stats=[]
    bpm=float(project.get('bpm',project['audio']['bpm']))
    duration=float(project['audio']['duration'])
    if not math.isfinite(bpm) or not 20<=bpm<=1000:errors.append('BPM must be between 20 and 1000.')
    identities=set();slots=set()
    for c in project.get('charts', []):
        if c['id'] in identities:errors.append('Duplicate chart identifier.')
        identities.add(c['id'])
        mode=c.get('mode','Standard');rank=int(c.get('rank',9));slot=(mode,rank)
        if mode not in MODES:errors.append(f'Unsupported mode {mode}.')
        if rank not in RANKS:errors.append(f'Invalid native difficulty rank {rank}.')
        if slot in slots:errors.append('Each mode has five native slots. Rename them freely, but assign each chart a unique slot.')
        slots.add(slot)
        chart=c['data'];tempo_events=chart.get('bpmEvents',[])
        if not str(chart.get('version','')).startswith('3') or not isinstance(chart.get('colorNotes'),list):errors.append(f"{c['name']}: a complete V3 chart is required.")
        invalid_tempo=any(not isinstance(e.get('b'),(int,float)) or not math.isfinite(e['b']) or e['b']<0 or not isinstance(e.get('m'),(int,float)) or not math.isfinite(e['m']) or not 20<=e['m']<=1000 for e in tempo_events)
        if invalid_tempo:errors.append(f"{c['name']}: invalid BPM marker.")
        clock=BeatClock(bpm if 20<=bpm<=1000 else 120,[] if invalid_tempo else tempo_events);occupied=set();timings=[];hand=collections.defaultdict(list)
        for n in chart.get('colorNotes', []):
            custom=n.get('customData',{})
            if custom.get('fake') or custom.get('uninteractable') or custom.get('interactable') is False:continue
            required=['b','x','y','c','d']
            if any(k not in n for k in required):errors.append(f"{c['name']}: incomplete note.");continue
            if not all(isinstance(n[k],(int,float)) and math.isfinite(n[k]) for k in required):errors.append(f"{c['name']}: invalid note value.");continue
            t=clock.time(n['b'])+float(project.get('songTimeOffset',0));timings.append(t);hand[n['c']].append((t,n))
            if n['c'] not in [0,1] or n['d'] not in list(range(9))+list(range(1000,1361))+list(range(2000,2361)):
                errors.append(f"{c['name']}: invalid hand or cut direction.")
            if t<0 or t>duration+.01:errors.append(f"{c['name']}: a note falls outside the audio.")
            key=(round(n['b'],5),n['x'],n['y'])
            if key in occupied:errors.append(f"{c['name']}: overlapping notes at beat {n['b']:.3f}.")
            occupied.add(key)
        for color,notes in hand.items():
            grouped={}
            for t,n in sorted(notes,key=lambda p:p[0]):grouped.setdefault(round(t,6),[]).append(n)
            previous=None
            for t,ns in grouped.items():
                if previous:
                    pt,pn=previous;gap=t-pt
                    if gap<.065:warnings.append(f"{c['name']}: hand {color+1} has a {gap*1000:.0f} ms transition near {t:.2f}s.")
                previous=(t,ns[0])
        for w in chart.get('obstacles',[]):
            if w.get('d',0)<0:errors.append(f"{c['name']}: negative wall duration.")
        for s in chart.get('sliders',[])+chart.get('burstSliders',[]):
            if s.get('tb',0)<s.get('b',0):errors.append(f"{c['name']}: a sustain ends before it starts.")
        nps=len(timings)/max(.1,duration) if timings else 0
        stats.append({'id':c['id'],'notes':len(timings),'nps':round(nps,2),
            'bombs':len(chart.get('bombNotes',[])),'walls':len(chart.get('obstacles',[])),
            'arcs':len(chart.get('sliders',[])),'chains':len(chart.get('burstSliders',[]))})
    if not project.get('charts'):errors.append('The project has no charts.')
    return {'status':'error' if errors else 'review' if warnings else 'passed',
        'errors':list(dict.fromkeys(errors))[:100],'warnings':list(dict.fromkeys(warnings))[:100],
        'charts':stats,'scope':'File/geometry checks; gameplay still needs headset playtesting.'}

def cover(path, title):
    image=Image.new('RGB',(512,512),'#101219');draw=ImageDraw.Draw(image)
    for i in range(12):
        x=40+i*17
        draw.line((x,410-i*8,256,90+i*5,512-x,410-i*8),fill=(100+i*8,70+i*5,210),width=3)
    draw.text((38,438),title[:45],fill='#eef0f4')
    image.save(path)

def export_project(identity, version=None):
    with _EXPORT_LOCKS.setdefault(identity,threading.RLock()):
        return _export_project(identity, version=version)

def _export_project(identity, project=None, folder=None, extras_folder=None, version=None):
    """Export a saved project, or an isolated draft without touching its library entry."""
    draft=project is not None
    project=read_project(identity) if project is None else project
    validation=validate_project(project)
    if validation['errors']:raise ValueError('; '.join(validation['errors'][:5]))
    folder=DATA/'projects'/identity if folder is None else pathlib.Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    extras_folder=DATA/'projects'/identity if extras_folder is None else pathlib.Path(extras_folder)
    config=project.get('config',{})
    native_version=str(version if version is not None else config.get('beatmapVersion','3'))
    if native_version not in ['3','4']:raise ValueError('Native export format must be 3 or 4.')
    target=folder/f"map-r{int(project.get('revision',1))}{'-v4' if native_version=='4' else ''}.zip"
    # A corrected serializer must invalidate old same-revision downloads too.
    # Project-only caching can otherwise keep serving a broken native archive
    # after an application update until the user edits their map.
    serializer_files=[pathlib.Path(__file__)]
    if native_version=='4':serializer_files.append(pathlib.Path(__file__).parent/'engine/formats/v4.py')
    cache_updated=max((folder/'project.json').stat().st_mtime_ns if not draft else 0,
                      *(path.stat().st_mtime_ns for path in serializer_files))
    if not draft and target.is_file() and target.stat().st_mtime_ns>=cache_updated:
        return target
    audio=DATA/'audio'/project['audio']['id']/'song.ogg'
    if not audio.is_file():raise ValueError('The project audio is missing.')
    bpm=float(project['bpm'])
    preview=config.get('previewStart')
    if preview is None:
        sections=[s for s in project['audio'].get('sections',[]) if s['start']<project['audio']['duration']-5]
        preview=max(sections,key=lambda s:s.get('energy',0))['start'] if sections else min(45,project['audio']['duration']*.35)
    env=project.get('environment','DefaultEnvironment');allenv='GlassDesertEnvironment'
    imported_custom=copy.deepcopy(project.get('importCustomData',{}))
    imported_custom.setdefault('_editors',{}).update({'_lastEditedBy':'ArcForge','_arcforge':{'version':'1.3.0'}})
    info={'_version':'2.1.0','_songName':project['name'],'_songSubName':config.get('songSubName',''),
        '_songAuthorName':config.get('artist',''),'_levelAuthorName':config.get('author','ArcForge'),
        '_beatsPerMinute':bpm,'_songTimeOffset':float(project.get('songTimeOffset',0)),'_shuffle':0,'_shufflePeriod':.5,
        '_previewStartTime':min(max(0,project['audio']['duration']-1),float(preview)),
        '_previewDuration':min(12,project['audio']['duration']),
        '_songFilename':'song.ogg','_coverImageFilename':'cover.png','_environmentName':env,
        '_allDirectionsEnvironmentName':allenv,'_environmentNames':project.get('importEnvironmentNames') or [env,allenv],
        '_colorSchemes':project.get('importColorSchemes',[]),'_difficultyBeatmapSets':[],'_customData':imported_custom}
    sets={};files={};shared_tempo=None
    for c in project['charts']:
        mode=c['mode'];rank=int(c.get('rank',9));filename=f'{RANKS[rank]}{mode}.dat'
        chart=c['data']
        digest=hashlib.sha256(json.dumps(chart,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        unchanged=digest==c.get('originalCanonicalHash')
        if native_version=='3' and c.get('originalData') and unchanged:chart=c['originalData']
        elif c.get('legacyMod'):
            raise ValueError('This imported V2 chart uses legacy mod animations. Export it unchanged or convert it to V3 in ChroMapper before editing.')
        if native_version=='4':
            tempo=json.dumps(_canonical_tempo(bpm,chart.get('bpmEvents',[])),sort_keys=True,separators=(',',':'))
            if shared_tempo is None:shared_tempo=tempo
            elif tempo!=shared_tempo:
                raise FormatError('V4 uses one shared AudioData tempo timeline. Charts with different BPM markers must be aligned before V4 export.')
            light_filename=filename.rsplit('.',1)[0]+'.lightshow.dat'
            if unchanged and c.get('originalV4Data') is not None and c.get('originalV4Lightshow') is not None:
                native={'difficulty':c['originalV4Data'],'lightshow':c['originalV4Lightshow']}
            else:native=encode_v4(chart)
            from engine.formats import validate_v4
            validate_v4(native['difficulty'],native['lightshow'])
            files[filename]=json.dumps(native['difficulty'],separators=(',',':'),allow_nan=False).encode('utf-8')
            files[light_filename]=json.dumps(native['lightshow'],separators=(',',':'),allow_nan=False).encode('utf-8')
        else:
            if str(chart.get('version','')).startswith('3'):ensure_v3_compatible(chart)
            files[filename]=json.dumps(chart,separators=(',',':'),allow_nan=False).encode('utf-8')
        njs=float(c.get('njs',20));offset,distance=jump_settings(bpm,njs,c.get('jumpDistance'))
        cd=copy.deepcopy(c.get('importBeatmapCustomData',{}))
        cd.update({'_difficultyLabel':c['name'],'_requirements':c.get('modInfo',{}).get('requirements',[]),
            '_suggestions':list(dict.fromkeys(cd.get('_suggestions',[])+c.get('modInfo',{}).get('dependencies',[])))})
        for name in ['colorLeft','colorRight']:
            if name in c.get('modInfo',{}):cd['_'+name]=c['modInfo'][name]
        d={'_difficulty':RANKS[rank],'_difficultyRank':rank,'_beatmapFilename':filename,
            '_noteJumpMovementSpeed':njs,'_noteJumpStartBeatOffset':offset,
            '_beatmapColorSchemeIdx':c.get('importColorSchemeIdx',-1),'_environmentNameIdx':c.get('importEnvIndex',1 if mode in ['90Degree','360Degree'] else 0),'_customData':cd}
        if native_version=='4':d['_lightshowFilename']=light_filename
        if c.get('importV4Authors') is not None:d['_v4Authors']=copy.deepcopy(c['importV4Authors'])
        sets.setdefault(mode,[]).append(d)
    for mode,diffs in sets.items():info['_difficultyBeatmapSets'].append({'_beatmapCharacteristicName':mode,'_difficultyBeatmaps':sorted(diffs,key=lambda d:d['_difficultyRank'])})
    asset=project.get('asset')
    if project.get('importCustomData',{}).get('_assetBundle'):
        info['_customData']['_assetBundle']=project['importCustomData']['_assetBundle']
    if asset:
        platform=asset.get('platform','windows2021')
        info['_customData']['_assetBundle']={'_'+platform:int(asset['checksum'])}
        files['bundle'+('Windows2019' if platform=='windows2019' else 'Windows2021')+'.vivify']=(DATA/'assets'/asset['id']/'bundle.vivify').read_bytes()
    if native_version=='4':
        clock_events=json.loads(shared_tempo or '[]')
        sample_rate=int(project['audio'].get('sampleRate',48000) or 48000)
        lufs=project['audio'].get('integratedLufs',project.get('importV4Info',{}).get('audio',{}).get('lufs'))
        if lufs is None:
            from engine.formats import measure_loudness
            lufs=measure_loudness(audio)
        lufs=float(lufs)
        audio_data=build_audio_data(bpm,float(project['audio']['duration']),clock_events,
                                   float(project.get('songTimeOffset',0)),sample_rate,lufs=lufs)
        original_audio=project.get('nativeAudioData')
        # Reuse untouched native tempo/LUFS/checksum data only when its timing
        # conversion still agrees with every chart and the project audio clock.
        if original_audio and all(hashlib.sha256(json.dumps(c['data'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==c.get('originalCanonicalHash') for c in project['charts']):
            from engine.formats import audio_data_to_tempo
            original_clock=audio_data_to_tempo(original_audio)
            if abs(original_clock['bpm']-bpm)<1e-5 and abs(original_clock['songTimeOffset']-float(project.get('songTimeOffset',0)))<1e-5:
                audio_data=copy.deepcopy(original_audio)
        files['AudioData.dat']=json.dumps(audio_data,separators=(',',':'),allow_nan=False).encode('utf-8')
        info=info_v2_to_v4(info,float(project['audio']['duration']),
                          lufs=lufs,
                          color_schemes=project.get('importV4Info',{}).get('colorSchemes'))
        original_info=project.get('importV4Info',{})
        preview_name=original_info.get('songPreviewFilename')
        if preview_name and preview_name!=original_info.get('audio',{}).get('songFilename'):
            safe_preview=pathlib.PurePosixPath(preview_name.replace('\\','/')).name
            if (extras_folder/'extras'/safe_preview).is_file():info['songPreviewFilename']=safe_preview
    coverpath=folder/'cover.png'
    if draft and not coverpath.exists() and (extras_folder/'cover.png').is_file():
        shutil.copyfile(extras_folder/'cover.png',coverpath)
    if not coverpath.is_file():cover(coverpath,project['name'])
    temporary=target.with_suffix('.zip.pending')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('Info.dat',json.dumps(info,indent=2).encode('utf-8'))
        for name,contents in files.items():z.writestr(name,contents)
        z.write(audio,'song.ogg');z.write(coverpath,'cover.png')
        for extra in (extras_folder/'extras').glob('*') if (extras_folder/'extras').exists() else []:
            if extra.is_file() and extra.name not in files and extra.name not in ['Info.dat','song.ogg','cover.png']:z.write(extra,extra.name)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip():raise ValueError('The archive failed its integrity check.')
    temporary.replace(target)
    write_json(folder/'validation.json',validation)
    return target

def upgrade_v2(chart):
    """Preserve legacy data while presenting editable notes in the v3 schema."""
    if str(chart.get('version','')).startswith('3'):return chart
    result={'version':'3.3.0','bpmEvents':[],'rotationEvents':[],'colorNotes':[],'bombNotes':[],
        'obstacles':[],'sliders':[],'burstSliders':[],'waypoints':[],'basicBeatmapEvents':[],
        'colorBoostBeatmapEvents':[],'lightColorEventBoxGroups':[],'lightRotationEventBoxGroups':[],
        'lightTranslationEventBoxGroups':[],'vfxEventBoxGroups':[],
        '_fxEventsCollection':{'_fl':[],'_il':[]},'basicEventTypesWithKeywords':{'d':[]},
        'useNormalEventsAsCompatibleEvents':False,'customData':copy.deepcopy(chart.get('_customData',{}))}
    def modern(cd):
        return {k.lstrip('_'):v for k,v in cd.items()}
    for n in chart.get('_notes',[]):
        v={'b':n['_time'],'x':n['_lineIndex'],'y':n['_lineLayer'],'customData':modern(n.get('_customData',{}))}
        if n['_type']==3:result['bombNotes'].append(v)
        elif n['_type'] in [0,1]:result['colorNotes'].append({**v,'c':n['_type'],'d':n['_cutDirection'],'a':0})
    for w in chart.get('_obstacles',[]):
        result['obstacles'].append({'b':w['_time'],'x':w['_lineIndex'],'y':2 if w['_type']==1 else 0,
            'd':w['_duration'],'w':w['_width'],'h':5 if w['_type']==0 else 3,'customData':modern(w.get('_customData',{}))})
    for e in chart.get('_events',[]):
        if e['_type']==100:result['bpmEvents'].append({'b':e['_time'],'m':e.get('_floatValue',e['_value'])})
        else:result['basicBeatmapEvents'].append({'b':e['_time'],'et':e['_type'],'i':e['_value'],'f':e.get('_floatValue',1),
            'customData':modern(e.get('_customData',{}))})
    return result
