"""Optional audio attention, non-destructive continuation previews and live advisory checks."""
import copy, hashlib, json, pathlib, time, uuid, threading, collections, math
from fastapi import APIRouter, Request
from projects import DATA, ROOT, write_json, read_project, project_lock
from studio import safe_id, persist
router=APIRouter()
MODES={'off','legacy','arcaudio','hybrid','arcaudio3'}

@router.post('/api/training/composition')
def train_full_levels():
    from server import start_job,ML_LOCK,DATASET,MODEL
    from engine.composition import train_composition
    def work(checkpoint):
        with ML_LOCK:
            return train_composition(DATASET,MODEL,progress=lambda p:checkpoint(5+94*float(p.get('progress',0)),p.get('message','Learning full human levels')),
                cancel_callback=getattr(checkpoint,'cancelled',None))
    return start_job('composition',work)

def attention_config(config):
    mode=str(config.get('arcAudioMode','legacy')).lower()
    if mode not in MODES:raise ValueError('Choose Off, Legacy, ArcAudio v2.0, Hybrid or ArcAudio 3.0.')
    from engine.arcaudio.settings import normalize_settings
    value=config.get('arcAudioSettings',{})
    if not isinstance(value,dict):raise ValueError('Attention settings must be an object.')
    return mode,normalize_settings(value)

def generation_overrides(settings,body):
    """Preview operations use the user's current choices, never stale saved UI state."""
    if not isinstance(settings,dict) or not isinstance(body,dict):
        raise ValueError('Generation settings and overrides must be objects.')
    result=copy.deepcopy(settings)
    analysis=body.get('analysis',{})
    if not isinstance(analysis,dict) or set(analysis)-{'arcAudioMode','arcAudioSettings','analysisCompute','mappingRegions','sourceFocusRegions'}:
        raise ValueError('Analysis overrides support audio settings, compute and mapping regions only.')
    result.update(analysis)
    mode,knobs=attention_config(result)
    backend=body.get('backend',result.get('backend','hybrid'))
    if not isinstance(backend,str) or backend not in {'local','inferno','hybrid','auto'}:
        raise ValueError('Choose a supported note placement engine: local, inferno, hybrid or auto.')
    from engine.analysis_runtime import normalize_compute
    result.update(arcAudioMode=mode,arcAudioSettings=knobs,analysisCompute=normalize_compute(result.get('analysisCompute')),backend=backend)
    return result

def regional_config(config,duration):
    """One region contract for creation, inspection and partial regeneration."""
    from engine.regions import validate_regions
    from engine.arcaudio.focus import normalize_focus_regions
    result=copy.deepcopy(config)
    result['mappingRegions']=validate_regions(result.get('mappingRegions',result.get('regions',[])),duration)
    focus=result.get('sourceFocusRegions',[])
    if not isinstance(focus,list):raise ValueError('Source focus regions must be a list.')
    # Validate legacy source-focus data through the same attention policy.
    effective=normalize_focus_regions(result['mappingRegions'],focus,duration)
    if effective and result.get('arcAudioMode','legacy') in {'off','legacy'}:
        raise ValueError('Regional source controls need ArcAudio 2.0, Hybrid or ArcAudio 3.0. Select an attention engine; Legacy timing stays unchanged.')
    return result

def cached_attention(analysis,mode,settings,compute=None,mapping_regions=None,focus_regions=None):
    from engine.arcaudio import analyze_attention
    from engine.analysis_runtime import normalize_compute
    compute=normalize_compute(compute)
    regions=regional_config({'arcAudioMode':mode,'mappingRegions':[] if mapping_regions is None else mapping_regions,'sourceFocusRegions':[] if focus_regions is None else focus_regions},analysis['duration'])
    mapping_regions=regions['mappingRegions'];focus_regions=regions.get('sourceFocusRegions',[])
    if mode in {'off','legacy'}:return analyze_attention(analysis,settings,mode=mode,compute=compute)
    folder=ROOT/'engine/arcaudio';fingerprint=hashlib.sha256(b''.join(f.read_bytes() for f in sorted(folder.glob('*.py')))).hexdigest()
    if mode=='arcaudio3':
        recognition=ROOT/'engine/recognition';manifest=ROOT/'models/recognition/yamnet-install.json'
        fingerprint+=hashlib.sha256(b''.join(f.read_bytes() for f in sorted(recognition.glob('*.py')))+(ROOT/'engine/analysis_runtime.py').read_bytes()+(manifest.read_bytes() if manifest.exists() else b'')).hexdigest()
    path=pathlib.Path(analysis.get('audioPath',''));source=(str(path),path.stat().st_mtime_ns,path.stat().st_size) if path.is_file() else analysis
    clock={k:analysis.get(k) for k in ['bpm','offset','beatTimes','sourceEvidence']}
    digest=hashlib.sha256(json.dumps([source,clock,mode,settings,compute,mapping_regions,focus_regions,fingerprint],sort_keys=True).encode()).hexdigest()
    identity=analysis.get('id')
    target=DATA/'audio'/safe_id(identity)/'attention'/f'{digest}.json' if identity else None
    if target and target.exists():
        result=json.loads(target.read_text('utf-8'));result.setdefault('performance',{})['cacheHit']=True
        recognition_performance=result.get('recognition',{}).get('performance')
        if isinstance(recognition_performance,dict):
            recognition_performance.update(cacheHit=True,inferenceExecuted=False,cacheOriginDevice=recognition_performance.get('device'))
        return result
    result=analyze_attention(analysis,settings,mode=mode,compute=compute,mapping_regions=mapping_regions,focus_regions=focus_regions);result.setdefault('performance',{})['cacheHit']=False
    if target:write_json(target,result)
    return result

@router.post('/api/audio/{key}/attention')
async def analyze_audio_attention(key:str,request:Request):
    from server import audio_record
    body=await request.json()
    if not isinstance(body,dict):raise ValueError('Supply audio analysis settings as an object.')
    mode=str(body.get('mode','arcaudio')).lower()
    mode,settings=attention_config({'arcAudioMode':mode,'arcAudioSettings':body.get('settings',{})})
    import asyncio
    return await asyncio.to_thread(cached_attention,audio_record(safe_id(key)),mode,settings,body.get('compute'),body.get('mappingRegions'),body.get('sourceFocusRegions'))

@router.post('/api/playtest/advisory')
async def advisory(request:Request):
    from server import audio_record
    from gameplay import analyze_playability
    body=await request.json();chart=body.get('chart',{});config=body.get('config',{})
    if not isinstance(chart,dict) or not isinstance(config,dict):raise ValueError('Supply a chart and its settings.')
    if not isinstance(chart.get('colorNotes',[]),list):raise ValueError('Draft notes must be an array.')
    if len(chart.get('colorNotes',[]))>150000:raise ValueError('Limit a live draft check to 150,000 notes.')
    analysis=audio_record(safe_id(body['audioId']))
    for note in chart.get('colorNotes',[]):
        if not isinstance(note,dict) or any(k not in note for k in ['b','x','y','c','d']):raise ValueError('Draft notes require beat, position, hand and direction.')
        if any(not isinstance(note[k],(int,float)) or not math.isfinite(note[k]) for k in ['b','x','y','c','d']):raise ValueError('Draft notes must have finite numeric values.')
    report=analyze_playability(chart,analysis,config)
    for issue in report['issues']:
        if issue.get('code') in ['parity','handTiming','sameHandStack','saberClash','reach','recoveryReach','noteOcclusion','wallOcclusion','bombClearance']:
            issue['severity']='warning'
        if issue.get('code')=='parity':
            note=chart['colorNotes'][issue['noteIndex']];hand='blue/right' if note.get('c')==1 else 'red/left'
            issue['description']=f'Possible repeated-direction {hand} swings: review the cut and follow-through. Intentional resets can be valid when recovery is sufficient.'
    errors=sum(i['severity']=='error' for i in report['issues'])
    report.update(errors=errors,warnings=len(report['issues'])-errors,passed=errors==0)
    return report|{'advisory':True,'changedNotes':0,'autoFix':False,'method':'Deterministic timing and geometric checks on the unsaved draft; no edits are applied.'}

def preview_path(key,preview):return DATA/'projects'/safe_id(key)/'continuation'/f'{safe_id(preview)}.json'

def start_continuation(key,body):
    from server import audio_record,start_job,ML_LOCK,MODEL
    from engine import generate_map
    from engine.continuation import plan_continuation,adapt_continuation
    previous=read_project(safe_id(key));chart_id=body.get('chartId',previous['charts'][0]['id'])
    chart=next((c for c in previous['charts'] if c['id']==chart_id),None)
    if not chart:raise ValueError('Chart not found.')
    if chart.get('legacyMod'):raise ValueError('Convert advanced legacy mod data to V3 before AI continuation.')
    if len(body.get('ranges',[]))>32:raise ValueError('Select at most 32 regions in one operation.')
    analysis=copy.deepcopy(audio_record(previous['audio']['id']))
    settings=previous.get('config',{})|{'bpm':previous['bpm'],'mode':chart['mode'],'rank':chart['rank'],
        'nps':chart.get('report',{}).get('requestedNps',8),'njs':chart.get('njs',20),'jumpDistance':chart.get('jumpDistance',24),
        'songTimeOffset':previous.get('songTimeOffset',0),'seed':int(time.time_ns()%2147483647)}
    settings=generation_overrides(settings,body)
    settings=regional_config(settings,analysis['duration'])
    if body.get('inheritStyle') is False:settings['style']=body.get('style','Modern')
    settings['philosophy']=settings.get('philosophy',{})|body.get('overrides',{})
    mode,attention_settings=attention_config(settings)
    settings.update(arcAudioMode=mode,arcAudioSettings=attention_settings)
    options={'operation':body.get('operation','fill'),'contextSeconds':body.get('contextSeconds',16),
        'useSurroundingPatterns':body.get('matchMotifs',body.get('useSurroundingPatterns',True)),
        'matchNps':body.get('matchDensity',body.get('matchNps',True)),
        'matchMovement':body.get('matchFlow',body.get('matchMovement',True)),
        'matchRhythm':body.get('matchRhythm',True)}
    plan=plan_continuation(chart['data'],analysis,settings,body.get('ranges',[]),options)
    def work(checkpoint):
        checkpoint(8,'Inspecting musical attention and both map boundaries')
        if mode in ['arcaudio','hybrid','arcaudio3']:analysis['_arcAudioAttention']=cached_attention(analysis,mode,attention_settings,settings.get('analysisCompute'),settings.get('mappingRegions'),settings.get('sourceFocusRegions'))
        with ML_LOCK:
            generated=generate_map(analysis,plan['generationConfig'],MODEL)
            checkpoint(68,'Connecting the new phrase to both existing trajectories')
            result=adapt_continuation(chart['data'],generated['beatmap'],analysis,settings,plan)
        candidate=copy.deepcopy(previous);target=next(c for c in candidate['charts'] if c['id']==chart_id)
        candidate['config']=candidate.get('config',{})|{k:settings[k] for k in ['arcAudioMode','arcAudioSettings','analysisCompute','mappingRegions','backend']}
        if 'sourceFocusRegions' in settings:candidate['config']['sourceFocusRegions']=settings['sourceFocusRegions']
        target['data']=result['beatmap'];target.setdefault('report',{})['continuation']=result['report']
        target['report']['continuation']['engineSelection']=generated['report'].get('engineSelection',{})
        target['report']['generationReportStale']=True
        preview_id=uuid.uuid4().hex
        value={'id':preview_id,'projectId':key,'chartId':chart_id,'baseRevision':previous['revision'],
            'created':time.time(),'project':candidate,'request':body,'report':result['report'],'changes':result['changes'],'status':'pending'}
        write_json(preview_path(key,preview_id),value)
        checkpoint(98,'Draft ready for review; the saved level is unchanged')
        return {'previewId':preview_id,'baseRevision':previous['revision'],'report':result['report'],'summary':result['report'],
            'projectId':key,'previewProject':candidate}
    return start_job('continuation',work,{'audioId':analysis['id'],'projectId':key})

@router.post('/api/projects/{key}/continue')
async def continue_project(key:str,request:Request):return start_continuation(key,await request.json())

@router.post('/api/projects/{key}/continue/{preview}/accept')
def accept_continuation(key:str,preview:str):
    with project_lock(safe_id(key)):return _accept_continuation(key,preview)

def _accept_continuation(key,preview):
    path=preview_path(key,preview);value=json.loads(path.read_text('utf-8'));previous=read_project(key)
    if value['status']!='pending':raise ValueError('This preview has already been accepted or discarded.')
    if previous['revision']!=value['baseRevision']:raise ValueError('The saved map changed. Regenerate against its latest version before accepting.')
    result=persist(previous,value['project'],'Accepted AI continuation')
    value.update(status='accepted',acceptedRevision=result['revision']);write_json(path,value)
    return result

@router.post('/api/projects/{key}/continue/{preview}/regenerate')
def regenerate_continuation(key:str,preview:str):
    value=json.loads(preview_path(key,preview).read_text('utf-8'));return start_continuation(key,value['request'])

@router.post('/api/projects/{key}/continue/{preview}/undo')
def undo_continuation(key:str,preview:str):
    with project_lock(safe_id(key)):return _undo_continuation(key,preview)

def _undo_continuation(key,preview):
    value=json.loads(preview_path(key,preview).read_text('utf-8'));previous=read_project(key)
    if value.get('acceptedRevision')!=previous['revision']:raise ValueError('A later edit exists. Restore the desired version from Map Details instead.')
    old=json.loads((DATA/'projects'/key/'history'/f"r{value['baseRevision']}.json").read_text('utf-8'))
    result=persist(previous,old,'Undid AI continuation');value['status']='undone';write_json(preview_path(key,preview),value);return result

@router.post('/api/projects/{key}/attention/supervision')
async def human_attention_supervision(key:str,request:Request):
    from server import audio_record
    from engine.arcaudio import export_human_supervision
    import asyncio
    body=await request.json();project=read_project(safe_id(key))
    chart=next((c for c in project['charts'] if c['id']==body.get('chartId',project['charts'][0]['id'])),None)
    if not chart:raise ValueError('Chart not found.')
    tolerance=float(body.get('toleranceSeconds',.045))
    if not .005<=tolerance<=.15:raise ValueError('Timing tolerance must be between 5 and 150 milliseconds.')
    _,settings=attention_config({'arcAudioSettings':body.get('settings',{})})
    analysis=copy.deepcopy(audio_record(project['audio']['id']))
    analysis.update(bpm=project['bpm'],songTimeOffset=project.get('songTimeOffset',0))
    def work():
        attention=cached_attention(analysis,'arcaudio',settings)
        return export_human_supervision(analysis,chart['data'],attention,tolerance_seconds=tolerance)
    return await asyncio.to_thread(work)
