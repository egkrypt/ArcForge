"""Local studio management, version history and inspectable map analytics."""
import bisect, collections, copy, json, math, os, pathlib, re, statistics, time, uuid
from fastapi import APIRouter, HTTPException, Request
from projects import DATA, ROOT, read_project, write_json, validate_project, project_lock
from modding import BeatClock

router = APIRouter()
SETTINGS = DATA / 'preferences.json'
DEFAULTS = {'theme':'Midnight','accent':'#bda8ff','density':'comfortable',
    'editor':{'wheelBeats':.25,'playerCamera':True,'ghost':True},
    'ai':{'style':'Modern','compatibility':'Modded','dynamicDensity':True,'arcAudioMode':'legacy'},
    'training':{'backend':'auto','threads':8},
    'workflow':{'autoOpenGenerated':True},
    'storage':{'rememberImportedAudio':False,'permanentDeletion':False},
    'profile':{'skill':'Expert+','speed':85,'tech':72,'accuracy':91,'stamina':78,'jumping':65}}
STYLE_NAMES=['OST','Modern','Tech','Speed','Accuracy','Tech/Acc','Stream','Jump','Challenge','Dance',
    'Acc-focused','Flow','Saber control','Powerhouse','High-intensity','Tech-heavy','Light/easy','Experimental']

def safe_id(value):
    if not re.fullmatch(r'[a-f0-9]{32}',str(value)):raise ValueError('Invalid local identifier.')
    return str(value)

def snapshot(project):
    folder=DATA/'projects'/safe_id(project['id'])/'history'
    path=folder/f"r{int(project.get('revision',1))}.json"
    if not path.exists():write_json(path,project)

def persist(previous,project,message='Edited in Studio'):
    with project_lock(previous['id']):return _persist(previous,project,message)

def _persist(previous,project,message):
    current=DATA/'projects'/safe_id(previous['id'])/'project.json'
    if not current.is_file():raise FileNotFoundError('This project was deleted. Reopen an active level before saving.')
    if current.is_file() and json.loads(current.read_text('utf-8')).get('revision',1)!=previous.get('revision',1):
        raise ValueError('This project changed during the operation. Reload its latest revision.')
    snapshot(previous)
    project['id']=previous['id'];project['created']=previous['created']
    project['revision']=int(previous.get('revision',1))+1;project['updated']=time.time()
    project['revisionMessage']=message
    project['validation']=validate_project(project)
    if project['validation']['errors']:raise ValueError('; '.join(project['validation']['errors'][:5]))
    write_json(DATA/'projects'/safe_id(project['id'])/'project.json',project)
    snapshot(project)
    return project

def chart_clock(project,chart):return BeatClock(project['bpm'],chart['data'].get('bpmEvents',[]))

def analytics(project,chart):
    clock=chart_clock(project,chart);offset=float(project.get('songTimeOffset',0));duration=project['audio']['duration']
    from engine.diagnostics import _physical
    notes=[n for n in chart['data'].get('colorNotes',[]) if _physical(n)]
    times=[clock.time(n['b'])+offset for n in notes]
    series=[{'time':i,'nps':sum(i<=t<i+1 for t in times)} for i in range(math.ceil(duration))]
    transitions=[]
    for hand in [0,1]:
        group=sorted((n for n in notes if n['c']==hand),key=lambda n:n['b'])
        for a,b in zip(group,group[1:]):
            dt=clock.time(b['b'])-clock.time(a['b'])
            if dt<=0:continue
            distance=math.hypot(b['x']-a['x'],b['y']-a['y'])*.6
            transitions.append({'time':round(clock.time(b['b'])+offset,4),'hand':hand,
                'recoveryMs':round(dt*1000,2),'travelMeters':round(distance,3),'travelSpeed':round(distance/dt,2)})
    report={};analysis={}
    try:
        from gameplay import analyze_playability
        from server import audio_record
        from audio import ensure_music_analysis
        analysis=ensure_music_analysis(audio_record(project['audio']['id']))
        report=analyze_playability(chart['data'],analysis,
            project.get('config',{})|{'bpm':project['bpm'],'njs':chart.get('njs',20),'jumpDistance':chart.get('jumpDistance',24),'rank':chart.get('rank',9),'songTimeOffset':offset})
    except ImportError:
        report={'issues':[],'scope':'Native file validation only; motion module is not loaded.'}
    categories={'visionBlock':'Readability','faceNote':'Readability','noteOcclusion':'Readability','readingTime':'Readability',
        'parity':'Flow','handTiming':'Flow','sameHandStack':'Flow','saberClash':'Collision','wallOverlap':'Collision','wallOcclusion':'Collision','bombClearance':'Collision',
        'reach':'Physical demand','recoveryReach':'Physical demand'}
    for issue in report.get('issues',[]):issue['category']=categories.get(issue.get('code'),'File structure')
    counts=report.get('counts',{})
    report['metrics']={'Readability':str(sum(v for k,v in counts.items() if categories.get(k)=='Readability'))+' findings',
        'Flow':str(sum(v for k,v in counts.items() if categories.get(k)=='Flow'))+' findings',
        'Collision':str(sum(v for k,v in counts.items() if categories.get(k)=='Collision'))+' findings',
        'Physical demand':str(report.get('p95MotionMetersPerSecond',0))+' m/s (p95)',
        'Reading window':str(report.get('readingTimeMs',0))+' ms'}
    report['definitions']={'Physical demand':'95th percentile of 0.6m-grid position distance divided by same-hand recovery time; excludes real wrist biomechanics.',
        'Readability':'Count of implemented approach visibility and reading-window findings.',
        'Flow':'Count of parity, same-hand timing and stack findings. Zero findings is not a guarantee.',
        'Collision':'Count of sampled sweep proximity and blocking geometry findings.'}
    attacks=sorted(p['time'] for p in analysis.get('musicAnalysis',{}).get('accents',[]))
    def nearest_attack(second):
        i=bisect.bisect_left(attacks,second)
        return min((abs(attacks[j]-second) for j in [i-1,i] if 0<=j<len(attacks)),default=None)
    timing=[{'time':t,'distanceMs':round(nearest_attack(t)*1000,2)} for t in times if attacks]
    lighting=[nearest_attack(clock.time(e['b'])+offset)*1000 for e in chart['data'].get('basicBeatmapEvents',[]) if attacks]
    if timing:
        ordered=sorted(t['distanceMs'] for t in timing)
        report['metrics']['Rhythm alignment']=str(round(statistics.median(ordered),2))+' ms median'
        report['timingEvidence']={'medianMs':round(statistics.median(ordered),2),'p95Ms':ordered[min(len(ordered)-1,int(len(ordered)*.95))],
            'notesWithin40Ms':sum(t['distanceMs']<=40 for t in timing),'measuredNotes':len(timing),'source':'Detected spectral attacks, not a perceptual accuracy score'}
        for t in sorted((t for t in timing if t['distanceMs']>60),key=lambda t:t['distanceMs'],reverse=True)[:20]:
            report.setdefault('issues',[]).append({'category':'Rhythm alignment','severity':'review','time':t['time'],
                'description':f"Nearest detected attack is {t['distanceMs']} ms away; listen to check this hit. Sustains and intentional subdivisions may be valid."})
    if lighting:
        report['metrics']['Lighting sync']=str(round(statistics.median(lighting),2))+' ms median'
        report['lightingEvidence']={'medianAttackDistanceMs':round(statistics.median(lighting),2),'events':len(lighting)}
    bars=collections.defaultdict(list)
    for n in notes:
        if n.get('customData',{}).get('fake'):continue
        bars[int(n['b']//4)].append((round(n['b']%4,3),n['x'],n['y'],n['c'],n['d']))
    signatures={b:tuple(sorted(v)) for b,v in bars.items()}
    report['metrics']['Pattern variety']=f'{len(set(signatures.values()))} / {len(signatures)} distinct bars'
    report['patternEvidence']={'distinctBars':len(set(signatures.values())),'activeBars':len(signatures),
        'adjacentIdenticalBars':sum(signatures.get(b-1)==v for b,v in signatures.items())}
    report['definitions'].update({'Rhythm alignment':'Distance from physical note hits to the nearest detected multiband attack. Review evidence, not a guarantee of musical correctness.',
        'Lighting sync':'Median distance of native lighting events to detected attacks. Fades, strobes and anticipation can deliberately differ.',
        'Pattern variety':'Distinct four-beat timing, position, hand and direction signatures among nonempty bars. Musical motifs may intentionally repeat.',
        'Reading window':'Jump distance divided by twice NJS; approach timing estimate in milliseconds.'})
    return {'chartId':chart['id'],'chartName':chart['name'],'series':series,'transitions':transitions,
        'averageNps':round(len(times)/max(duration,1),3),'peakNps':max((n['nps'] for n in series),default=0),
        'notes':len(times),'njs':chart.get('njs'),'jumpDistance':chart.get('jumpDistance'),
        'reactionTimeMs':round(1000*chart.get('jumpDistance',24)/max(chart.get('njs',20),1)/2),
        'diagnostics':report,'generation':chart.get('report',{}),'validation':validate_project(project)}

@router.get('/api/preferences')
def preferences():
    value=json.loads(SETTINGS.read_text('utf-8')) if SETTINGS.exists() else {}
    defaults=copy.deepcopy(DEFAULTS)
    for key,entry in value.items():
        defaults[key]=(defaults.get(key,{})|entry) if isinstance(entry,dict) and isinstance(defaults.get(key),dict) else entry
    return defaults

@router.put('/api/preferences')
async def set_preferences(request:Request):
    body=await request.json()
    if len(json.dumps(body))>80000:raise ValueError('Preferences are too large.')
    current=preferences()
    for key in DEFAULTS:
        if key in body:current[key]=body[key]
    if not re.fullmatch(r'#[a-fA-F0-9]{6}',str(current.get('accent'))):raise ValueError('Choose a valid accent color.')
    if current.get('theme') not in ['Midnight','Slate','Light','Forest','Ember']:raise ValueError('Choose a supported theme.')
    write_json(SETTINGS,current);return current

@router.get('/api/changelog')
def changelog():return json.loads((ROOT/'integrations/changelog.json').read_text('utf-8'))

@router.get('/api/styles')
def styles():return {'styles':STYLE_NAMES,'method':'Corpus-conditioned placements with explicit rhythm, motion and density goals; presets are goals, not copied pattern templates.'}

@router.get('/api/dashboard')
def dashboard():
    from server import library,training_status
    maps=library()['projects'];jobs=[json.loads(p.read_text('utf-8')) for p in (DATA/'jobs').glob('*.json')]
    jobs=sorted(jobs,key=lambda j:j.get('updated',j['created']),reverse=True)
    return {'recent':maps[:6],'mapCount':len(maps),'activity':jobs[:12],
        'training':training_status()['model'],'version':'1.2.0'}

@router.post('/api/projects/{key}/action')
async def project_action(key:str,request:Request):
    previous=read_project(safe_id(key));body=await request.json();action=body.get('action')
    if action=='rename':
        name=str(body.get('name','')).strip()
        if not name or len(name)>150:raise ValueError('Use a name between 1 and 150 characters.')
        value=copy.deepcopy(previous);value['name']=name
        return persist(previous,value,'Renamed project')
    if action=='duplicate':
        value=copy.deepcopy(previous);newkey=uuid.uuid4().hex
        value.update(id=newkey,name=(previous['name']+' — Copy')[:150],revision=1,created=time.time(),updated=time.time())
        for chart in value['charts']:chart['id']=uuid.uuid4().hex
        folder=DATA/'projects'/newkey;write_json(folder/'project.json',value)
        cover=DATA/'projects'/key/'cover.png'
        if cover.exists():(folder/'cover.png').write_bytes(cover.read_bytes())
        extras=DATA/'projects'/key/'extras'
        if extras.exists():
            (folder/'extras').mkdir()
            for entry in extras.iterdir():
                if entry.is_file():(folder/'extras'/entry.name).write_bytes(entry.read_bytes())
        snapshot(value);return value
    if action=='delete':
        entry=(DATA/'projects'/key).resolve();target=(DATA/'trash'/key).resolve()
        if entry.parent!=(DATA/'projects').resolve() or target.parent!=(DATA/'trash').resolve():raise ValueError('Invalid workspace path.')
        if preferences().get('storage',{}).get('permanentDeletion',False):
            from services_api import remove_workspace_folder,remove_unreferenced_media
            with project_lock(key):remove_workspace_folder(DATA/'projects'/key,DATA/'projects')
            return {'id':key,'deleted':True,'permanent':True,'recoverable':False,**remove_unreferenced_media(previous),
                'message':'Permanently deleted this level. Shared media and explicitly remembered audio were preserved.'}
        target.parent.mkdir(exist_ok=True)
        if target.exists():raise ValueError('A trashed project already uses this identifier.')
        with project_lock(key):entry.rename(target)
        return {'id':key,'deleted':True,'permanent':False,'recoverable':True,'message':'Moved to the workspace trash.'}
    if action=='explorer':
        folder=(DATA/'projects'/key).resolve()
        if not folder.is_relative_to(DATA.resolve()) or not folder.exists():raise ValueError('Project folder is unavailable.')
        if os.name=='nt':os.startfile(str(folder))
        return {'path':str(folder),'opened':os.name=='nt'}
    raise ValueError('Choose a supported project action.')

@router.get('/api/trash')
def trash():
    return {'projects':[{'id':p.parent.name,'name':json.loads(p.read_text('utf-8'))['name']} for p in (DATA/'trash').glob('*/project.json')]}

@router.post('/api/trash/{key}/restore')
def restore_trash(key:str):
    key=safe_id(key);entry=(DATA/'trash'/key).resolve();target=(DATA/'projects'/key).resolve()
    if not entry.is_relative_to(DATA.resolve()) or not target.is_relative_to(DATA.resolve()):raise ValueError('Invalid workspace path.')
    if not entry.exists():raise FileNotFoundError('Trashed project not found.')
    if target.exists():raise ValueError('This project is already active.')
    entry.rename(target);return read_project(key)

@router.get('/api/projects/{key}/analytics')
def project_analytics(key:str,chartId:str|None=None):
    project=read_project(safe_id(key));chart=next((c for c in project['charts'] if c['id']==chartId),project['charts'][0])
    return analytics(project,chart)

@router.get('/api/projects/{key}/versions')
def versions(key:str):
    project=read_project(safe_id(key));snapshot(project)
    result=[]
    for file in (DATA/'projects'/key/'history').glob('r*.json'):
        value=json.loads(file.read_text('utf-8'))
        result.append({'revision':value['revision'],'time':value.get('updated',value['created']),
            'message':value.get('revisionMessage','Studio save'),'notes':sum(len(c['data'].get('colorNotes',[])) for c in value['charts'])})
    return {'current':project['revision'],'versions':sorted(result,key=lambda n:n['revision'],reverse=True)}

@router.post('/api/projects/{key}/versions/{revision}/restore')
def restore_version(key:str,revision:int):
    previous=read_project(safe_id(key));path=DATA/'projects'/key/'history'/f'r{revision}.json'
    if not path.exists():raise FileNotFoundError('Version not found.')
    value=json.loads(path.read_text('utf-8'));return persist(previous,value,f'Restored version {revision}')

@router.get('/api/projects/{key}/versions/{revision}/compare')
def compare_version(key:str,revision:int):
    current=read_project(safe_id(key));path=DATA/'projects'/key/'history'/f'r{revision}.json'
    if not path.exists():raise FileNotFoundError('Version not found.')
    old=json.loads(path.read_text('utf-8'));diff=[]
    for chart in current['charts']:
        before=next((c for c in old['charts'] if c['id']==chart['id']),{'data':{}})
        keys=['colorNotes','sliders','burstSliders','obstacles','bombNotes','basicBeatmapEvents']
        diff.append({'name':chart['name'],'counts':[{ 'object':k,'before':len(before['data'].get(k,[])),'after':len(chart['data'].get(k,[]))} for k in keys]})
    return {'beforeRevision':revision,'afterRevision':current['revision'],'charts':diff}

def merge_range(project,chart,source,start,end,source_bpm=None):
    clock=chart_clock(project,chart);source_clock=BeatClock(source_bpm or project['bpm'],source.get('bpmEvents',[]))
    offset=float(project.get('songTimeOffset',0));a=clock.beat(start-offset);b=clock.beat(end-offset)
    for key in ['colorNotes','bombNotes','obstacles','sliders','burstSliders','basicBeatmapEvents','rotationEvents','colorBoostBeatmapEvents']:
        additions=[]
        for item in source.get(key,[]):
            t=source_clock.time(item['b'])+offset
            if not start<=t<end:continue
            value=copy.deepcopy(item);value['b']=clock.beat(t-offset)
            if 'tb' in value:value['tb']=clock.beat(min(end-offset,source_clock.time(item['tb'])))
            if key=='obstacles':value['d']=max(0,clock.beat(min(end-offset,source_clock.time(item['b']+item.get('d',0))))-value['b'])
            additions.append(value)
        chart['data'][key]=sorted([n for n in chart['data'].get(key,[]) if not a<=n['b']<b]+additions,key=lambda n:n['b'])
    return chart

@router.post('/api/projects/{key}/section')
async def revise_section(key:str,request:Request):
    from server import start_job,ML_LOCK,audio_record,MODEL
    from engine import generate_map
    previous=read_project(safe_id(key));body=await request.json()
    start=float(body.get('start',0));end=float(body.get('end',previous['audio']['duration']))
    if not 0<=start<end<=previous['audio']['duration']:raise ValueError('Choose a valid time range within the song.')
    chart_id=body.get('chartId',previous['charts'][0]['id']);action=body.get('action','regenerate')
    if action=='merge':
        revision=int(body.get('revision',0));path=DATA/'projects'/key/'history'/f'r{revision}.json'
        if not path.exists():raise FileNotFoundError('Source version not found.')
        source_project=json.loads(path.read_text('utf-8'));value=copy.deepcopy(previous)
        chart=next(c for c in value['charts'] if c['id']==chart_id)
        source_chart=next(c for c in source_project['charts'] if c['id']==chart_id)
        merge_range(value,chart,source_chart['data'],start,end,source_project['bpm'])
        return persist(previous,value,f'Merged {start:.2f}–{end:.2f}s from version {revision}')
    if action not in ['regenerate','harder','variation','easier','repair']:raise ValueError('Choose a supported section tool.')
    from attention_api import generation_overrides,attention_config,cached_attention,regional_config
    selected_config=regional_config(generation_overrides(previous.get('config',{}),body),previous['audio']['duration'])
    def work(checkpoint):
        value=copy.deepcopy(previous);chart=next((c for c in value['charts'] if c['id']==chart_id),None)
        if not chart:raise ValueError('Chart not found.')
        if chart.get('legacyMod'):raise ValueError('Convert legacy mod data to V3 before regeneration.')
        analysis=copy.deepcopy(audio_record(previous['audio']['id']))
        from audio import ensure_music_analysis
        analysis=ensure_music_analysis(analysis)
        checkpoint(8,'Analyzing the selected musical phrase')
        config=selected_config|{'bpm':previous['bpm'],'mode':chart['mode'],'rank':chart['rank'],
            'njs':chart.get('njs',20),'jumpDistance':chart.get('jumpDistance',24),'songTimeOffset':previous.get('songTimeOffset',0),
            'nps':float(body.get('nps',chart.get('report',{}).get('requestedNps',8))),
            'style':body.get('style',previous.get('config',{}).get('style','Modern')),
            'seed':int(time.time())%1000000,'temperature':1.05 if action=='variation' else .9}
        if action=='harder':config['nps']=min(15,config['nps']*1.18)
        if action=='easier':config['nps']=max(1,config['nps']*.75)
        mode,knobs=attention_config(config)
        if action!='repair' and mode in ['arcaudio','hybrid','arcaudio3']:
            analysis['_arcAudioAttention']=cached_attention(analysis,mode,knobs,config.get('analysisCompute'),config.get('mappingRegions'),config.get('sourceFocusRegions'))
        with ML_LOCK:
            if action=='repair':
                from gameplay import repair_playability
                result=repair_playability(chart['data'],analysis,config)
                generated=result['beatmap']
            else:
                checkpoint(25,'Composing and testing a new phrase')
                result=generate_map(analysis,config,MODEL);generated=result['beatmap']
                from gameplay import add_events
                from modding import apply_mods
                lighting_config=config|{'_arcForgeMapPlan':result.get('report',{}).get('mapPlan',{})}
                generated,_=apply_mods(add_events(generated,analysis,lighting_config),analysis,lighting_config,previous.get('asset'))
        merge_range(value,chart,generated,start,end)
        from gameplay import analyze_playability,repair_playability
        repaired=repair_playability(chart['data'],analysis,config)
        # Repair the selected phrase without silently changing the rest of a map.
        merge_range(value,chart,repaired['beatmap'],start,end)
        chart.setdefault('report',{})['playtest']=analyze_playability(chart['data'],analysis,config)
        chart['report']['generationReportStale']=True
        chart['report']['sectionRevision']={'action':action,'start':start,'end':end,'changes':repaired['changes']}
        chart['report']['sectionRevision']['engineSelection']=result.get('report',{}).get('engineSelection',{})
        value['config']=value.get('config',{})|{k:config[k] for k in ['arcAudioMode','arcAudioSettings','analysisCompute','mappingRegions','backend']}
        if 'sourceFocusRegions' in config:value['config']['sourceFocusRegions']=config['sourceFocusRegions']
        checkpoint(85,'Checking transitions and preserving a revision')
        latest=read_project(key)
        if latest['revision']!=previous['revision']:raise ValueError('The project changed during generation. Retry on its latest version.')
        persist(previous,value,f'{action.title()} {start:.2f}–{end:.2f}s')
        return {'projectId':key,'project':value,'range':{'start':start,'end':end},'action':action}
    return start_job('generate',work)

@router.post('/api/profile/scores')
async def profile_scores(request:Request):
    body=await request.json();records=body if isinstance(body,list) else body.get('scores',[])
    if not isinstance(records,list) or len(records)>20000:raise ValueError('Upload a JSON list of scores, or an object with a scores list.')
    parsed=[]
    for row in records:
        if not isinstance(row,dict):continue
        accuracy=row.get('accuracy',row.get('acc'))
        if accuracy is None and row.get('modifiedScore') and row.get('maxScore'):accuracy=row['modifiedScore']/row['maxScore']
        if not isinstance(accuracy,(int,float)) or not math.isfinite(accuracy):continue
        accuracy=accuracy/100 if accuracy>1 else accuracy
        if not 0<=accuracy<=1:continue
        parsed.append({'accuracy':accuracy,'nps':row.get('nps'),'name':str(row.get('name',row.get('songName','')))[:150]})
    if not parsed:raise ValueError('No usable accuracy records found. Include accuracy or modifiedScore and maxScore.')
    preferred=[float(r['nps']) for r in parsed if isinstance(r['nps'],(int,float)) and 0<r['nps']<30 and r['accuracy']>=.8]
    result={'scores':len(parsed),'meanAccuracy':round(sum(r['accuracy'] for r in parsed)/len(parsed),4),
        'comfortableNps':round(sum(preferred)/len(preferred),2) if preferred else None,
        'scope':'Score summaries inform density preferences. Scores alone do not identify wrist resets, movement strengths or injuries.'}
    current=preferences();current['profile']['scoreSummary']=result;write_json(SETTINGS,current);return result
