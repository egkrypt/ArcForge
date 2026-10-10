"""ArcForge's local service: real training, persistent editing and native exports."""
import bootstrap
import copy, hashlib, json, math, pathlib, re, shutil, tempfile, threading, time, uuid, zipfile, sys, functools
sys.modules.setdefault('server',sys.modules[__name__])
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from audio import analyze, public_analysis, waveform_window
from engine import scan_dataset, import_archive, train_model, model_status, generate_map
from projects import ROOT, DATA, RANKS, MODES, write_json, read_project, validate_project, export_project, upgrade_v2, jump_settings
from modding import apply_mods, mod_names
from gameplay import add_events

MODEL=ROOT/'models/style'
DATASET=DATA/'dataset'
REFERENCES=ROOT.parent/'Learn from these levels'
for directory in ['audio','dataset','projects','assets','jobs','uploads']: (DATA/directory).mkdir(parents=True,exist_ok=True)
app=FastAPI(title='ArcForge',version='1.3.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://127.0.0.1:8769','http://localhost:8769','http://127.0.0.1:5173','http://localhost:5173'],allow_methods=['*'],allow_headers=['*'])
POOL=ThreadPoolExecutor(max_workers=2,thread_name_prefix='arcforge')
ML_LOCK=threading.Lock()
JOBS={};CANCEL={};JOB_LOCK=threading.RLock()

@app.middleware('http')
async def local_access(request,call_next):
    # Binding to loopback does not stop a remote hostname from resolving there.
    # Reject rebinding Host headers on reads as well as writes.
    hosts=request.headers.getlist('host')
    valid_host=False
    if len(hosts)==1 and hosts[0]==hosts[0].strip():
        try:
            host=urlparse('//'+hosts[0])
            port=host.port
            valid_host=(host.username is None and host.password is None and not host.path
                and not host.query and not host.fragment and (port is None or 1<=port<=65535)
                and host.hostname in {'127.0.0.1','localhost','::1'})
            # Starlette's in-process transport has this extension and non-IP
            # client marker. Uvicorn network requests do not get this exception.
            if (host.hostname=='testserver' and host.username is None and host.password is None
                    and not host.path and not host.query and not host.fragment
                    and (port is None or 1<=port<=65535)
                    and request.client and request.client.host=='testclient'
                    and request.scope.get('server',('',))[0]=='testserver'
                    and 'http.response.debug' in request.scope.get('extensions',{})):
                valid_host=True
        except ValueError:
            pass
    if not valid_host:
        return JSONResponse({'detail':'This local workspace accepts only loopback hostnames.'},status_code=403)
    if request.method not in ['GET','HEAD','OPTIONS']:
        origin=request.headers.get('origin')
        if origin and urlparse(origin).hostname not in ['127.0.0.1','localhost']:
            return JSONResponse({'detail':'This local workspace only accepts local requests.'},status_code=403)
    return await call_next(request)

@app.exception_handler(ValueError)
async def bad_value(request,error):return JSONResponse({'detail':str(error)},status_code=400)
@app.exception_handler(FileNotFoundError)
async def missing(request,error):return JSONResponse({'detail':str(error)},status_code=404)

def identity(value):
    if not re.fullmatch(r'[a-f0-9]{32}',str(value)):raise ValueError('Invalid local identifier.')
    return value

def audio_record(value):
    path=DATA/'audio'/identity(value)/'analysis.json'
    if not path.is_file():raise FileNotFoundError('Audio not found.')
    return copy.deepcopy(_audio_record_cached(str(path),path.stat().st_mtime_ns))

@functools.lru_cache(maxsize=8)
def _audio_record_cached(path,modified):
    from engine.tempo import ensure_tempo_analysis
    return ensure_tempo_analysis(json.loads(pathlib.Path(path).read_text('utf-8')))

def asset_record(value):
    path=DATA/'assets'/identity(value)/'asset.json'
    if not path.is_file():raise FileNotFoundError('Asset not found.')
    return json.loads(path.read_text('utf-8'))

def corpus():
    data=scan_dataset(DATASET)
    return data|{'acceptedCharts':data['mapCount'],'charts':data['mapCount'],'songs':data['archiveCount'],
                 'samples':model_status(MODEL).get('examples',0),'rejected':[{'name':e.get('archive','Archive'),'reason':e.get('error','Invalid')} if isinstance(e,dict) else {'name':'Dataset','reason':e} for e in data['errors']]}

class Cancelled(Exception):pass

def job_update(key,**changes):
    with JOB_LOCK:
        JOBS[key].update(changes,updated=time.time())
        write_json(DATA/'jobs'/f'{key}.json',JOBS[key])

def start_job(kind,work,resources=None):
    key=uuid.uuid4().hex
    with JOB_LOCK:
        JOBS[key]={'id':key,'type':kind,'status':'queued','progress':0,'message':'Queued','created':time.time(),'resources':resources or {}}
        CANCEL[key]=threading.Event()
    def checkpoint(progress=None,message=None,**extra):
        if CANCEL[key].is_set():raise Cancelled()
        changes=extra
        if progress is not None:changes['progress']=round(float(progress),2)
        if message is not None:changes['message']=str(message)
        job_update(key,**changes)
    checkpoint.cancelled=lambda:CANCEL[key].is_set()
    def execute():
        try:
            checkpoint(1,'Starting',status='running')
            result=work(checkpoint)
            checkpoint(100,'Complete',status='completed',result=result)
        except (Cancelled,__import__('training_runtime').TrainingCancelled):job_update(key,status='cancelled',message='Cancelled',error=None)
        except Exception as e:job_update(key,status='failed',message='Stopped',error=str(e))
    POOL.submit(execute)
    return {'jobId':key}

def train(checkpoint,epochs,backend='auto',threads=None):
    with ML_LOCK:
        return train_model(DATASET,MODEL,progress_callback=lambda p:checkpoint(p['progress']*100,p['message'],metrics=p.get('metrics')),epochs=epochs,backend=backend,threads=threads,cancel_callback=getattr(checkpoint,'cancelled',None))

def import_references():
    from engine.dataset import import_references_fast
    return import_references_fast(REFERENCES,DATASET)|{'dataset':corpus()}

async def store_upload(file,folder,max_bytes=400*1024*1024):
    folder.mkdir(parents=True,exist_ok=True)
    suffix=pathlib.Path(file.filename or 'upload').suffix.lower()[:12]
    path=folder/('upload'+suffix);total=0
    with path.open('wb') as output:
        while chunk:=await file.read(1024*1024):
            total+=len(chunk)
            if total>max_bytes:
                output.close();path.unlink(missing_ok=True)
                raise ValueError('Upload exceeds the supported size limit.')
            output.write(chunk)
    await file.close()
    if not total:raise ValueError('The uploaded file is empty.')
    return path

@app.get('/api/status')
def status():
    model=model_status(MODEL)
    return {'name':'ArcForge','version':'1.3.0','local':True,'ready':True,'modelReady':model['ready'],
        'engine':model,'runtime':'Local CPU/CUDA adaptation training and upstream TensorFlow inference','referenceArchives':len(list(DATASET.glob('*.zip'))),
        'viewer':'ArcViewer v0.8.1-beta WebGL','editor':'Native web editor with ChroMapper-derived jump calculations',
        'limitations':['Upstream Inferno weights remain frozen; uploaded maps train the local adaptation network.',
                      'Ranking and headset comfort require human playtesting.','Vivify requires an existing Unity asset bundle; ArcViewer does not render Vivify.']}

@app.get('/api/capabilities')
def capabilities():
    environments=json.loads((ROOT/'integrations/environments.json').read_text('utf-8'))
    if isinstance(environments,dict):environments=environments.get('environments',environments.get('items',[]))
    return {'environments':environments,'modelVersions':['v3'],'environmentModelVersions':['V2','V3'],
        'beatmapVersions':['3','4'],'nativeFormatSupport':{'3':'3.3.0 combined beatmap','4':'4.1.0 beatmap, 4.0.0 lightshow and sample-based AudioData'},
        'modes':sorted(MODES),'audio':'Every audio format decodable by bundled FFmpeg; exports OGG Vorbis.',
        'mods':[{'id':'Chroma','status':'implemented','description':'Audio-triggered colors, custom lights, environment geometry and bounded flicker accents.'},
                {'id':'Noodle Extensions','status':'implemented','description':'Peripheral animated tracks and decorative walls; hit-time notes retain readable positions.'},
                {'id':'Vivify','status':'requires assets','description':'Imports a real Unity bundle and CRC, instantiates supplied prefabs. Preview requires the game.'},
                {'id':'Mapping Extensions','status':'implemented','description':'Precision grid coordinates, with geometry bounds retained for gameplay.'},
                {'id':'CustomJSONData','status':'framework','description':'Editable map custom data; game-side serialization dependency.'},
                {'id':'Heck','status':'framework','description':'Track/point animation and custom-event framework; installed as a dependency of visual mods.'},
                {'id':'Synapse','status':'external integration','description':'Network synchronization and chat plugin, not a map-visual effect or browser renderer.'}],
        'training':{'model':'Audio-conditioned local neural adaptation','upstreamWeights':'Frozen InfernoSaber v1.7.1'},
        'sources':'/api/sources','recognition':{'optional':True,'version':'3.0','statusUrl':'/api/recognition/status'},
        'preview':{'native':True,'arcViewer':(ROOT/'integrations/arcviewer-web/Build/ArcViewer.wasm').is_file(),'draftZip':True,'modRuntime':'Documented object/track/path transforms and lighting; partial browser approximation',
            'vivifyCompiledShaders':False,'unityEnvironmentParity':False}}

@app.get('/api/sources')
def sources():return json.loads((ROOT/'integrations/source-manifest.json').read_text('utf-8'))

@app.get('/api/library')
def library():
    songs=[public_analysis(json.loads(p.read_text('utf-8'))) for p in (DATA/'audio').glob('*/analysis.json')]
    projects=[]
    for p in (DATA/'projects').glob('*/project.json'):
        value=json.loads(p.read_text('utf-8'))
        projects.append({'id':value['id'],'audioId':value['audio']['id'],'name':value['name'],'bpm':value['bpm'],'duration':value['audio']['duration'],
                         'chartCount':len(value['charts']),'coverUrl':f"/api/projects/{value['id']}/cover",'updated':value.get('updated',0)})
    return {'audio':songs,'projects':sorted(projects,key=lambda p:p['updated'],reverse=True)}

@app.post('/api/audio')
async def upload_audio(file:UploadFile=File(...)):
    key=uuid.uuid4().hex;folder=DATA/'audio'/key
    path=await store_upload(file,folder)
    try:
        result=await __import__('asyncio').to_thread(analyze,path,folder,key,pathlib.Path(file.filename or 'Untitled').stem)
        path.unlink(missing_ok=True)
        return public_analysis(result)
    except Exception:
        path.unlink(missing_ok=True);raise

@app.get('/api/audio/{key}')
def get_audio(key:str):
    from audio import ensure_music_analysis
    previous=audio_record(key);value=ensure_music_analysis(previous)
    if previous.get('musicAnalysis',{}).get('version')!=value.get('musicAnalysis',{}).get('version'):
        write_json(DATA/'audio'/identity(key)/'analysis.json',value)
    return public_analysis(value)
@app.get('/api/audio/{key}/song.ogg')
def get_song(key:str):return FileResponse(DATA/'audio'/identity(key)/'song.ogg',media_type='audio/ogg')

@app.get('/api/audio/{key}/waveform')
def detailed_waveform(key:str,start:float=0,end:float=8,points:int=2048):
    path=DATA/'audio'/identity(key)/'song.ogg'
    if not path.is_file():raise FileNotFoundError('Audio not found.')
    return {'id':key,**waveform_window(path,start,end,points)}

@app.get('/api/training/status')
def training_status():
    model=model_status(MODEL)
    runs=[json.loads(p.read_text('utf-8')) for p in (DATA/'jobs').glob('*.json')]
    runs=[r for r in runs if r['type']=='train']
    from training_runtime import runtime_status,recommend_training
    dataset=corpus();runtime=runtime_status()
    return {'dataset':dataset,'model':model,'models':[model],'jobs':runs,'runs':sorted(runs,key=lambda r:r['created'],reverse=True),'metrics':model.get('history',[]),'infernoReady':model['inferno']['inferenceVerified'],'device':model.get('runtime',{}).get('device','Auto · CPU/GPU detection'),'runtime':runtime,'recommendation':recommend_training(dataset),'validation':{'loss':model.get('bestValidationLoss'),'baselineLoss':model.get('baselineValidationLoss'),'examples':model.get('validationExamples'),'split':model.get('heldOutSplit')}}

@app.post('/api/training/import')
async def import_training(request:Request):
    if request.headers.get('content-type','').startswith('application/json'):
        body=await request.json()
        if body.get('folder') is not True:raise ValueError('Choose the supplied reference folder or upload ZIPs.')
        return await __import__('asyncio').to_thread(import_references)
    form=await request.form();accepted=[];rejected=[]
    for file in form.getlist('files'):
        if not hasattr(file,'filename'):continue
        key=uuid.uuid4().hex;folder=DATA/'uploads'/key
        try:
            path=await store_upload(file,folder,200*1024*1024)
            accepted.append(await __import__('asyncio').to_thread(import_archive,path,DATASET))
        except Exception as e:rejected.append({'name':file.filename,'reason':str(e)})
        finally:
            for entry in folder.glob('*'):entry.unlink()
            if folder.exists():folder.rmdir()
    result={'accepted':accepted,'rejected':rejected,'message':f'Imported {len(accepted)} archives.','dataset':corpus()}
    if accepted and len(list(DATASET.glob('*.zip')))>=3:
        from training_runtime import recommend_training
        result.update(start_job('train',lambda checkpoint:train(checkpoint,recommend_training(result['dataset'])['epochs'])))
        result['message']+=' Local retraining started automatically.'
    return result

@app.post('/api/training/start')
async def start_training(request:Request):
    body=await request.json();epochs=int(body.get('epochs',24))
    if not 1<=epochs<=200:raise ValueError('Choose between 1 and 200 training epochs.')
    if not list(DATASET.glob('*.zip')):import_references()
    backend=body.get('backend','auto');threads=body.get('threads')
    if backend not in ['auto','cpu','gpu']:raise ValueError('Choose Auto, CPU or GPU training.')
    if threads is not None and not 1<=int(threads)<=max(1,(__import__('os').cpu_count() or 1)):raise ValueError('Thread count exceeds available CPU threads.')
    return start_job('train',lambda checkpoint:train(checkpoint,epochs,backend,threads))

@app.post('/api/training/playlist')
async def import_playlist(file:UploadFile=File(...)):
    from playlists import parse_playlist,download_playlist
    folder=DATA/'uploads'/uuid.uuid4().hex
    path=await store_upload(file,folder,8*1024*1024);payload=path.read_bytes()
    parse_playlist(payload);path.unlink();folder.rmdir()
    def work(checkpoint):
        result=download_playlist(payload,REFERENCES,
            progress_callback=lambda p:checkpoint(p['progress']*55,p['message'],playlistProgress=p),
            cancel_callback=checkpoint.cancelled)
        checkpoint(57,'Importing verified playlist maps into the training corpus')
        imported=[];failures=[]
        for entry in result['downloaded']+result['skipped']:
            if not entry.get('path'):continue
            checkpoint(58,'Inspecting downloaded human charts')
            try:imported.append(import_archive(pathlib.Path(entry['path']),DATASET))
            except Exception as error:failures.append({'path':entry['path'],'error':str(error)})
        if not imported:raise ValueError('No usable playlist maps imported. '+json.dumps(result['errors'][:3]))
        from training_runtime import recommend_training
        recommendation=recommend_training(corpus())
        def training_checkpoint(progress=None,message=None,**extra):checkpoint(60+(progress or 0)*.39,message,**extra)
        training_checkpoint.cancelled=checkpoint.cancelled
        trained=train(training_checkpoint,recommendation['epochs'])
        return {'playlist':result,'imported':len(imported),'importErrors':failures,'training':trained,'recommendation':recommendation}
    return start_job('playlist',work)

@app.get('/api/jobs/{key}')
def get_job(key:str):
    identity(key)
    if key in JOBS:return copy.deepcopy(JOBS[key])
    path=DATA/'jobs'/f'{key}.json'
    if not path.exists():raise FileNotFoundError('Job not found.')
    result=json.loads(path.read_text('utf-8'))
    if result['status'] in ['queued','running']:result.update(status='failed',error='The service restarted during this job. Please retry.')
    return result

@app.post('/api/jobs/{key}/cancel')
def cancel_job(key:str):
    value=get_job(key)
    if value['status'] not in ['queued','running']:return {'status':value['status']}
    CANCEL[key].set()
    return {'status':'cancelling','message':'Cancellation will complete at the next safe checkpoint.'}

@app.post('/api/generate')
async def generation(request:Request):
    config=await request.json()
    if not isinstance(config,dict):raise ValueError('Generation settings must be an object.')
    analysis=audio_record(config.get('audioId'))
    from attention_api import attention_config,cached_attention,generation_overrides,regional_config
    config=generation_overrides(config,{})
    version=str(config.get('beatmapVersion','3'))
    if version not in {'3','3.3.0','4','4.0.0','4.1.0'}:raise ValueError('Choose Beatmap V3 or V4.1 export.')
    config['beatmapVersion']=version.split('.')[0]
    if not isinstance(config.get('dynamicNjs',False),bool):raise ValueError('Dynamic NJS must be enabled or disabled.')
    from engine.regions import validate_regions
    attention_mode,attention_settings=attention_config(config)
    config=regional_config(config,analysis['duration'])
    if attention_mode in ['arcaudio','hybrid','arcaudio3']:config.update(arcAudioMode=attention_mode,arcAudioSettings=attention_settings)
    target=config.get('compatibility','Modded')
    if target not in ['Vanilla','Ranked-compatible','Modded','Unrestricted']:raise ValueError('Choose a supported compatibility target.')
    if target=='Vanilla':config['mods']=[]
    elif target=='Ranked-compatible':config['mods']=[m for m in config.get('mods',[]) if m in ['Chroma','CustomJSONData']]
    selected_mods=mod_names(config.get('mods',[]))
    allowed_mods={m['id'] for m in capabilities()['mods']}
    if selected_mods-allowed_mods:raise ValueError('Unsupported mod requirements: '+', '.join(sorted(selected_mods-allowed_mods)))
    from engine.lighting_plan import STYLES as LIGHTING_STYLES
    if not isinstance(config.get('lightingStyle','AI'),str) or config.get('lightingStyle','AI') not in LIGHTING_STYLES:raise ValueError('Choose a supported lighting style.')
    if {'Mapping Extensions','Noodle Extensions'}<=selected_mods:
        raise ValueError('Choose Noodle Extensions or Mapping Extensions for a generated chart. Their game runtimes are incompatible; both formats remain editable separately.')
    if 'Synapse' in selected_mods:raise ValueError('Synapse handles network play rather than map effects. It is documented in the editor; do not add it as a chart requirement.')
    diffs=[d for d in config.get('difficulties',[]) if d.get('enabled',True)]
    modes=config.get('modes') or ['Standard']
    if not diffs or len(diffs)>5:raise ValueError('Select one to five named native difficulty slots.')
    ranks=[int(d.get('rank',9)) for d in diffs]
    if len(set(ranks))!=len(ranks) or any(r not in RANKS for r in ranks):raise ValueError('Assign each difficulty a unique native slot.')
    if len(set(modes))!=len(modes) or any(m not in MODES for m in modes):raise ValueError('Choose a supported game mode once.')
    for d in diffs:
        if not 1<=float(d.get('nps',10))<=15:raise ValueError('Notes per second must be between 1 and 15.')
    bpm=float(config['bpm'] if config.get('bpm') is not None else analysis['bpm'])
    if not 20<=bpm<=1000:raise ValueError('BPM must be between 20 and 1000.')
    for field,lower,upper in [('njs',5,40),('jumpDistance',1,100),('previewStart',0,analysis['duration']),('startOffset',0,analysis['duration'])]:
        if config.get(field) is not None and not lower<=float(config[field])<=upper:raise ValueError(f'{field} is outside its allowed range.')
    allowed_env={e['id'] for e in capabilities()['environments']}
    if config.get('environment','DefaultEnvironment') not in allowed_env:raise ValueError('Choose a known game environment.')
    asset=asset_record(config['assetId']) if config.get('assetId') else None
    if 'Vivify' in mod_names(config.get('mods',[])) and asset is None:raise ValueError('Import a Vivify bundle, Unity CRC and prefab path before selecting Vivify.')
    def work(checkpoint):
        if not model_status(MODEL)['ready']:
            if not list(DATASET.glob('*.zip')):import_references()
            train(checkpoint,24)
        from audio import ensure_music_analysis
        source=ensure_music_analysis(copy.deepcopy(analysis));source['bpm']=bpm
        supplied_offset=float(config.get('startOffset') if config.get('startOffset') is not None else source['offset'])
        bpm_override=abs(bpm-float(analysis['bpm']))>.001
        offset_delta=supplied_offset-float(source['offset'])
        # The wizard always sends displayed BPM/offset values. Merely echoing
        # detected values must not erase drift and an entire tempo timeline.
        if bpm_override:
            source['beatTimes']=[round(supplied_offset+i*60/bpm,6) for i in range(int(source['duration']*bpm/60)+2)]
            source['tempoTimeline']=[{'start':0,'end':source['duration'],'bpm':bpm,'confidence':1.,'source':'user_override'}]
        elif abs(offset_delta)>.00001:
            source['beatTimes']=[round(t+offset_delta,6) for t in source.get('beatTimes',[]) if 0<=t+offset_delta<=source['duration']]
        source['offset']=supplied_offset
        source['beatMarkers']=source.get('beatTimes',[])
        if attention_mode in ['arcaudio','hybrid','arcaudio3'] and not config.get('manual'):
            checkpoint(4,'Selecting musical focus with '+('ArcAudio 3.0' if attention_mode=='arcaudio3' else 'ArcAudio v2.0'))
            source['_arcAudioAttention']=cached_attention(source,attention_mode,attention_settings,config.get('analysisCompute'),config.get('mappingRegions'),config.get('sourceFocusRegions'))
        key=uuid.uuid4().hex;charts=[];total=len(modes)*len(diffs)
        with ML_LOCK:
            for mode in modes:
                for d in diffs:
                    count=len(charts);checkpoint(5+85*count/total,f"Composing {d['name']} · {mode}")
                    settings=config|{'mode':mode,'label':d['name'],'rank':int(d.get('rank',9)),'nps':float(d.get('nps',10)),'bpm':bpm,
                        'backend':config.get('backend','hybrid'),'seed':int(config.get('seed',42))+count*107,
                        'events':config.get('events',{})|{'dotBlocks':bool(config.get('events',{}).get('dots',False))}}
                    settings['temperature']={'Flow':.75,'Technical':.9,'Expressive':1.1}.get(config.get('style'),.9)
                    settings['arcs']=config.get('events',{}).get('arcs',True)
                    settings['chains']=config.get('events',{}).get('chains',False)
                    settings['_learnedLighting']=model_status(MODEL).get('lighting',{})
                    if config.get('learnLighting') or config.get('learnWholeLevel'):settings['_compositionModelDir']=str(MODEL)
                    if config.get('manual'):
                        generated={'beatmap':{'version':'3.3.0','bpmEvents':[],'rotationEvents':[],'colorNotes':[],
                            'bombNotes':[],'obstacles':[],'sliders':[],'burstSliders':[],'waypoints':[],
                            'basicBeatmapEvents':[],'colorBoostBeatmapEvents':[],'lightColorEventBoxGroups':[],
                            'lightRotationEventBoxGroups':[],'lightTranslationEventBoxGroups':[],
                            'basicEventTypesWithKeywords':{'d':[]},'useNormalEventsAsCompatibleEvents':True,'customData':{}},
                            'report':{'passed':True,'issues':[],'backend':'Manual blank chart','noteCount':0}}
                    else:generated=generate_map(source,settings,MODEL)
                    if not generated['report']['passed']:raise ValueError('Generated chart failed flow checks: '+', '.join(generated['report']['issues']))
                    lighting_settings=settings|{'_arcForgeMapPlan':generated['report'].get('mapPlan',{})}
                    chart,modinfo=apply_mods(add_events(generated['beatmap'],source,lighting_settings),source,lighting_settings,asset)
                    from gameplay import recommend_jump_settings,analyze_playability,repair_playability
                    measured_motion=analyze_playability(chart,source,settings|{'njs':20,'jumpDistance':24})
                    recommendation=recommend_jump_settings(generated['report']|measured_motion,settings)
                    njs=float(config.get('njs') or recommendation['njs'])
                    jd=config.get('jumpDistance') or recommendation['jumpDistance'];_,actual=jump_settings(bpm,njs,jd)
                    final_settings=settings|{'njs':njs,'jumpDistance':actual}
                    repaired=repair_playability(chart,source,final_settings)
                    chart=repaired['beatmap']
                    if not repaired['report']['passed']:raise ValueError('The draft still has motion errors after repair. Adjust its density or movement target and retry.')
                    from engine.jump import apply_dynamic_njs
                    chart,jump_plan=apply_dynamic_njs(chart,source,settings,generated['report'],njs,actual)
                    generated['report']['playtest']=repaired['report']
                    generated['report']['repairs']=repaired['changes']
                    generated['report']['jumpRecommendation']=recommendation
                    generated['report']['jumpPlan']=jump_plan
                    if generated['report'].get('mapPlan'):
                        generated['report']['mapPlan']['jumpStrategy']={'baseNjs':njs,'baseJumpDistance':actual,
                            'dynamic':jump_plan['applied'],'eventCount':jump_plan['eventCount'],'reason':jump_plan['reason']}
                    charts.append({'id':uuid.uuid4().hex,'name':str(d['name'])[:100],'rank':int(d.get('rank',9)),
                        'difficulty':RANKS[int(d.get('rank',9))],'mode':mode,'njs':njs,'jumpDistance':actual,
                        'data':chart,'report':generated['report'],'modInfo':modinfo})
        checkpoint(93,'Checking geometry and writing native map files')
        project={'id':key,'name':str(config.get('name') or source['name'])[:150],'bpm':bpm,
            'audio':public_analysis(source),'environment':config.get('environment','DefaultEnvironment'),
            'charts':charts,'config':config,'asset':asset,'created':time.time(),'updated':time.time(),'revision':1}
        project['validation']=validate_project(project)
        if project['validation']['errors']:raise ValueError('; '.join(project['validation']['errors'][:5]))
        checkpoint(96,'Saving your editable project')
        write_json(DATA/'projects'/key/'project.json',project)
        export_project(key)
        return {'projectId':key,'project':project,'summary':{'charts':len(charts),'notes':sum(len(c['data']['colorNotes']) for c in charts)}}
    return start_job('generate',work,{'audioId':analysis['id']})

@app.get('/api/projects/{key}')
def get_project(key:str):return read_project(identity(key))

@app.post('/api/projects/import')
async def import_project(file:UploadFile=File(...)):
    key=uuid.uuid4().hex;folder=DATA/'projects'/key;path=await store_upload(file,folder)
    def work():
        with zipfile.ZipFile(path) as z:
            if len(z.infolist())>2000 or sum(e.file_size for e in z.infolist())>1024*1024*1024:raise ValueError('The expanded archive is too large.')
            entries={}
            for entry in z.infolist():
                p=pathlib.PurePosixPath(entry.filename.replace('\\','/'))
                if p.is_absolute() or '..' in p.parts or entry.file_size>400*1024*1024:raise ValueError('Unsafe or oversized archive entry.')
                if not entry.is_dir():
                    if p.name.lower() in entries:raise ValueError('Archive has ambiguous duplicate filenames.')
                    entries[p.name.lower()]=entry.filename
            def contents(name):
                entry=entries.get(pathlib.PurePosixPath(str(name)).name.lower())
                if not entry:raise ValueError(f'Missing {name} in the level ZIP.')
                return z.read(entry)
            from engine.formats import (read_native_json, info_v4_to_v2, decode_v4, audio_data_to_tempo)
            original_info=read_native_json(contents('Info.dat'));info=original_info;native_audio=None
            info4=str(info.get('_version',info.get('version',''))).startswith('4')
            if info4:
                info=info_v4_to_v2(original_info)
                native_audio=read_native_json(contents(original_info['audio']['audioDataFilename']))
                tempo=audio_data_to_tempo(native_audio)
                info['_beatsPerMinute']=tempo['bpm'];info['_songTimeOffset']=tempo['songTimeOffset']
            audio_id=uuid.uuid4().hex;afolder=DATA/'audio'/audio_id;afolder.mkdir(parents=True)
            source=afolder/('source'+pathlib.Path(info['_songFilename']).suffix)
            source.write_bytes(contents(info['_songFilename']))
            analysis=analyze(source,afolder,audio_id,str(info.get('_songName','Imported level')));source.unlink()
            charts=[];warnings=[]
            for group in info.get('_difficultyBeatmapSets',[]):
                mode=group['_beatmapCharacteristicName']
                if mode not in MODES:continue
                for d in group['_difficultyBeatmaps']:
                    raw=read_native_json(contents(d['_beatmapFilename']))
                    native4=str(raw.get('version','')).startswith('4')
                    lightshow=read_native_json(contents(d['_lightshowFilename'])) if native4 and d.get('_lightshowFilename') else None
                    if native4 and not info4:raise ValueError('Native V4 beatmaps require Info4 and shared AudioData metadata.')
                    native=native4 or str(raw.get('version','')).startswith('3')
                    data=decode_v4(raw,lightshow,native_audio,info['_beatsPerMinute']) if native4 else upgrade_v2(raw)
                    data['colorNotes'].sort(key=lambda n:(n.get('b',0),n.get('c',0)))
                    requirements=d.get('_customData',{}).get('_requirements',[])
                    legacy=not native and (bool(requirements) or bool(raw.get('_customData',{})))
                    if legacy:warnings.append('Legacy V2 mod charts are preserved for viewing and unchanged export; use a V3 conversion for editing.')
                    rank=int(d.get('_difficultyRank',9));njs=float(d.get('_noteJumpMovementSpeed',20) or 20)
                    half=4
                    while njs*60/info['_beatsPerMinute']*half>17.999:half/=2
                    distance=2*njs*60/info['_beatsPerMinute']*max(.25,half+float(d.get('_noteJumpStartBeatOffset',0)))
                    charts.append({'id':uuid.uuid4().hex,'name':d.get('_customData',{}).get('_difficultyLabel',d['_difficulty']),
                        'rank':rank,'difficulty':d['_difficulty'],'mode':mode,'njs':njs,'jumpDistance':distance,'data':data,
                        'modInfo':{'requirements':requirements,'dependencies':[]},'originalData':raw if not native else None,
                        'importBeatmapCustomData':d.get('_customData',{}),'importColorSchemeIdx':d.get('_beatmapColorSchemeIdx',-1),
                        'importEnvIndex':d.get('_environmentNameIdx',0),
                        'originalV4Data':raw if native4 else None,'originalV4Lightshow':lightshow if native4 else None,
                        'importV4Authors':d.get('_v4Authors') if native4 else None,
                        'originalCanonicalHash':hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                        'legacyMod':legacy})
            if not charts:raise ValueError('No supported charts were found in this ZIP.')
            project={'id':key,'name':str(info.get('_songName','Imported level')),'bpm':float(info['_beatsPerMinute']),
                'audio':public_analysis(analysis),'charts':charts,'songTimeOffset':float(info.get('_songTimeOffset',0)),
                'environment':info.get('_environmentName','DefaultEnvironment'),'config':{'artist':info.get('_songAuthorName',''),
                    'author':info.get('_levelAuthorName',''),'previewStart':info.get('_previewStartTime',0),
                    'beatmapVersion':'4' if info4 else '3'},
                'created':time.time(),'updated':time.time(),'revision':1,'importWarnings':list(set(warnings)),
                'importCustomData':info.get('_customData',{}),'importColorSchemes':info.get('_colorSchemes',[]),
                'importEnvironmentNames':info.get('_environmentNames',[])}
            if info4:
                project['importV4Info']=original_info;project['nativeAudioData']=native_audio
                project['importWarnings'].append('Native V4 indexed gameplay, split lightshows and shared tempo are preserved. Older mod and viewer versions may require V3; V4-only features cannot be silently downgraded.')
                preview_name=original_info.get('songPreviewFilename')
                if preview_name and preview_name!=original_info.get('audio',{}).get('songFilename'):
                    safe_preview=pathlib.PurePosixPath(preview_name.replace('\\','/')).name
                    extra=folder/'extras';extra.mkdir(exist_ok=True);(extra/safe_preview).write_bytes(contents(preview_name))
            from PIL import Image
            try:Image.open(__import__('io').BytesIO(contents(info.get('_coverImageFilename','cover.png')))).convert('RGB').resize((512,512)).save(folder/'cover.png')
            except (ValueError,OSError):pass
            for name,entry in entries.items():
                if name.endswith('.vivify'):
                    extra=folder/'extras';extra.mkdir(exist_ok=True);(extra/pathlib.PurePosixPath(entry).name).write_bytes(z.read(entry))
            write_json(folder/'project.json',project)
            return project
    try:return await __import__('asyncio').to_thread(work)
    finally:path.unlink(missing_ok=True)

@app.put('/api/projects/{key}')
async def save_project(key:str,request:Request):
    from projects import project_lock
    body=await request.json()
    with project_lock(identity(key)):return _save_project(key,body)

def _save_project(key,body):
    previous=read_project(identity(key))
    if body.get('revision') is not None and body['revision']!=previous.get('revision',1):raise HTTPException(409,'This project changed in another tab. Reload it before saving.')
    body['id']=key;body['audio']=previous['audio'];body['asset']=previous.get('asset');body['created']=previous['created']
    for chart in body.get('charts',[]):
        old=next((c for c in previous['charts'] if c['id']==chart.get('id')),None)
        if old and old.get('legacyMod'):
            digest=hashlib.sha256(json.dumps(chart['data'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if digest!=old.get('originalCanonicalHash'):raise ValueError('Legacy V2 mod charts need a V3 conversion before editing. The original chart remains preserved.')
    body['updated']=time.time();body['revision']=previous.get('revision',1)+1
    for chart in body.get('charts',[]):chart['data']['colorNotes'].sort(key=lambda n:(n.get('b',0),n.get('c',0)))
    body['validation']=validate_project(body)
    if body['validation']['errors']:raise ValueError('; '.join(body['validation']['errors'][:5]))
    from studio import snapshot
    snapshot(previous)
    write_json(DATA/'projects'/key/'project.json',body)
    snapshot(body)
    return body

@app.get('/api/projects/{key}/validate')
def project_validation(key:str):return validate_project(read_project(identity(key)))

@app.get('/api/projects/{key}/export')
def project_export(key:str,version:str|None=None):
    project=read_project(identity(key));path=export_project(key,version=version)
    filename=re.sub(r'[^\w .-]','',project['name'])[:100] or 'Level'
    return FileResponse(path,media_type='application/zip',filename=filename+'.zip')

@app.get('/api/projects/{key}/cover')
def project_cover(key:str):
    project=read_project(identity(key));path=DATA/'projects'/key/'cover.png'
    if not path.exists():
        from projects import cover
        cover(path,project['name'])
    return FileResponse(path,media_type='image/png')

@app.post('/api/assets')
async def upload_asset(file:UploadFile=File(...),checksum:int=Form(...),prefab:str=Form(...),platform:str=Form('windows2021')):
    if not (file.filename or '').lower().endswith('.vivify'):raise ValueError('Choose a compiled .vivify Unity asset bundle.')
    if not 0<=checksum<=4294967295:raise ValueError('Unity CRC must be an unsigned 32-bit integer from the bundle manifest.')
    if not prefab.strip() or '..' in prefab or '\\' in prefab:raise ValueError('Enter the bundle prefab asset path, for example assets/prefabs/halo.prefab.')
    if platform not in ['windows2019','windows2021']:raise ValueError('Select Windows Unity 2019 or 2021.')
    key=uuid.uuid4().hex;folder=DATA/'assets'/key
    path=await store_upload(file,folder)
    with path.open('rb') as check:header=check.read(8)
    if not header.startswith((b'UnityFS',b'UnityRaw',b'UnityWeb')):
        path.unlink();raise ValueError('The file is not a recognized compiled Unity bundle.')
    path.replace(folder/'bundle.vivify')
    result={'id':key,'name':file.filename,'checksum':checksum,'prefab':prefab.lower().strip(),'platform':platform,'size':(folder/'bundle.vivify').stat().st_size,'checksumVerification':'User-supplied Unity CRC; verified by the game at load time.'}
    write_json(folder/'asset.json',result);return result

@app.get('/api/assets')
def assets():return {'assets':[json.loads(p.read_text('utf-8')) for p in (DATA/'assets').glob('*/asset.json')]}

from studio import router as studio_router
app.include_router(studio_router)
from attention_api import router as attention_router
app.include_router(attention_router)
from services_api import router as services_router
app.include_router(services_router)

viewer=ROOT/'integrations/arcviewer-web'
if viewer.exists():app.mount('/viewer',StaticFiles(directory=viewer,html=True),name='arcviewer')

@app.get('/{path:path}')
def frontend(path:str):
    if path.startswith('api/'):raise HTTPException(404,'Endpoint not found.')
    folder=ROOT/'frontend/dist';candidate=(folder/path).resolve()
    if candidate.is_relative_to(folder.resolve()) and candidate.is_file():return FileResponse(candidate)
    if (folder/'index.html').is_file():return FileResponse(folder/'index.html')
    return HTMLResponse('<html><body>ArcForge frontend is being built. The local service is running.</body></html>',status_code=503)

if __name__=='__main__':
    import uvicorn
    engine=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=8769,log_level='info'))
    app.state.localServer=engine
    engine.run()
