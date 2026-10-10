"""Local lifecycle controls, draft previews and independent analysis preferences."""
import asyncio, copy, json, os, pathlib, shutil, subprocess, sys, time, uuid
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from projects import DATA, ROOT, read_project, project_lock, _export_project
from studio import safe_id, persist, preferences

router=APIRouter()
PREVIEWS=DATA/'previews'

@router.get('/api/analysis/runtime')
def analysis_runtime():
    from engine.analysis_runtime import analysis_compute_status
    return analysis_compute_status()

def remove_workspace_folder(path, root):
    """Never recurse through a link or a path outside the explicitly named root."""
    root=pathlib.Path(root).resolve();path=pathlib.Path(path)
    if path.is_symlink() or path.resolve().parent!=root:
        raise ValueError('Refusing to remove a folder outside this workspace collection.')
    if path.exists():shutil.rmtree(path)

def remove_unreferenced_media(project):
    """Remove media belonging solely to a permanently deleted level.

    Active maps and recoverable trash both count as owners. Imported-audio
    history is a separate explicit retention preference.
    """
    owners=[]
    for category in ['projects','trash']:
        for path in (DATA/category).glob('*/project.json'):
            try:owners.append(json.loads(path.read_text('utf-8')))
            except (OSError,ValueError):
                # A damaged ownership index must never cause media loss.
                return {'audioDeleted':False,'assetDeleted':False,'mediaCleanup':'Skipped: an ownership record could not be read.'}
    result={'audioDeleted':False,'assetDeleted':False}
    audio=project.get('audio',{}).get('id');asset=(project.get('asset') or {}).get('id')
    from server import JOBS,JOB_LOCK
    with JOB_LOCK:
        if any(j['status'] in ['queued','running'] and j.get('resources',{}).get('audioId')==audio for j in JOBS.values()):
            return result|{'mediaCleanup':'Retained media used by an active job.'}
    if audio and not preferences().get('storage',{}).get('rememberImportedAudio',False) and not any(p.get('audio',{}).get('id')==audio for p in owners):
        remove_workspace_folder(DATA/'audio'/safe_id(audio),DATA/'audio');result['audioDeleted']=True
    if asset and not any((p.get('asset') or {}).get('id')==asset for p in owners):
        remove_workspace_folder(DATA/'assets'/safe_id(asset),DATA/'assets');result['assetDeleted']=True
    return result

@router.post('/api/trash/purge')
def purge_trash():
    root=DATA/'trash';root.mkdir(exist_ok=True);count=0
    projects_to_clean=[]
    for folder in root.iterdir():
        if folder.is_dir() and len(folder.name)==32:
            if (folder/'project.json').is_file():
                try:projects_to_clean.append(json.loads((folder/'project.json').read_text('utf-8')))
                except (OSError,ValueError):pass
            safe_id(folder.name);remove_workspace_folder(folder,root);count+=1
    for project in projects_to_clean:remove_unreferenced_media(project)
    return {'deleted':count,'permanent':True,'message':f'Permanently removed {count} trashed levels. Shared audio and active levels were preserved.'}

@router.patch('/api/projects/{key}/analysis')
async def set_project_analysis(key:str,request:Request):
    from attention_api import attention_config,regional_config
    body=await request.json()
    allowed={'arcAudioMode','arcAudioSettings','analysisCompute','mappingRegions','learnWholeLevel','learnLighting','sourceFocusRegions'}
    if not isinstance(body,dict) or set(body)-allowed:raise ValueError('Supply analysis settings only; edit notes through Studio.')
    with project_lock(safe_id(key)):
        previous=read_project(key);project=copy.deepcopy(previous)
        config=project.setdefault('config',{});config.update(body)
        mode,settings=attention_config(config);config.update(arcAudioMode=mode,arcAudioSettings=settings)
        from engine.analysis_runtime import normalize_compute
        config['analysisCompute']=normalize_compute(config.get('analysisCompute'))
        for field in ['learnWholeLevel','learnLighting']:
            if field in body and not isinstance(body[field],bool):raise ValueError(f'{field} must be enabled or disabled.')
        project['config']=regional_config(config,project['audio']['duration'])
        return persist(previous,project,'Changed this level’s audio analysis settings')

@router.post('/api/projects/{key}/preview')
async def draft_preview(key:str,request:Request):
    key=safe_id(key);body=await request.json();saved=read_project(key)
    if not isinstance(body,dict):raise ValueError('Supply a Studio draft.')
    candidate=copy.deepcopy(body.get('project') or saved)
    # Drafts can alter visuals and editable map objects, never redirect local assets.
    for field in ['id','audio','asset','created','revision']:
        if field in saved:candidate[field]=copy.deepcopy(saved[field])
        else:candidate.pop(field,None)
    if body.get('chart') is not None:
        target=next((c for c in candidate.get('charts',[]) if c['id']==body.get('chartId')),None)
        if not target:raise ValueError('Select an existing chart for the preview.')
        target['data']=body['chart']
    if not isinstance(candidate.get('charts'),list) or len(candidate['charts'])>25:raise ValueError('Invalid chart collection.')
    if len(json.dumps(candidate))>40_000_000:raise ValueError('The preview is too large.')
    preview=uuid.uuid4().hex;folder=PREVIEWS/preview;PREVIEWS.mkdir(exist_ok=True)
    # Preview cache is independent of saved projects. Retain a bounded recent set.
    for old in sorted((p for p in PREVIEWS.iterdir() if p.is_dir()),key=lambda p:p.stat().st_mtime,reverse=True)[20:]:
        if time.time()-old.stat().st_mtime>3600:remove_workspace_folder(old,PREVIEWS)
    # The installed ArcViewer build consumes V3. The exporter verifies that a
    # V4 draft can be represented without losing native NJS/lane rotations; an
    # unsupported draft gets a precise error instead of a misleading preview.
    path=await asyncio.to_thread(_export_project,key,candidate,folder,DATA/'projects'/key,version='3')
    return {'previewId':preview,'zipUrl':f'/api/previews/{preview}/map.zip','savedRevision':saved['revision'],'savedProjectChanged':False}

@router.get('/api/previews/{key}/map.zip')
def preview_zip(key:str):
    folder=PREVIEWS/safe_id(key);path=next(folder.glob('map-r*.zip'),None)
    if path is None:raise FileNotFoundError('Preview expired. Refresh it from Studio.')
    return FileResponse(path,media_type='application/zip',filename='ArcForge-studio-preview.zip')

@router.get('/api/recognition/status')
def recognition_status():
    from engine.recognition import status
    return status()

@router.post('/api/recognition/install')
def recognition_install():
    from server import start_job
    from engine.recognition import install
    return start_job('recognition',lambda checkpoint:install(progress=checkpoint,cancel_callback=checkpoint.cancelled))

async def lifecycle(request, restart=False):
    from server import JOBS, JOB_LOCK
    if request.client and request.client.host not in {'127.0.0.1','::1','localhost','testclient'}:
        raise HTTPException(403,'Only this computer can control the local engine.')
    with JOB_LOCK:
        if any(job['status'] in {'queued','running'} for job in JOBS.values()):
            raise ValueError('Finish or cancel the active job before stopping the engine.')
    engine=getattr(request.app.state,'localServer',None)
    if engine is None:raise ValueError('Lifecycle controls are available when ArcForge is launched as a local service.')
    if restart:
        flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'DETACHED_PROCESS',0)
        subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'service_restart.py'),str(os.getpid())],
            cwd=ROOT,creationflags=flags,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,close_fds=True)
    async def stop_after_response():
        await asyncio.sleep(.7);engine.should_exit=True
    asyncio.create_task(stop_after_response())
    return {'status':'restarting' if restart else 'stopping','message':'The local engine is restarting.' if restart else 'The local engine is stopping. You can close this tab.'}

@router.post('/api/system/stop')
async def stop_service(request:Request):return await lifecycle(request)

@router.post('/api/system/restart')
async def restart_service(request:Request):return await lifecycle(request,True)
