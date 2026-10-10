"""Local .bplist import with official BeatSaver resolution and verified chart hashes.

No supplied URL, script, filename or ZIP path is executed. Downloaded maps stay in
the user-selected reference folder, and cancellation preserves completed maps.
"""
from __future__ import annotations
import hashlib, io, json, pathlib, re, socket, tempfile, time, urllib.error, urllib.parse, urllib.request, zipfile, ipaddress
from training_runtime import check_cancel
from engine.dataset import read_archive, MAX_ARCHIVE_BYTES

MAX_PLAYLIST_BYTES=8*1024*1024
MAX_SONGS=2000
API='https://api.beatsaver.com'

def parse_playlist(payload):
    if isinstance(payload,dict): data=payload
    else:
        if isinstance(payload,pathlib.Path): payload=payload.read_bytes()
        if isinstance(payload,str): payload=payload.encode('utf-8')
        if not isinstance(payload,bytes) or len(payload)>MAX_PLAYLIST_BYTES: raise ValueError('Playlist must be JSON smaller than 8 MB')
        try:data=json.loads(payload.decode('utf-8-sig'))
        except (UnicodeDecodeError,json.JSONDecodeError) as e:raise ValueError('Invalid .bplist JSON') from e
    if not isinstance(data,dict) or not isinstance(data.get('songs'),list): raise ValueError('Playlist needs a songs array')
    if len(data['songs'])>MAX_SONGS: raise ValueError('Playlist exceeds the 2000-song limit')
    songs,invalid,seen=[],[],set()
    for i,song in enumerate(data['songs']):
        if not isinstance(song,dict): invalid.append({'index':i,'error':'Song must be an object'});continue
        raw=str(song.get('hash','')).lower().removeprefix('custom_level_')
        key=str(song.get('key',song.get('id',''))).lower()
        digest=raw if re.fullmatch(r'[0-9a-f]{40}',raw) else None
        key=key if re.fullmatch(r'[0-9a-f]{1,8}',key) else None
        if not digest and not key: invalid.append({'index':i,'error':'Missing valid BeatSaver SHA-1 hash or map key'});continue
        identity=digest or 'key:'+key
        if identity in seen:continue
        seen.add(identity);songs.append({'hash':digest,'key':key,'songName':str(song.get('songName',''))[:300]})
    return {'title':str(data.get('playlistTitle',data.get('title','Imported playlist')))[:300],
            'author':str(data.get('playlistAuthor',data.get('author','')))[:200], 'songs':songs,'invalid':invalid,'songCount':len(songs)}

def validate_url(url,resolve_dns=False):
    p=urllib.parse.urlparse(str(url));host=(p.hostname or '').lower()
    if p.scheme!='https' or p.username or p.password or p.port not in [None,443] or not (host=='beatsaver.com' or host.endswith('.beatsaver.com')):
        raise ValueError('Only HTTPS BeatSaver API/CDN URLs are allowed')
    if resolve_dns:
        addresses=socket.getaddrinfo(host,443,type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):raise ValueError('BeatSaver URL resolved to a non-public address')
    return str(url)

class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        validate_url(newurl,True)
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def fetch_bytes(url,limit=MAX_PLAYLIST_BYTES,cancel_callback=None):
    validate_url(url,True);check_cancel(cancel_callback)
    request=urllib.request.Request(url,headers={'User-Agent':'ArcForge/1.1 (local playlist importer)','Accept':'application/json, application/zip'})
    opener=urllib.request.build_opener(_SafeRedirect())
    for attempt in range(3):
        try:
            with opener.open(request,timeout=30) as response:
                if int(response.headers.get('Content-Length','0'))>limit:raise ValueError('Download exceeds its size limit')
                chunks,total=[],0
                while True:
                    check_cancel(cancel_callback);block=response.read(256*1024)
                    if not block:break
                    total+=len(block)
                    if total>limit:raise ValueError('Download exceeds its size limit')
                    chunks.append(block)
                return b''.join(chunks)
        except urllib.error.HTTPError as error:
            if error.code not in [429,500,502,503,504] or attempt==2:raise
            delay=min(15,max(1,int(error.headers.get('Retry-After','2'))))
            for _ in range(delay*10):check_cancel(cancel_callback);time.sleep(.1)
        except (TimeoutError,urllib.error.URLError):
            if attempt==2:raise
            for _ in range(10):check_cancel(cancel_callback);time.sleep(.1)

def beatsaver_hash(path):
    """SongCore SHA-1: UTF-8 Info text then chart bytes in Info ordering.

    For v4 the audio-data JSON and each gameplay/lightshow file are included.
    The music stream and ZIP container are not part of BeatSaver's map hash.
    """
    with zipfile.ZipFile(path) as z:
        if len(z.infolist())>2000 or sum(e.file_size for e in z.infolist())>1024**3:raise ValueError('Unsafe expanded archive size')
        files={}
        for e in z.infolist():
            p=pathlib.PurePosixPath(e.filename.replace('\\','/'))
            if p.is_absolute() or '..' in p.parts or e.file_size>150*1024*1024:raise ValueError('Unsafe archive entry')
            if not e.is_dir():
                if str(p).lower() in files:raise ValueError('Duplicate archive filename')
                files[str(p).lower()]=e.filename
        infos=[name for name in files if pathlib.PurePosixPath(name).name=='info.dat']
        if len(infos)!=1:raise ValueError('Archive must contain exactly one Info.dat')
        info_name=infos[0];raw=z.read(files[info_name]);text=raw.decode('utf-8-sig');info=json.loads(text)
        digest=hashlib.sha1(text.encode('utf-8'));prefix=pathlib.PurePosixPath(info_name).parent
        if str(info.get('version',info.get('_version','2'))).startswith('4'):
            paths=[info.get('audio',{}).get('audioDataFilename')]
            for c in info.get('difficultyBeatmaps',[]):paths.extend([c.get('beatmapDataFilename'),c.get('lightshowDataFilename')])
        else:paths=[d.get('_beatmapFilename') for g in info.get('_difficultyBeatmapSets',[]) for d in g.get('_difficultyBeatmaps',[])]
        if not paths:raise ValueError('Info.dat lists no map files')
        for name in paths:
            if not isinstance(name,str):raise ValueError('Invalid referenced chart filename')
            p=pathlib.PurePosixPath(name.replace('\\','/'))
            if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe referenced map path')
            entry=files.get(str(prefix/p).lower())
            if not entry:raise ValueError('Missing referenced chart: '+name)
            digest.update(z.read(entry))
        if z.testzip() is not None:raise ValueError('Corrupt archive CRC')
        return digest.hexdigest()

def resolve_song(song,fetcher=fetch_bytes,cancel_callback=None):
    url=API+'/maps/'+('hash/'+song['hash'] if song.get('hash') else 'id/'+song['key'])
    data=json.loads(fetcher(url,MAX_PLAYLIST_BYTES,cancel_callback).decode('utf-8'))
    if isinstance(data,list):data=next((d for d in data if isinstance(d,dict)),{})
    if not isinstance(data,dict):raise ValueError('Invalid BeatSaver map response')
    versions=data.get('versions',[])
    version=next((v for v in versions if song.get('hash') and str(v.get('hash','')).lower()==song['hash']),None)
    if not song.get('hash'):version=next((v for v in versions if v.get('state')=='Published'),None)
    if not version:raise ValueError('Requested map version is unavailable')
    digest=str(version.get('hash','')).lower()
    if not re.fullmatch('[0-9a-f]{40}',digest):raise ValueError('Invalid BeatSaver version hash')
    validate_url(version.get('downloadURL',''))
    return {'hash':digest,'key':str(data.get('id','')),'songName':str(data.get('metadata',{}).get('songName',data.get('name','')))[:300],
            'downloadURL':version['downloadURL'],'automapper':bool(data.get('automapper')) or data.get('declaredAi') not in [None,'None']}

def download_playlist(payload,destination,progress_callback=None,cancel_callback=None,fetcher=fetch_bytes):
    playlist=parse_playlist(payload);folder=pathlib.Path(destination).resolve();folder.mkdir(parents=True,exist_ok=True)
    if not playlist['songs']:raise ValueError('Playlist contains no valid songs')
    existing={};downloaded=[];skipped=[];errors=list(playlist['invalid'])
    for path in folder.glob('*.zip'):
        check_cancel(cancel_callback)
        try:existing[beatsaver_hash(path)]=path
        except (ValueError,KeyError,zipfile.BadZipFile,UnicodeDecodeError,json.JSONDecodeError):continue
    total=len(playlist['songs'])
    for i,song in enumerate(playlist['songs']):
        check_cancel(cancel_callback)
        if progress_callback:progress_callback({'phase':'playlist','progress':i/total,'completed':i,'total':total,'downloaded':len(downloaded),'message':'Resolving '+(song.get('songName') or song.get('key') or song['hash'])})
        try:
            if song.get('hash') in existing:
                skipped.append({'hash':song['hash'],'reason':'Already present','path':str(existing[song['hash']])});continue
            resolved=resolve_song(song,fetcher,cancel_callback);digest=resolved['hash']
            if digest in existing:skipped.append({'hash':digest,'reason':'Already present','path':str(existing[digest])});continue
            if resolved['automapper']:skipped.append({'hash':digest,'reason':'Declared AI map excluded from human training corpus'});continue
            if progress_callback:progress_callback({'phase':'playlist','progress':i/total,'completed':i,'total':total,'downloaded':len(downloaded),'message':'Downloading and verifying '+resolved['songName']})
            payload=fetcher(resolved['downloadURL'],MAX_ARCHIVE_BYTES,cancel_callback)
            with tempfile.TemporaryDirectory(dir=folder,prefix='.playlist-') as temp:
                candidate=pathlib.Path(temp)/'map.zip';candidate.write_bytes(payload)
                actual=beatsaver_hash(candidate)
                if actual!=digest:raise ValueError('BeatSaver SHA-1 verification failed')
                data=read_archive(candidate)
                if not data['maps']:raise ValueError('Map has no usable v2/v3 Standard gameplay for training')
                check_cancel(cancel_callback)
                dest=folder/(digest+'.zip')
                if dest.parent.resolve()!=folder:raise ValueError('Destination escapes reference folder')
                candidate.replace(dest)
                record=resolved|{'path':str(dest),'zipSha256':hashlib.sha256(payload).hexdigest(),'beatSaverSha1Verified':True,'maps':len(data['maps']),'notes':sum(len(m['notes']) for m in data['maps'])}
                dest.with_suffix('.source.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
                downloaded.append(record);existing[digest]=dest
        except Exception as error:
            from training_runtime import TrainingCancelled
            if isinstance(error,TrainingCancelled):raise
            errors.append({'song':song,'error':str(error)[:500]})
    result={'title':playlist['title'],'destination':str(folder),'requested':total,'downloaded':downloaded,'skipped':skipped,'errors':errors}
    if progress_callback:progress_callback({'phase':'complete','progress':1,'completed':total,'total':total,'downloaded':len(downloaded),'message':f'Imported {len(downloaded)} verified human maps; {len(skipped)} skipped, {len(errors)} errors'})
    return result
