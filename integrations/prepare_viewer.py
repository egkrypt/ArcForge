"""Reproduce the local viewer assets and catalogue from pinned source archives."""
import hashlib, json, re, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
arc = ROOT / 'vendor/arcviewer'
chro = ROOT / 'vendor/chromapper'
web = ROOT / 'integrations/arcviewer-web'
template = arc / 'Assets/WebGLTemplates/Fullscreen/TemplateData'
for src in template.rglob('*'):
    if src.is_file() and src.suffix != '.meta':
        dest = web / 'TemplateData' / src.relative_to(template)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): shutil.copy2(src, dest)
index = (ROOT / 'integrations/arcviewer-upstream-index.html').read_text(encoding='utf-8')
index = index.replace('gameInstance = unityInstance;', '''gameInstance = unityInstance;
        window.parent.postMessage({type: "arcforge:viewer-ready", version: "0.8.1"}, location.origin);''')
index = index.replace('alert(message);', '''document.body.insertAdjacentHTML('beforeend', '<p style="color:#fff;position:absolute;inset:40%;font-family:sans-serif">ArcViewer could not start. Switch to Studio view or reload.</p>');
        console.error(message);
        window.parent.postMessage({type: "arcforge:viewer-error", message: String(message)}, location.origin);''')
(web / 'index.html').write_text(index, encoding='utf-8')
src = (arc / 'Assets/__Scripts/Previewer/MapControl/Environments/EnvironmentManager.cs').read_text()
def array(name):
    return re.findall(r'"([\w]+Environment)"', re.search(rf'{name}\s*=\s*new string\[\]\s*\{{(.*?)\}};',src,re.S).group(1))
v2, v3, supported = array('V2Environments'), array('V3Environments'), array('supportedEnvironments')
labels = dict((b,a) for a,b in re.findall(r'new Environment\("([^"]+)",\s*"([^"]+)"\)', (chro/'Assets/__Scripts/UI/SongInfoEditUI.cs').read_text()))
entries = []
for name in v2 + v3 + ['GlassDesertEnvironment']:
    label = labels.get(name, re.sub(r'(?<=[a-z0-9])(?=[A-Z])',' ',name.removesuffix('Environment')))
    entries.append({'id':name,'name':label,'modelVersion':'V3' if name in v3 else 'V2','lightingVersion':3 if name in v3 else 2,'arcViewerSupported':name in supported or name == 'CrabRaveEnvironment','arcViewerScene':'MonstercatEnvironment' if name == 'CrabRaveEnvironment' else name if name in supported else 'DefaultEnvironment','nativePreview':'procedural','allDirections':name == 'GlassDesertEnvironment'})
(ROOT/'integrations/environments.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
manifest = {'arcviewer':{'sourceVersion':'v0.8.1-beta','webProductVersion':'0.8.1','sourceUrl':'https://github.com/AllPoland/ArcViewer/archive/refs/tags/v0.8.1-beta.zip','webAssetsOrigin':'https://allpoland.github.io/ArcViewer/','license':'GPL-3.0','unity':'6000.0.67f1'},'chromapper':{'sourceVersion':'0.13.892','sourceUrl':'https://github.com/Caeden117/ChroMapper/archive/refs/tags/0.13.892.zip','license':'GPL-2.0','unity':'2021.3.45f1'},'downloadHashes':{},'viewerAssetHashes':{}}
for p in (ROOT/'vendor/downloads').glob('*.zip'): manifest['downloadHashes'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in (web/'Build').glob('*'): manifest['viewerAssetHashes'][p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
(ROOT/'integrations/source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(f'Prepared {len(entries)} environment choices, {len(supported)} distinct ArcViewer scenes and Crab Rave alias.')
