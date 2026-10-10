"""Statistical style conditioning learned from uploaded human charts.

Named presets specify goals, not block templates or verified genre/style labels.
Nearest measured chart archetypes supply placement distributions and motion scale.
"""
import functools, hashlib, json, pathlib
import numpy as np
from .dataset import read_archive
from .features import token
from modding import BeatClock
ROOT=pathlib.Path(__file__).resolve().parents[1]
# Speed, angular complexity, travel and chord goals in normalized corpus space.
GOALS={'OST':(.20,.12,.25,.18),'Modern':(.6,.55,.55,.45),'Tech':(.55,.90,.65,.40),
    'Technical':(.55,.90,.65,.40),'Speed':(.95,.35,.25,.25),'Accuracy':(.25,.15,.20,.20),
    'Tech/Acc':(.45,.75,.40,.30),'Stream':(.95,.20,.20,.10),'Jump':(.80,.45,.90,.45),
    'Challenge':(1.,.90,.85,.65),'Dance':(.40,.35,.95,.40),'Acc-focused':(.30,.20,.25,.20),
    'Flow':(.55,.35,.50,.20),'Saber control':(.45,.80,.40,.30),'Powerhouse':(.85,.65,.70,.65),
    'High-intensity':(.95,.85,.65,.55),'Tech-heavy':(.65,1.,.70,.40),'Light/easy':(.12,.10,.20,.10),
    'Experimental':(.65,.90,.80,.60),'Expressive':(.45,.60,.80,.35)}

def _signature(folder):
    return tuple((p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in sorted(folder.glob('*.zip')))

@functools.lru_cache(maxsize=2)
def _corpus(folder,signature):
    cache=ROOT/'models/style/style-corpus.json';digest=hashlib.sha256(repr(signature).encode()).hexdigest()
    if cache.exists():
        saved=json.loads(cache.read_text('utf-8'))
        if saved.get('digest')==digest:return saved['charts']
    charts=[]
    for name,_,_ in signature:
        try:archive=read_archive(pathlib.Path(folder)/name)
        except Exception:continue
        for chart in archive['maps']:
            events=[e for e in chart.get('bpmEvents',[]) if isinstance(e,dict) and isinstance(e.get('b'),(int,float)) and isinstance(e.get('m'),(int,float)) and e['b']>=0 and 20<=e['m']<=1000]
            clock=BeatClock(chart['bpm'],events);notes=chart['notes'];duration=max(.1,clock.time(notes[-1]['b'])-clock.time(notes[0]['b']))
            travel=[];previous={};chords={}
            angles=sum(n['d'] in [2,3,4,5,6,7] for n in notes)/len(notes)
            for n in notes:
                if n['c'] in previous:
                    p=previous[n['c']];travel.append(float(np.hypot(n['x']-p['x'],n['y']-p['y'])))
                previous[n['c']]=n;chords[n['b']]=chords.get(n['b'],0)+1
            prior=np.ones(108,np.float64)
            for n in notes:prior[token(n)]+=1
            prior/=prior.sum()
            charts.append({'source':name,'song':archive['songName'],'difficulty':chart['difficulty'],
                'features':[len(notes)/duration,angles,float(np.mean(travel)) if travel else 0,sum(v==2 for v in chords.values())/max(1,len(chords))],
                'prior':prior.tolist()})
    cache.parent.mkdir(parents=True,exist_ok=True)
    cache.write_text(json.dumps({'digest':digest,'charts':charts},separators=(',',':')),'utf-8')
    return charts

def style_profile(style,config):
    folder=ROOT/'data/dataset';charts=_corpus(str(folder),_signature(folder));goal=np.array(GOALS.get(style,GOALS['Modern']))
    rank=int(config.get('rank',9));scale=min(1.,.35+rank*.075);goal*=scale
    if not charts:return {'prior':np.ones(108)/108,'goal':goal,'matches':[],'method':'No corpus style archetypes available'}
    matrix=np.array([c['features'] for c in charts]);lo=np.quantile(matrix,.05,axis=0);hi=np.quantile(matrix,.95,axis=0)
    normalized=np.clip((matrix-lo)/np.maximum(hi-lo,.01),0,1)
    distance=np.linalg.norm(normalized-goal,axis=1);ids=np.argsort(distance)[:min(24,len(charts))]
    weights=np.exp(-distance[ids]*5);weights/=weights.sum()
    prior=sum(w*np.array(charts[i]['prior']) for w,i in zip(weights,ids))
    return {'prior':prior,'goal':goal,'matches':[{'song':charts[i]['song'],'difficulty':charts[i]['difficulty'],'distance':round(float(distance[i]),3)} for i in ids[:5]],
        'method':'Unsupervised human-chart archetypes: density, angles, movement and doubles. Preset names are goals; no verified style labels supplied.'}
