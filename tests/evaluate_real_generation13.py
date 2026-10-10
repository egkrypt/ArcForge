"""Read-only real-song evaluation; private artifacts never enter source releases.

Run from ArcForge: python -X utf8 tests/evaluate_real_generation13.py --limit 2
This creates isolated exports under data/test-artifacts, never library entries.
Attack distance measures timing evidence, not perceptual quality or map ranking.
"""
import argparse, bisect, copy, json, pathlib, statistics, sys, time, uuid, zipfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import bootstrap
from attention_api import cached_attention, attention_config
from engine.generate import generate_map
from engine.trainer import model_status
from gameplay import add_events
from modding import BeatClock, apply_mods
from projects import DATA, _export_project


def evaluate(limit=2):
    output=DATA/'test-artifacts'/'generation-final13';output.mkdir(parents=True,exist_ok=True)
    results=[];model=ROOT/'models/style'
    for index,record in enumerate(sorted((DATA/'projects').glob('*/project.json'))[:limit]):
        project=json.loads(record.read_text('utf-8'))
        analysis=json.loads((DATA/'audio'/project['audio']['id']/'analysis.json').read_text('utf-8'))
        for mode in ['legacy','arcaudio3']:
            started=time.perf_counter();source=copy.deepcopy(analysis)
            settings={'bpm':analysis['bpm'],'nps':10,'rank':9,'mode':'Standard','backend':'local',
                'seed':313,'style':'Modern','arcAudioMode':mode,'dynamicDensity':True,
                'learnWholeLevel':True,'learnLighting':True,'mods':['Chroma','Noodle Extensions'],
                'events':{'lighting':True,'arcs':True,'chains':False},'arcs':True,
                '_compositionModelDir':str(model),'_learnedLighting':model_status(model).get('lighting',{})}
            if mode=='arcaudio3':
                _,knobs=attention_config(settings)
                source['_arcAudioAttention']=cached_attention(source,mode,knobs)
            generated=generate_map(source,settings,model)
            if not generated['report']['passed']:raise AssertionError(generated['report']['issues'])
            chart,mod_info=apply_mods(add_events(generated['beatmap'],source,settings),source,settings)
            clock=BeatClock(source['bpm'],chart.get('bpmEvents',[]))
            attacks=sorted(a['time'] for a in source.get('musicAnalysis',{}).get('accents',[]))
            errors=[]
            for note in chart['colorNotes']:
                second=clock.time(note['b']);position=bisect.bisect_left(attacks,second)
                nearby=[abs(second-attacks[i]) for i in [position-1,position] if 0<=i<len(attacks)]
                if nearby:errors.append(min(nearby)*1000)
            destination=output/f'{index}-{mode}';destination.mkdir(exist_ok=True)
            draft={'id':uuid.uuid4().hex,'name':'ArcForge release evaluation','created':0,'updated':0,'revision':1,
                'bpm':source['bpm'],'environment':'DefaultEnvironment','audio':project['audio'],
                'config':{'arcAudioMode':mode},'charts':[{'id':uuid.uuid4().hex,'name':'Expert+',
                'rank':9,'mode':'Standard','njs':20,'jumpDistance':24,'data':chart,'modInfo':mod_info}]}
            archive=_export_project(draft['id'],draft,destination,destination)
            # Native files only; do not make redundant extracted copies of audio.
            with zipfile.ZipFile(archive) as zipped:
                for name in zipped.namelist():
                    if name.lower().endswith('.dat'):(destination/name).write_bytes(zipped.read(name))
            errors.sort();report=generated['report']
            row={'projectId':project['id'],'mode':mode,'requestedActiveNps':10,
                'notes':len(chart['colorNotes']),'averageNps':round(len(chart['colorNotes'])/source['duration'],3),
                'passed':report['passed'],'seconds':round(time.perf_counter()-started,3),
                'medianAttackDistanceMs':round(statistics.median(errors),3) if errors else None,
                'p95AttackDistanceMs':round(errors[min(len(errors)-1,int(len(errors)*.95))],3) if errors else None,
                'consecutiveIdenticalBars':report.get('consecutiveIdenticalBars'),
                'densityCurve':report.get('attention',{}).get('densityCurve',[]),
                'lights':len(chart.get('basicBeatmapEvents',[])),
                'ringRotations':sum(e['et']==8 for e in chart.get('basicBeatmapEvents',[])),
                'animatedNotes':sum(bool(n.get('customData',{}).get('animation')) for n in chart['colorNotes']),
                'exportFolder':str(destination),'timingMetric':'Distance to measured attacks; not a listening or VR-playability score.'}
            results.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ['densityCurve','exportFolder']}),flush=True)
            (output/'results.json').write_text(json.dumps(results,indent=2),'utf-8')
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=2)
    args=parser.parse_args();evaluate(max(1,min(4,args.limit)))
