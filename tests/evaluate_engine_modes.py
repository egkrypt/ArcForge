"""Generate isolated native maps to measure actual engine-selection effects.

No library entries or user charts are saved. Timing distances are evidence
alignment measurements, not calibrated musicality or headset-playability scores.
"""
import argparse, collections, copy, hashlib, json, pathlib, sys, time, uuid, zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import bootstrap
import numpy as np
from engine.arcaudio import analyze_attention
from engine.generate import generate_map
from engine.music import ensure_music_analysis
from engine.tempo import ensure_tempo_analysis
from engine.trainer import model_status
from gameplay import add_events
from modding import apply_mods, BeatClock
from projects import _export_project
from quality_benchmark import metrics

def checksum(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def timing_overlap(first,second,tolerance=.020):
    """Symmetric overlap of unique hit times, allowing a measured 20 ms window."""
    a=sorted(set(first));b=sorted(set(second));i=j=matched=0
    while i<len(a) and j<len(b):
        if abs(a[i]-b[j])<=tolerance:matched+=1;i+=1;j+=1
        elif a[i]<b[j]:i+=1
        else:j+=1
    return round(2*matched/max(1,len(a)+len(b)),4)


def activity_density(chart,analysis,report):
    """Compare the same measured music sections across engines, including rests."""
    clock=BeatClock(analysis['bpm'],chart.get('bpmEvents',[]))
    times=np.asarray([clock.time(n['b']) for n in chart['colorNotes']])
    sections=analysis.get('musicAnalysis',{}).get('sections',[]);groups={}
    for name,predicate in [('sparse',lambda s:s.get('activity',0)<.25),
            ('highActivity',lambda s:s.get('activity',0)>=.75),
            ('dropOrClimax',lambda s:s.get('label','').lower() in ['drop','climax'])]:
        selected=[s for s in sections if predicate(s)]
        seconds=sum(max(0,s['end']-s['start']) for s in selected)
        hits=sum(int(np.sum((times>=s['start'])&(times<s['end']))) for s in selected)
        groups[name]={'seconds':round(seconds,3),'notes':hits,'actualNps':round(hits/seconds,3) if seconds else None,
            'sectionCount':len(selected)}
    def peak(window):
        return round(float(np.max(np.arange(len(times))+1-np.searchsorted(times,times-window,side='right')))/window,3) if len(times) else 0
    groups['peakOneSecondNps']=peak(1)
    groups['peakFourSecondNps']=peak(4)
    actual_sections=report.get('critic',{}).get('sections',[])
    active=[s for s in actual_sections if s.get('targetNps',0)>=5]
    target_hits=sum(s['targetNps']*max(0,s['end']-s['start']) for s in active)
    groups['activePlannedDensity']={'targetHits':round(target_hits,3),
        'actualHits':sum(s['noteCount'] for s in active),'sectionCount':len(active),
        'actualToTarget':round(sum(s['noteCount'] for s in active)/target_hits,4) if target_hits else None,
        'sectionsBelow75PercentTarget':sum(s['actualNps']<s['targetNps']*.75 for s in active)}
    return groups


def evaluate(audio_ids,phase='current',strict=False,modes=None):
    folder=ROOT/'data/test-artifacts/engine-comparison'/phase;folder.mkdir(parents=True,exist_ok=True)
    originals={str(p):checksum(p) for p in (ROOT/'data/projects').glob('*/project.json')}
    model=ROOT/'models/style';model_hash=checksum(model/'model.npz');rows=[];overlaps=[]
    model_fingerprints={name:checksum(model/name) for name in ['model.npz','composition.json','style-corpus.json','report.json'] if (model/name).is_file()}
    source_fingerprints={name:checksum(ROOT/name) for name in ['engine/arcaudio/sources.py','engine/arcaudio/attention.py',
        'engine/recognition/__init__.py','engine/attention_candidates.py','engine/generate.py',
        'engine/tempo.py','engine/lighting_plan.py','engine/arcaudio/hybrid.py',
        'engine/arcaudio/events.py','engine/arcaudio/selector.py','engine/scheduling.py',
        'engine/planning.py','engine/music.py','modding.py'] if (ROOT/name).is_file()}
    learned_lighting=model_status(model).get('lighting',{})
    for identity in audio_ids:
        record=json.loads((ROOT/'data/audio'/identity/'analysis.json').read_text('utf-8'))
        record=ensure_tempo_analysis(ensure_music_analysis(record))
        modes=modes or ['off','legacy','arcaudio','hybrid','arcaudio3']
        output_times={}
        for mode in modes:
            analysis=copy.deepcopy(record);started=time.perf_counter()
            knobs=({'vocalPriority':.8,'drumPriority':.7,'attentionStrength':.6,'competition':.8,
                'maxDensity':.6,'minImportance':.75,'secondaryAllowance':.7,'smoothingSeconds':.5} if strict else None)
            settings={'bpm':analysis['bpm'],'nps':12.,'rank':9,'mode':'Standard','backend':'local',
                'seed':32026,'style':'Modern','arcAudioMode':mode,'dynamicDensity':True,
                'learnWholeLevel':True,'learnLighting':True,'mods':['Chroma'],
                '_compositionModelDir':str(model),'_learnedLighting':learned_lighting,
                'events':{'lighting':True,'arcs':True},'arcs':True,
                'philosophy':{'musicality':85,'tech':55,'movement':55,'variety':80,'contrast':90,'stamina':65}}
            attention=analyze_attention(analysis,knobs,mode)
            if attention['enabled']:analysis['_arcAudioAttention']=attention
            generated=generate_map(analysis,copy.deepcopy(settings),model)
            lighting_settings=dict(settings,_arcForgeMapPlan=generated['report'].get('mapPlan',{}))
            chart,mod_info=apply_mods(add_events(generated['beatmap'],analysis,settings),analysis,lighting_settings)
            destination=folder/f'{identity}-{mode}';destination.mkdir(exist_ok=True)
            draft={'id':uuid.uuid4().hex,'name':'ArcForge engine evaluation','created':0,'updated':0,'revision':1,
                'bpm':analysis['bpm'],'environment':'DefaultEnvironment',
                'audio':{**{k:analysis[k] for k in ['bpm','duration']},'id':identity},
                'config':{'arcAudioMode':mode},'charts':[{'id':uuid.uuid4().hex,'name':'Expert+',
                    'rank':9,'mode':'Standard','njs':20,'jumpDistance':24,'data':chart,'modInfo':mod_info}]}
            archive=_export_project(draft['id'],draft,destination,destination)
            with zipfile.ZipFile(archive) as zipped:
                for name in zipped.namelist():
                    if name.lower().endswith('.dat'):(destination/name).write_bytes(zipped.read(name))
            result_metrics=metrics(chart,analysis,settings)
            clock=BeatClock(analysis['bpm'],chart.get('bpmEvents',[]))
            output_times[mode]=[clock.time(n['b']) for n in chart['colorNotes']]
            row={'audioId':identity,'name':record.get('name'),'mode':mode,'strictPdfSettings':strict,
                'modelHash':model_hash,'modelFingerprints':model_fingerprints,'seed':settings['seed'],'requestedNps':settings['nps'],
                'sourceFingerprints':source_fingerprints,
                'seconds':round(time.perf_counter()-started,3),**result_metrics,
                'attentionSummary':attention['summary'],'analysisSeconds':attention['performance']['analysisSeconds'],
                'sourceEventCounts':dict(collections.Counter(e.get('source') for e in attention['events'])),
                'primaryFrameCounts':dict(collections.Counter(f.get('primarySource') for f in attention['frames'])),
                'secondaryEnabledFrames':sum(bool(f.get('allowSecondary')) for f in attention['frames']),
                'sourcePresentFrames':{source:sum(f['sources'].get(source,0)>.15 for f in attention['frames'])
                    for source in ['drums','bass','piano','chords','melody','vocals','spoken','other']},
                'tempoChanged':analysis.get('tempoAnalysis',{}).get('changed'),
                'timingHash':hashlib.sha256(json.dumps([n['b'] for n in chart['colorNotes']]).encode()).hexdigest(),
                'chartHash':hashlib.sha256(json.dumps(chart,sort_keys=True).encode()).hexdigest(),
                'critic':generated['report'].get('critic'),'engineSelection':generated['report'].get('engineSelection'),
                'densityByMusicalActivity':activity_density(chart,analysis,generated['report']),
                'lightEvents':len(chart.get('basicBeatmapEvents',[])),
                'lightingPlan':chart.get('customData',{}).get('arcforge',{}).get('lightingPlan'),
                'sourceTrace':dict(collections.Counter(n.get('customData',{}).get('arcforgeSource') for n in chart['colorNotes']))}
            rows.append(row)
            (destination/'generation-report.json').write_text(json.dumps(generated['report'],indent=2),'utf-8')
            (folder/'results.json').write_text(json.dumps(rows,indent=2),'utf-8')
            print(json.dumps({k:v for k,v in row.items() if k in ['name','mode','notes','averageNps','colorCounts','maxColorRun','medianAttackDistanceMs','p95AttackDistanceMs','lightEvents','timingHash']}),flush=True)
        for first,second in [('off','legacy'),('arcaudio','hybrid'),('arcaudio','arcaudio3')]:
            if first not in modes or second not in modes:continue
            a=next(r for r in rows if r['audioId']==identity and r['mode']==first)
            b=next(r for r in rows if r['audioId']==identity and r['mode']==second)
            if first=='off':assert a['chartHash']==b['chartHash'],'OFF and LEGACY must preserve the same existing mapper'
            else:assert a['timingHash']!=b['timingHash'],'Selected attention engines must produce a measured output difference'
        for first,second in [('arcaudio','hybrid'),('arcaudio','arcaudio3'),('hybrid','arcaudio3')]:
            if first not in modes or second not in modes:continue
            overlaps.append({'audioId':identity,'name':record.get('name'),'modes':[first,second],
                'timingDice20ms':timing_overlap(output_times[first],output_times[second])})
        (folder/'timing-overlap.json').write_text(json.dumps(overlaps,indent=2),'utf-8')
    assert originals=={str(p):checksum(p) for p in (ROOT/'data/projects').glob('*/project.json')},'User projects changed during read-only evaluation'
    assert all(checksum(model/name)==digest for name,digest in model_fingerprints.items()),'Learned model changed during engine comparison; rerun using one fixed checkpoint'
    assert all(checksum(ROOT/name)==digest for name,digest in source_fingerprints.items()),'Source changed during engine comparison; rerun with one implementation'
    (folder/'preservation.json').write_text(json.dumps({'projectsUnchanged':True,'projectCount':len(originals),
        'projectFingerprints':originals,'modelHash':model_hash,'modelFingerprints':model_fingerprints,
        'sourceFingerprints':source_fingerprints},indent=2),'utf-8')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audio-id',action='append');parser.add_argument('--phase',default='current');parser.add_argument('--strict-pdf',action='store_true')
    parser.add_argument('--mode',action='append',choices=['off','legacy','arcaudio','hybrid','arcaudio3'])
    args=parser.parse_args()
    ids=args.audio_id or [p.parent.name for p in sorted((ROOT/'data/audio').glob('*/analysis.json'))]
    if not ids:parser.error('No imported audio records exist; import a song or pass --audio-id.')
    evaluate(ids,args.phase,args.strict_pdf,args.mode)
