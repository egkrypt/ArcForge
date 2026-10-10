"""Reproducible, read-only corpus / real-generation measurements.

Outputs are private local evaluation artifacts, never Library projects. This is
an engineering comparison; it does not assign fabricated musicality scores.
"""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
import bootstrap
import numpy as np
from engine.generate import generate_map
from engine.diagnostics import analyze_playability
from engine.variety import analyze_variety
from engine.music import ensure_music_analysis
from modding import BeatClock


def metrics(chart, analysis, config):
    clock = BeatClock(config.get('bpm', analysis['bpm']), chart.get('bpmEvents', []))
    notes = sorted(chart['colorNotes'], key=lambda n: (n['b'], n['c']))
    times = np.array([clock.time(n['b']) for n in notes]); colors = [n['c'] for n in notes]
    runs = [len(list(group)) for _, group in __import__('itertools').groupby(colors)]
    previous = {}; reach = []; angular = []; crossing = 0
    for note in notes:
        hand = note['c']; crossing += int(note['x'] > 1 if hand == 0 else note['x'] < 2)
        if hand in previous:
            before = previous[hand]; reach.append(float(np.hypot(note['x'] - before['x'], note['y'] - before['y'])))
            angular.append(abs(((90,270,180,0,135,45,225,315,0)[note['d']] - (90,270,180,0,135,45,225,315,0)[before['d']] + 180) % 360 - 180))
        previous[hand] = note
    attacks = np.array([p['time'] for p in analysis.get('musicAnalysis', {}).get('accents', [])])
    errors = [float(np.min(abs(attacks - t))) * 1000 for t in times] if len(attacks) else []
    simultaneous=collections.defaultdict(set)
    for note in notes:simultaneous[note['b']].add(note['c'])
    doubles=sum(len(hands)==2 for hands in simultaneous.values())
    traced_errors=[abs(t-float(n['customData']['arcforgeObservedTime']))*1000 for n,t in zip(notes,times)
        if n.get('customData',{}).get('arcforgeObservedTime') is not None]
    report = analyze_playability(chart, analysis, config)
    return {'notes': len(notes), 'averageNps': round(len(notes) / analysis['duration'], 4),
        'colorCounts': dict(collections.Counter(colors)), 'maxColorRun': max(runs, default=0),
        'meanGridTravel': round(float(np.mean(reach)), 4) if reach else 0,
        'nonVerticalFraction': round(sum(n['d'] not in (0, 1, 8) for n in notes) / max(1,len(notes)), 4),
        'crossLaneFraction': round(crossing/max(1,len(notes)),4),
        'doubleAccents':doubles,'accentNoteFraction':round(2*doubles/max(1,len(notes)),4),
        'tracedObservedHits':len(traced_errors),
        'maxTracedObservedDistanceMs':round(max(traced_errors),3) if traced_errors else None,
        'medianAttackDistanceMs': round(float(np.median(errors)),3) if errors else None,
        'p95AttackDistanceMs': round(float(np.percentile(errors,95)),3) if errors else None,
        'playability': {k:report[k] for k in ('passed','errors','warnings','counts','minimumHandIntervalMs','p95RecoveryMetersPerSecond')},
        'variety': {k:v for k,v in analyze_variety(chart, analysis, config).items() if k not in ('repeatedBars','dominantMotionNgrams','motifVariations')}}


def run(phase, names=None):
    folder=ROOT/'data/test-artifacts/master-quality'; folder.mkdir(parents=True,exist_ok=True)
    names=names or ['Fresh!', 'Fallen Winner 2', 'Katy Perry - The One That Got Away (Lyrics)']
    files=sorted((ROOT/'data/audio').glob('*/analysis.json'))
    analyses={}
    for file in files:
        value=json.loads(file.read_text('utf-8'))
        if value.get('name') in names and value.get('musicAnalysis'):
            analyses[value['name']]=value
    # An import created before music analysis can still be analyzed without
    # modifying its persisted record.
    for file in files:
        value=json.loads(file.read_text('utf-8'))
        if value.get('name') in names and value['name'] not in analyses:
            analyses[value['name']]=value
    rows=[]
    for name in names:
        if name not in analyses: continue
        analysis=ensure_music_analysis(copy.deepcopy(analyses[name])); config={'nps':12,'rank':9,'backend':'local','style':'Modern',
            'mode':'Standard','arcAudioMode':'legacy','seed':32026,'dynamicDensity':True,
            'philosophy':{'musicality':85,'tech':55,'movement':55,'variety':80,'contrast':90,'stamina':65}}
        began=time.perf_counter(); result=generate_map(analysis,config,ROOT/'models/style')
        filename=hashlib.sha256(name.encode()).hexdigest()[:12]+'.json'
        (folder/(phase+'-'+filename)).write_text(json.dumps({'beatmap':result['beatmap'],'report':result['report']},indent=2),'utf-8')
        row={'song':name,'bpm':analysis['bpm'],'requestedNps':12,'seconds':round(time.perf_counter()-began,3),
            **metrics(result['beatmap'],analysis,config),'sections':result['report'].get('musicalSections',[]),
            'critic':result['report'].get('critic'), 'artifact':phase+'-'+filename}
        rows.append(row); print(json.dumps({k:v for k,v in row.items() if k not in ('sections','critic')}),flush=True)
    human=ROOT/'models/style/style-corpus.json'
    if human.exists():
        corpus=json.loads(human.read_text('utf-8'))['charts']; matrix=np.asarray([c['features'] for c in corpus])
        reference={'charts':len(corpus),'features':['averageNps','nonVerticalFraction','meanSameHandGridTravel','doubleFraction'],
            'quantiles':{str(p):np.quantile(matrix,p,axis=0).round(4).tolist() for p in (.05,.25,.5,.75,.95)},
            'interpretation':'Unlabeled whole-chart distributions, not a quality or ranking target.'}
    else: reference={}
    source_files=['engine/generate.py','engine/scheduling.py','engine/planning.py','engine/attention_candidates.py']
    source_hash=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in source_files)).hexdigest()
    (folder/(phase+'-summary.json')).write_text(json.dumps({'phase':phase,'rows':rows,'humanCorpus':reference,
        'modelSha256':hashlib.sha256((ROOT/'models/style/model.npz').read_bytes()).hexdigest(),
        'sourceFingerprint':source_hash},indent=2),'utf-8')
    return rows


def style_report():
    from test_mapping_quality import rhythmic_fixture
    analysis=rhythmic_fixture(120,24); rows=[]
    base={'nps':4,'rank':9,'backend':'local','seed':324,'arcAudioMode':'legacy'}
    configs=[dict(base,style=s) for s in ('Accuracy','Speed','Tech','Dance','Flow','Challenge')]
    configs += [dict(base,style='Dance',philosophy={'movement':m,'tech':55,'variety':80,'musicality':85,'contrast':65,'stamina':65}) for m in (0,100)]
    for config in configs:
        output=generate_map(analysis,config,ROOT/'models/style')
        rows.append({'style':config['style'],'movementGoal':config.get('philosophy',{}).get('movement'),
            **metrics(output['beatmap'],analysis,config)})
    path=ROOT/'data/test-artifacts/master-quality/style-profile-effects.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(rows,indent=2),'utf-8')
    print(json.dumps([{k:r[k] for k in ('style','movementGoal','notes','meanGridTravel','nonVerticalFraction','crossLaneFraction')} for r in rows],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase');parser.add_argument('--song',action='append')
    args=parser.parse_args()
    if args.phase=='styles':style_report()
    else:run(args.phase,args.song)
