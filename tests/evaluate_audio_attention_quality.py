"""Reproduce analyzer evidence for a local audio ID, without saving user maps."""
import argparse, collections, hashlib, json, pathlib, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap
from engine.arcaudio import analyze_attention

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('audio_id');p.add_argument('--phase',default='before')
    p.add_argument('--settings',choices=['pdf','defaults'],default='pdf',help='Reproduce strict PDF settings or measure the shipping defaults separately.')
    args=p.parse_args()
    analysis=json.loads((ROOT/'data/audio'/args.audio_id/'analysis.json').read_text('utf-8'))
    folder=ROOT/'data/test-artifacts/master-audio';folder.mkdir(parents=True,exist_ok=True)
    settings=({'vocalPriority':.8,'drumPriority':.7,'attentionStrength':.6,'competition':.8,
        'maxDensity':.6,'minImportance':.75,'secondaryAllowance':.7,'smoothingSeconds':.5}
        if args.settings=='pdf' else None)
    rows=[]
    for mode in ['arcaudio','hybrid','arcaudio3']:
        started=time.perf_counter();result=analyze_attention(analysis,settings,mode)
        frames=result['frames'];at4=min(frames,key=lambda f:abs(f['time']-4))
        row={'mode':mode,'seconds':round(time.perf_counter()-started,3),'summary':result['summary'],
            'primaryFrameCounts':dict(collections.Counter(f['primarySource'] for f in frames)),
            'fourSeconds':at4,'settings':result['settings'],'settingsPreset':args.settings,
            'sourcePresenceFrames':{s:sum(f['sources'][s]>.15 for f in frames) for s in frames[0]['sources']},
            'eventTimingHash':hashlib.sha256(json.dumps([e['time'] for e in result['selectedEvents']]).encode()).hexdigest()}
        rows.append(row);(folder/f'{args.audio_id}-{args.phase}.json').write_text(json.dumps(rows,indent=2),'utf-8')
        print(json.dumps({k:v for k,v in row.items() if k not in ['fourSeconds','settings']},ensure_ascii=False),flush=True)
