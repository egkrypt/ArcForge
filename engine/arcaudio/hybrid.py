"""Fuse observed legacy attacks with spectral foreground evidence, before selection.

The legacy clock contributes support, not newly invented grid events. All added
candidates still pass the same source competition and finite attention budget.
"""
from __future__ import annotations
import bisect, copy
import numpy as np


def fuse_legacy(analysis, evidence):
    result=copy.deepcopy(evidence)
    frames=result['frames'];events=result['events']
    if not frames:return result,0
    times=[f['time'] for f in frames]
    existing=sorted(e['time'] for e in events)
    grid=np.asarray(analysis.get('beatTimes',[]),float)
    if len(grid)>1:
        quarters=np.sort(np.concatenate([grid[:-1]+fraction*np.diff(grid) for fraction in [0,.25,.5,.75]]))
    else:quarters=grid
    old=analysis.get('frames',[])
    old_times=[f['time'] for f in old]
    added=0
    for second in analysis.get('onsets',[]):
        second=float(second);index=bisect.bisect_left(existing,second)
        if any(abs(t-second)<.020 for t in existing[max(0,index-1):index+1]):continue
        frame=frames[min(len(frames)-1,max(0,bisect.bisect_right(times,second)-1))]
        onset=float(np.interp(second,old_times,[f.get('onset',0) for f in old])) if old else .5
        if onset<.18 or frame.get('energy',0)<.003:continue
        sources=frame['sources'];confidence=frame['sourceConfidence']
        eligible=['drums','bass','melody','vocals','spoken']
        source=max(eligible,key=lambda s:sources.get(s,0)*(.35+.65*confidence.get(s,0)))
        if sources.get(source,0)<.08:continue
        if source=='drums':
            source=max(['kick','snare','percussion'],key=lambda s:sources.get(s,0))
        distance=float(np.min(np.abs(quarters-second))) if len(quarters) else None
        strength=min(1.,.65*onset+.35*sources.get(source,0))
        if distance is not None and distance<.025:strength=min(1.,strength*1.08)
        events.append({'time':second,'source':source,'roles':['tonalChange' if source in ['melody','vocals','bass'] else 'midTransient'],
            'strength':strength,'confidence':min(.55,confidence.get(source,.35)),
            'evidence':'legacy_attack_with_spectral_foreground','timingEvidence':'observed_onset',
            'gridSupport':round(distance,5) if distance is not None else None,
            'componentEvidence':frame.get('componentEvidence',{})})
        added+=1
    result['events']=sorted(events,key=lambda e:e['time'])
    result['method']+=' + observed legacy transient/beat-phase fusion'
    return result,added
