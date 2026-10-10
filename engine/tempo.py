"""Measured local tempo evidence and a continuous, piecewise beat clock.

Autocorrelation proposes periods; onset-grid support confirms them. Weak or
isolated estimates inherit the existing clock. A tempo timeline is evidence,
not a claim that every percussion subdivision is a new musical meter.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import signal, ndimage

VERSION='1.2'


def _estimate(times,envelope,anchor):
    if len(times)<32 or np.max(envelope)<1e-7:return None
    step=float(np.median(np.diff(times)))
    values=np.maximum(envelope,0)
    values=np.maximum(values-ndimage.median_filter(values,size=max(3,int(.15/step)|1)),0)
    if values.max()<1e-7:return None
    values=np.minimum(values,np.percentile(values,99))
    # A short periodic fill inside a mostly silent eight-second window can
    # correlate perfectly. Require activity across the window before treating
    # that correlation as sustained tempo evidence.
    edges=np.linspace(times[0],times[-1]+step,9)
    active_bins=sum(np.any(values[(times>=a)&(times<b)]>values.max()*.18) for a,b in zip(edges,edges[1:]))
    if active_bins<5:return None
    ac=signal.correlate(values,values,mode='full',method='fft')[len(values)-1:]
    # Normalize for overlap so slower candidate periods are not penalized only
    # because they have fewer samples in the analysis window.
    ac/=np.maximum(1,len(values)-np.arange(len(values)))
    low=max(2,int(60/400/step));high=min(len(ac)-2,int(60/45/step))
    peaks,_=signal.find_peaks(ac[low:high+1],prominence=max(1e-12,ac[0]*.025))
    candidates=[]
    for index in peaks+low:
        period=index*step
        support=float(ac[index]/max(ac[0],1e-12))
        if support<.16:continue
        curvature=ac[index-1]-2*ac[index]+ac[index+1]
        delta=.5*(ac[index-1]-ac[index+1])/curvature if abs(curvature)>1e-12 else 0.
        period=(index+float(np.clip(delta,-.5,.5)))*step
        bpm=60/period
        candidates.append((support,bpm))
    if not candidates:return None
    best=max(v[0] for v in candidates)
    plausible=[v for v in candidates if v[0]>=best*.88]
    # The shortest supported period resolves repeated autocorrelation peaks at
    # two/three beats. Near-octave ambiguity stays explicit in the timeline.
    support,bpm=max(plausible,key=lambda v:v[1])
    raw_bpm=bpm
    # A full-beat correlation remains present when a song adds eighth/sixteenth
    # percussion. Prefer that measured meter rather than the shortest repeated
    # attack. Choosing the shortest peak made stable 190 BPM songs jump to 746.
    anchored=[v for v in candidates if abs(v[1]/anchor-1)<.045 and v[0]>=best*.45]
    if anchored:support,bpm=min(anchored,key=lambda v:abs(v[1]/anchor-1))
    confidence=float(np.clip((support-.12)/.65,0,.98))
    return {'bpm':bpm,'confidence':confidence,'periodicity':support,
            'rawBpm':raw_bpm,'anchorSupported':bool(anchored)}


def _phase(times,envelope,bpm,start,end,preferred):
    period=60/bpm
    phases=np.linspace(0,period,max(24,int(period/.003)),endpoint=False)
    ids=np.arange(math.ceil(start/period),math.floor(end/period))
    if len(ids)<3:return preferred
    positions=ids[:,None]*period+phases
    scores=np.interp(positions,times,envelope,left=0,right=0).mean(0)
    if scores.max()<=1e-8:return preferred
    return float(phases[int(scores.argmax())])


def analyze_tempo(times,envelope,bpm,offset,duration,legacy_beats=None):
    times=np.asarray(times,float);envelope=np.asarray(envelope,float)
    fallback={'start':0.,'end':float(duration),'bpm':float(bpm),'confidence':0.,'source':'existing_global_grid','interpretation':'meter'}
    output={'version':VERSION,'timeline':[fallback],'changed':False,'beatTimes':list(legacy_beats or []),
        'estimates':[],'method':'Local onset autocorrelation, repeated-window confirmation and phase support'}
    if len(times)!=len(envelope) or len(times)<32 or duration<12:return output
    if not np.isfinite(times).all() or not np.isfinite(envelope).all():return output
    if np.any(np.diff(times)<=0):return output
    estimates=[]
    for center in np.arange(4.,duration-3.,1.):
        mask=(times>=center-4)&(times<center+4)
        estimate=_estimate(times[mask],envelope[mask],bpm)
        if estimate:estimates.append({'time':float(center),**estimate})
    output['estimates']=[{k:(bool(v) if isinstance(v,bool) else round(float(v),5)) for k,v in e.items()} for e in estimates]
    if len(estimates)<5:return output
    # Median filtering and sustained confirmation reject isolated fills.
    bpms=np.array([e['bpm'] for e in estimates]);filtered=ndimage.median_filter(bpms,size=5,mode='nearest')
    runs=[]
    for e,value in zip(estimates,filtered):
        if e['confidence']<.55:continue
        if runs and e['time']-runs[-1]['end']<=2.5 and abs(value-runs[-1]['bpm'])/runs[-1]['bpm']<.025:
            runs[-1]['values'].append(float(value));runs[-1]['end']=e['time']
            runs[-1]['bpm']=float(np.median(runs[-1]['values']))
            runs[-1]['confidences'].append(e['confidence'])
        else:runs.append({'start':e['time'],'end':e['time'],'bpm':float(value),'values':[float(value)],'confidences':[e['confidence']]})
    # Windows crossing two tempi can produce their common subharmonic. They
    # are transition uncertainty, not a brief 60 BPM section between 120 and180.
    runs=[r for r in runs if len(r['values'])>=8]
    if not runs:return output
    # A stable detector octave is not permission to replace a user's established
    # beat interpretation. Only measured changes between sustained local runs
    # activate the dynamic clock.
    if len(runs)==1:
        fallback['confidence']=round(float(np.median(runs[0]['confidences'])),4)
        return output
    # No repeated attack signal can prove that an exact doubling represents a
    # meter change rather than denser percussion. Keep these hypotheses for
    # intensity decisions, but never rewrite the beat clock on aliases alone.
    aliases=(.25,1/3,.5,1.,2.,3.,4.)
    def alias(value):return min(abs(value/bpm/a-1) for a in aliases)<.045
    output['rhythmicRateTimeline']=[{'start':r['start'],'end':r['end'],
        'bpm':round(r['bpm'],4),'confidence':round(float(np.median(r['confidences'])),4),
        'ambiguousSubdivision':alias(r['bpm']) and abs(r['bpm']/bpm-1)>.045} for r in runs]
    # A conflicting introduction cannot establish a tempo *change*: there is
    # no preceding confirmed meter. Wait for a supported anchor or a smooth
    # high-confidence acceleration starting near it.
    while runs and abs(runs[0]['bpm']/bpm-1)>.08:runs.pop(0)
    if len(runs)<2 or not any(not alias(r['bpm']) for r in runs):
        output['clockPolicy']='Existing meter retained: periodic-rate changes are ambiguous subdivisions or lack a confirmed preceding meter.'
        return output
    segments=[]
    for index,run in enumerate(runs):
        start=0. if index==0 else (runs[index-1]['end']+run['start'])/2
        end=float(duration) if index==len(runs)-1 else (run['end']+runs[index+1]['start'])/2
        local=run['bpm']
        interpretation='double_time' if local/bpm>1.8 else 'half_time' if local/bpm<.56 else 'meter'
        segments.append({'start':round(start,5),'end':round(end,5),'bpm':round(local,4),
            'confidence':round(float(np.median(run['confidences'])),4),'source':'local_periodicity',
            'interpretation':interpretation})
    # Require a perceptible tempo change, rather than rewriting minor drift
    # already handled by the existing local grid fitter.
    if max(s['bpm'] for s in segments)/min(s['bpm'] for s in segments)<1.04:return output
    beats=[];attack_floor=max(1e-8,float(np.percentile(envelope,70)))
    for segment_index,segment in enumerate(segments):
        start,end,local=segment['start'],segment['end'],segment['bpm'];period=60/local
        phase=_phase(times,envelope,local,start,end,offset)
        first=phase+math.ceil((start-phase)/period)*period
        current=np.arange(first,end+1e-7,period)
        for t in current:
            # Resolve local phase against observed attacks, rather than letting
            # a constant-period approximation drift across a gradual ramp.
            radius=period*.24;mask=(times>=t-radius)&(times<=t+radius)
            if np.any(mask):
                local_times=times[mask];local_values=envelope[mask]
                maximum=int(np.argmax(local_values))
                if local_values[maximum]>attack_floor:
                    t=float(local_times[maximum])
            # A short junction interval is an uncertain transition, not a
            # genuine explosive tempo spike. Keep one side's supported beat.
            if beats and t-beats[-1]<min(period,60/segments[max(0,segment_index-1)]['bpm'])*.45:continue
            if 0<=t<=duration:beats.append(round(float(t),6))
    output.update(timeline=segments,changed=True,beatTimes=beats)
    return output


def ensure_tempo_analysis(analysis):
    """Upgrade old analysis in memory; never modify a saved chart or recording."""
    if analysis.get('tempoAnalysis',{}).get('version')==VERSION:return analysis
    frames=analysis.get('frames',[])
    if not frames:return analysis
    times=[f['time'] for f in frames]
    envelope=[float(f.get('bass',0))+.10*float(f.get('onset',0)) for f in frames]
    result=analyze_tempo(times,envelope,float(analysis.get('bpm',120)),float(analysis.get('offset',0)),float(analysis.get('duration',0)),analysis.get('beatTimes',[]))
    value=dict(analysis);value['tempoAnalysis']={k:v for k,v in result.items() if k!='beatTimes'}
    value['tempoTimeline']=result['timeline']
    if result['changed']:value.update(beatTimes=result['beatTimes'],beatMarkers=result['beatTimes'])
    return value
