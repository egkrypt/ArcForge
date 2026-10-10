"""Conservative jump recommendations and optional native V4.1 NJS events.

NJS easing runs from the previous event to the current event. Values are
relative to Info.dat's base NJS; this is not audio speed or a tempo change.
Reference: https://bsmg.wiki/mapping/map-format/beatmap.html#njs-events
"""
from __future__ import annotations
import collections, copy, math, statistics
from modding import BeatClock


def apply_dynamic_njs(chart, analysis, config, report, base_njs, base_jd):
    """Return (chart copy, truthful jump report), preserving all authored data.

    Pass the original creation settings, before calculated base NJS is inserted.
    V3, manual NJS and ranking targets receive advice only. Native animation is
    opt-in, section-smoothed, capped at +/-2 m/s and at 1.5 m/s peak slew.
    """
    chart=copy.deepcopy(chart);base_njs=float(base_njs);base_jd=float(base_jd)
    if not math.isfinite(base_njs) or not math.isfinite(base_jd) or base_njs<=0 or base_jd<=0:
        raise ValueError('Jump planning requires positive finite base NJS and jump distance.')
    clock=BeatClock(float(config.get('bpm') or analysis['bpm']),chart.get('bpmEvents',[]))
    measured={row['id']:row for row in report.get('critic',{}).get('sections',[])}
    sections=report.get('mapPlan',{}).get('sections',[])
    active=[float(measured.get(s['id'],{}).get('actualNps',s.get('targetNps',0))) for s in sections
        if float(measured.get(s['id'],{}).get('actualNps',s.get('targetNps',0)))>0]
    typical=max(1.,statistics.median(active) if active else float(config.get('nps',5)))
    typical_bpm=max(1.,statistics.median([float(s.get('localBpm') or analysis['bpm']) for s in sections]) if sections else float(analysis['bpm']))
    section_notes=collections.defaultdict(list)
    for note in chart.get('colorNotes',[]):
        section_notes[note.get('customData',{}).get('arcforgePlanSection')].append(note)
    recommendations=[]
    for section in sections:
        start,end=float(section['start']),float(section['end'])
        if end<=start:continue
        actual=float(measured.get(section['id'],{}).get('actualNps',section.get('targetNps',0)))
        local_bpm=float(section.get('localBpm') or typical_bpm)
        notes=section_notes[section['id']]
        angular=sum(2<=int(n.get('d',1))<8 for n in notes)/max(1,len(notes))
        goals=section.get('goals',{})
        tech=max(0.,min(1.,float(goals.get('tech',55))/100))
        movement=max(0.,min(1.,float(goals.get('movement',55))/100))
        complexity=.5*tech+.5*angular
        # Density and tempo change visual separation modestly; angular/travel
        # demand increases the reading margin instead of multiplying difficulty.
        delta=.9*math.log2((max(0.,actual)+1)/(typical+1))+.35*math.log2(max(1.,local_bpm)/typical_bpm)
        delta-=.4*complexity+.25*movement
        delta=max(-2.,min(2.,delta));njs=max(5.,base_njs+delta)
        delta=round(njs-base_njs,3)
        recommendations.append({'id':section['id'],'start':start,'end':end,'actualNps':actual,
            'localBpm':local_bpm,'angularFraction':round(angular,4),'recommendedNjs':round(njs,3),
            'relativeNjs':delta,'nativeJumpDistance':round(base_jd*max(1.,njs/base_njs),3),
            'nativeReactionTimeMs':round(base_jd/(2*min(base_njs,njs))*1000,2)})
    result={'version':'1.0','baseNjs':base_njs,'baseJumpDistance':base_jd,
        'baseReactionTimeMs':round(base_jd/(2*base_njs)*1000,2),'sections':recommendations,
        'applied':False,'eventCount':len(chart.get('njsEvents',[])),
        'policy':'Measured section density, tempo and angular demand; recommendations are bounded heuristics, not a comfort guarantee.',
        'nativeSemantics':'Below base NJS, jump distance stays constant; above base NJS, reaction time stays constant.',
        'previewLimitation':'ArcViewer V3 conversion cannot preview native V4.1 NJS events.'}
    reason=None
    if chart.get('njsEvents'):reason='Authored native NJS events preserved.'
    elif not config.get('dynamicNjs',False):reason='Dynamic NJS is disabled; section values are advisory.'
    elif not str(config.get('beatmapVersion','3')).startswith('4'):reason='Native V4.1 export is required; V3 base NJS remains static.'
    elif config.get('njs') or config.get('njsAuto',True) is False:reason='Manual NJS override preserved.'
    elif config.get('compatibility')=='Ranked-compatible':reason='Ranking target uses static base NJS; native events are not emitted.'
    elif config.get('mode') in ['90Degree','360Degree','90Degrees','360Degrees'] or chart.get('rotationEvents'):
        reason='Native NJS generation is not combined with rotating characteristics.'
    elif not recommendations:reason='No measured section plan is available.'
    if reason:
        result['reason']=reason
        return chart,result
    # Four-second phrase smoothing avoids reacting to every small bar change.
    buckets=[];pending=[]
    for row in recommendations:
        pending.append(row)
        if row['end']-pending[0]['start']>=4.:
            buckets.append(pending);pending=[]
    if pending:buckets.append(pending)
    events=[{'b':0.,'d':0.,'p':0,'e':0}];previous_delta=0.;previous_end=0.
    for bucket in buckets:
        start=max(0.,float(bucket[0]['start']),previous_end)
        end=float(bucket[-1]['end']);duration=sum(r['end']-r['start'] for r in bucket)
        delta=sum(r['relativeNjs']*(r['end']-r['start']) for r in bucket)/max(.001,duration)
        delta=round(max(-2.,min(2.,delta)),3)
        if abs(delta-previous_delta)<.2:continue
        # InOutSine peaks at pi/2 times average slew; bound its peak as well.
        ramp=max(1.5,math.pi/2*abs(delta-previous_delta)/1.5)
        if end-start<ramp:continue
        if start>previous_end+.00001:
            events.append({'b':clock.beat(start),'d':previous_delta,'p':1,'e':0})
        finish=start+ramp
        events.append({'b':clock.beat(finish),'d':delta,'p':0,'e':6})
        previous_delta=delta;previous_end=finish
    if len(events)>1:
        chart['njsEvents']=events
        result.update(applied=True,eventCount=len(events),reason='Native V4.1 section-smoothed NJS events emitted.',
            appliedNjsRange=[round(base_njs+min(e['d'] for e in events),3),round(base_njs+max(e['d'] for e in events),3)],
            minimumRampSeconds=1.5,maximumPeakSlewMetersPerSecond=1.5)
    else:result['reason']='Section changes were too small or short to warrant native NJS events.'
    return chart,result
