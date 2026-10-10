"""Budgeted musical attention feeds hit candidates, never raw detection directly."""
from __future__ import annotations
import bisect, collections, copy, math
import numpy as np
from modding import BeatClock
from .features import sample_context
from .regions import config_at_time, region_boundaries
from .continuation import rhythm_bias
ROLE={'drums':'midTransient','kick':'lowPercussion','snare':'midTransient','percussion':'highPercussion',
      'bass':'lowPercussion','vocals':'tonalChange','spoken':'midTransient','piano':'tonalChange',
      'chords':'tonalChange','melody':'tonalChange','lead':'tonalChange','background':'tonalChange','other':'midTransient'}


def _hit_times(points):
    """A double occupies both recovering hands at the same observed instant."""
    return [float(p['time']) for p in points for _ in range(2 if p.get('chord') else 1)]


def _event_evidence(event, all_events, event_times):
    from .arcaudio.settings import attention_source
    second=float(event['time'])
    nearby=all_events[bisect.bisect_left(event_times,second-.032):bisect.bisect_right(event_times,second+.032)]
    eligible=[]
    for other in nearby:
        admission=other.get('admission',{})
        # coincidentSources also contains rejected background detections. Only
        # evidence that passed source/importance admission can support a double.
        if (other.get('selected') or other.get('reason') in ('attention_budget','redundant_simultaneous')) and (
                admission.get('importancePassed',True) and not other.get('sustained')
                and other.get('timingEvidence','observed_onset')=='observed_onset'
                and float(other.get('strength',0))>=.60):
            eligible.append(other.get('source','other'))
    groups=sorted(set(attention_source(s) for s in eligible)-{'background','other'})
    return {'eventId':event.get('id'), 'eventTime':second,'phraseId':event.get('phraseId'),
        'timingEvidence':event.get('timingEvidence','observed_onset'),
        'observedAttack':bool(event.get('timingEvidence','observed_onset')=='observed_onset' and not event.get('sustained')),
        'strength':float(event.get('strength',0)), 'confidence':float(event.get('confidence',0)),
        'temporalSalience':float(event.get('temporalSalience',1)),
        'rhythmicSalience':float(event.get('rhythmicSalience',0)),
        'coincidentSources':list(event.get('coincidentSources',[])),
        'eligibleCoincidentSources':sorted(set(eligible)), 'eligibleCoincidentGroups':groups,
        'admission':copy.deepcopy(event.get('admission',{}))}


def _accent_doubles(points, quota, minimum, hands, intent, model, phrases, neighboring_times, carry=0.):
    """Learned accent frequency selects extra hits, never extra musical times.

    Supported singles remain intact. Only spare *note* quota may become a
    double, with recovery on both sides and actual foreground attack evidence.
    The fractional learned allowance carries between phrases so short phrases
    do not systematically round every possible double away.
    """
    from .styles import GOALS
    rank=max(1,min(9,int(intent.get('rank',9))))
    goal=intent.get('styleGoal',GOALS.get(intent.get('style','Modern'),GOALS['Modern']))
    philosophy=intent.get('philosophy',{}); profile=intent.get('profile',{})
    movement=float(philosophy.get('movement',50))/100
    stamina=float(philosophy.get('stamina',profile.get('stamina',75)))/100
    objective=(.55+1.2*float(goal[3]))*(.35+.075*rank)*(.8+.4*movement)*(.8+.4*stamina)
    learned=np.asarray(model.get('doubles',np.zeros((3,16))),dtype=float)
    feasible=[]; rejected=collections.Counter()
    for point in points:
        evidence=point['audioEvidence'];reason=None
        state=str(point.get('attentionState','')).lower()
        phase=int(round(float(point['musicalBeat'])*4))%16
        energy_index=min(2,max(0,int(float(point['context'][0])*2)))
        probability=float(learned[energy_index,phase]) if learned.shape==(3,16) else 0.
        probability=float(np.clip(probability,0,1)) if np.isfinite(probability) else 0.
        admission=evidence['admission']
        if hands!=2:reason='one_saber'
        elif state in ('silence','vocal_focused','low_energy_vocal_space','vocal_space'):reason='intentional_vocal_space'
        elif (not evidence['observedAttack'] or abs(float(point['time'])-evidence['eventTime'])>.01801
              or admission.get('importancePassed') is False):reason='not_admitted_observed_attack'
        elif point.get('source') in ('background','other') or evidence['strength']<.65 or float(point['importance'])<.34:
            reason='weak_foreground_accent'
        else:
            beat=float(point['musicalBeat']); integer_accent=abs(beat-round(beat))<=.10
            phrase_accent=any(abs(float(point['time'])-float(p[k]))<=.08
                for p in phrases if isinstance(p,dict) for k in ('start','end') if k in p)
            coincident=len(evidence['eligibleCoincidentGroups'])>=2
            prominent=(evidence['temporalSalience']>=1.05 or evidence['strength']>=.85)
            aligned=evidence['rhythmicSalience']>=.85 and integer_accent
            if not (coincident or prominent and (aligned or phrase_accent)):reason='not_a_musical_accent'
            elif probability<=0:reason='no_learned_double_support'
            elif any(0.0001<abs(float(t)-float(point['time']))<minimum-.0001 for t in neighboring_times):
                reason='both_hands_need_recovery'
        point['accentDecision']={'eligible':reason is None,'reason':reason or 'eligible_foreground_accent',
            'learnedProbability':round(probability,5),'objectiveScale':round(objective,5)}
        if reason:rejected[reason]+=1
        else:
            support=1.+.15*max(0,len(evidence['eligibleCoincidentGroups'])-1)
            priority=float(point['importance'])*support*(1.1 if phase==0 else 1.)
            feasible.append((priority,point,min(.60,probability*objective)))
    extra=max(0,int(quota)-len(points))
    max_fraction=min(.40,.12+.35*float(goal[3]))*(.35+.075*rank)
    learned_allowance=carry+sum(p for _,_,p in feasible)
    allowance=max(0,int(math.floor(learned_allowance+1e-8)))
    carry=learned_allowance-math.floor(learned_allowance+1e-8)
    cap=min(extra,allowance,int(math.ceil(len(points)*max_fraction)))
    accented=[]
    for _,point,_ in sorted(feasible,key=lambda item:(-item[0],item[1]['time'])):
        if len(accented)>=cap:
            point['accentDecision']['reason']='note_budget' if extra<=len(accented) else 'learned_accent_allowance'
            rejected[point['accentDecision']['reason']]+=1
            continue
        point['chord']=True;point['accentDecision']['reason']='learned_observed_accent'
        accented.append(point)
    return carry,{'eligibleAccents':len(feasible),'accentDoubles':len(accented),
        'chosenHits':len(points)+len(accented),'noteQuota':int(quota),
        'accentRejections':dict(rejected),'learnedAccentAllowance':allowance}

def select_attention_candidates(analysis,config,rng,model,clock,contexts,legacy=None):
    from .arcaudio import analyze_attention, apply_focus_to_result
    from .arcaudio.focus import focus_regions_from_config, focus_decision, focus_event_times, same_focus_region
    mode=config.get('arcAudioMode','legacy')
    focus_regions=focus_regions_from_config(config,analysis['duration'])
    result=analysis.get('_arcAudioAttention') or analyze_attention(analysis,config.get('arcAudioSettings',{}),mode=mode,
        compute=config.get('analysisCompute'),mapping_regions=config.get('_attentionMappingRegions',config.get('mappingRegions',[])),
        focus_regions=config.get('sourceFocusRegions',[]))
    result=apply_focus_to_result(result,analysis,focus_regions)
    frames=result.get('frames',[]);events=result.get('selectedEvents',[e for e in result.get('events',[]) if e.get('selected')])
    # The analyzer's information quota is independent of the requested map
    # difficulty. Source choice, importance, sustained/background rejection and
    # coincidence deduplication have already happened before this quota. Apply
    # one difficulty-aware density budget here, for every opt-in audio engine.
    # This never rescues supporting layers or events below minimum importance.
    if result.get('events'):
        events=[e for e in result['events'] if e.get('selected') or e.get('reason')=='attention_budget']
    rank=int(config.get('rank',9));minimum=float(config.get('minimumHandInterval',.12 if rank>=9 else .16 if rank>=7 else .22 if rank>=5 else .30))
    if config.get('profile'):minimum*=1+(100-float(config['profile'].get('speed',85)))*.002
    limit=(1 if config.get('mode')=='OneSaber' else 2)/minimum
    frame_times=[f['time'] for f in frames]
    def frame_at(t):return frames[max(0,min(len(frames)-1,bisect.bisect_right(frame_times,t)-1))] if frames else {}
    evidence_events=sorted(result.get('events',events),key=lambda e:float(e['time']))
    evidence_times=[float(e['time']) for e in evidence_events]
    focus_times=focus_event_times(evidence_events) if focus_regions else None
    boundaries_for_intent=region_boundaries(config.get('mappingRegions',[])+focus_regions,analysis['duration'])
    ownership_boundaries=sorted(set([0.,float(analysis['duration'])]+[float(r[k])
        for r in config.get('mappingRegions',[])+focus_regions for k in ('start','end')]))
    def same_segment(a,b):
        # Fade sampling edges are quota boundaries, not ownership boundaries.
        return bisect.bisect_right(ownership_boundaries,a)==bisect.bisect_right(ownership_boundaries,b)
    phrase_lookup={p['id']:p for p in result.get('phrases',[]) if isinstance(p,dict) and p.get('id') is not None}
    pool=[];ft,cv=contexts;offset=float(analysis.get('offset',0));bpm=float(config.get('bpm') or analysis['bpm'])
    grid=analysis.get('beatTimes') or []
    for event in events:
        t=float(event['time'])
        if t<.5 or t>analysis['duration']-.35:continue
        f=focus_decision(frame_at(t),t,focus_regions,event,focus_times) if focus_regions else frame_at(t)
        source=event.get('source','other');importance=float(event.get('importance',0))
        # Snap only when an observed attack agrees closely with the measured grid.
        # Vocal and expressive instrumental onsets keep their own actual times.
        original=t
        if grid:
            i=bisect.bisect_left(grid,t);near=min((grid[j] for j in (i-1,i) if 0<=j<len(grid)),key=lambda b:abs(b-t),default=t)
            if abs(near-t)<=.018 and same_segment(t,near):t=near
        musical=float(np.interp(t,grid,np.arange(len(grid)))) if grid else (t-offset)*bpm/60
        phrase=phrase_lookup.get(event.get('phraseId'),{})
        motif=event.get('phraseId')
        if phrase.get('motifId') is not None:
            from .arcaudio.settings import attention_source
            motif=f"attention-motif-{phrase['motifId']}:{attention_source(source)}"
        evidence=_event_evidence(event,evidence_events,evidence_times)
        evidence.update(motifId=phrase.get('motifId'),repeatConfidence=phrase.get('repeatConfidence'))
        pool.append({'time':t,'beat':clock.beat(t),'musicalBeat':musical,'context':sample_context(ft,cv,t),'chord':False,
            'role':ROLE.get(source,'midTransient'),'source':source,'section':f.get('state','attention'),
            'activity':f.get('rhythmDensity',0),'audioMotif':motif,'timingError':abs(t-original),
            'score':importance*rhythm_bias(config,t,clock.beat(t)),'importance':importance,'attentionConfidence':f.get('confidence',0),
            'attentionState':f.get('state'),'attentionDensity':f.get('densityTarget',.2),'targetNps':0,
            'audioEvidence':evidence})
    if mode in ('hybrid','arcaudio3') and legacy:
        foreground=sorted(pool,key=lambda p:p['time']);selected_times=[p['time'] for p in foreground]
        for point in legacy:
            t=point['time'];f=frame_at(t);role=point.get('role')
            # Legacy grid support must refer to an eligible foreground event,
            # not merely a loud frame with a drum/bass label. Otherwise a high
            # importance threshold can be bypassed by the fallback branch.
            nearby=foreground[bisect.bisect_left(selected_times,t-.035):bisect.bisect_right(selected_times,t+.035)]
            def compatible(evidence):
                source=evidence.get('source')
                roles=({'lowPercussion','midTransient','highPercussion'} if source in ['drums','kick','snare','percussion'] else
                    {'lowPercussion','tonalChange'} if source=='bass' else
                    {'tonalChange','midTransient'} if source in ['piano','melody','lead','vocals','spoken','chords'] else set())
                return role in roles
            nearby=[p for p in nearby if compatible(p) and same_segment(t,p['time'])
                    and (not focus_regions or same_focus_region(focus_regions,t,p['audioEvidence']['eventTime']))]
            if not nearby:continue
            evidence=max(nearby,key=lambda p:p['importance'])
            value=dict(point);value.update(source=evidence['source'],importance=evidence['importance'],chord=False,
                score=min(.65*point['score'],evidence['score']+.15),
                attentionConfidence=f.get('confidence',0),attentionState=f.get('state'),attentionDensity=f.get('densityTarget',.2),
                audioMotif=evidence.get('audioMotif'),audioEvidence=copy.deepcopy(evidence['audioEvidence']))
            pool.append(value)
    # One hit represents coincident musical information; do not independently
    # turn every source at the same moment into another physical note.
    merged=[]
    for p in sorted(pool,key=lambda p:(p['time'],-p['score'])):
        if merged and p['time']-merged[-1]['time']<.035 and same_segment(p['time'],merged[-1]['time']):
            if p['score']>merged[-1]['score']:merged[-1]=p
        else:merged.append(p)
    phrase_seconds=4*60/bpm;groups=collections.defaultdict(list)
    # Use the measured phrase clock, including local tempo changes.
    # A fixed global BPM window can split a slowed phrase and join unrelated
    # faster phrases, distorting their density quotas even with correct hits.
    phrase_edges=None
    duration=float(analysis['duration'])
    measured=[float(p['start']) for p in result.get('phrases',[]) if isinstance(p,dict) and 'start' in p]
    if not measured and grid:measured=[float(t) for t in grid[::4]]
    if measured:
        phrase_edges=sorted(set([0.,duration]+[t for t in measured if np.isfinite(t) and 0<t<duration]))
    boundaries=boundaries_for_intent
    for p in merged:
        phrase=(max(0,min(len(phrase_edges)-2,bisect.bisect_right(phrase_edges,p['time'])-1))
                if phrase_edges else int(max(0,p['time']-offset)//phrase_seconds))
        segment=max(0,min(len(boundaries)-2,bisect.bisect_right(boundaries,p['time'])-1))
        groups[(phrase,segment)].append(p)
    from .scheduling import select_supported
    chosen=[];density_rows=[];accent_carry=0.
    hands=1 if config.get('mode')=='OneSaber' else 2
    for (phrase,segment),group in sorted(groups.items()):
        density=float(np.mean([p['attentionDensity'] for p in group]))
        t=float(np.mean([p['time'] for p in group]));intent=config_at_time(config,t,activity=density) if config.get('mappingRegions') else config
        goal=float(intent.get('_resolvedNps',intent.get('nps',10)))
        contrast=float(intent.get('philosophy',{}).get('contrast',65))/100
        density=density**(.75+.5*contrast)
        start=max(boundaries[segment],phrase_edges[phrase] if phrase_edges else offset+phrase*phrase_seconds)
        end=min(boundaries[segment+1],phrase_edges[phrase+1] if phrase_edges else offset+(phrase+1)*phrase_seconds)
        observed_rate=len(group)/max(.1,end-start)
        # A confident onset can have an uncertain instrument label. That label
        # ambiguity must not repeatedly attenuate a clearly busy foreground.
        # Quiet passages still have few eligible observed events, while dense
        # backgrounds remain excluded before this stage.
        information=min(1.,observed_rate/max(goal,1.))
        density=max(density,information)
        if intent.get('dynamicDensity',True) is False:density=1.
        elif intent.get('style') in ['Light/easy','OST','Accuracy','Acc-focused']:density*=.65
        density=min(float(result.get('settings',{}).get('maxDensity',1.)),density)
        # A range already uses activity to resolve its value. Multiplying by
        # density again would silently turn 8–13 NPS into a 2–8 NPS target.
        target=min(limit,goal*float(result.get('settings',{}).get('maxDensity',1.))) if intent.get('npsRange') is not None else min(limit,goal*density)
        desired=max(0,int(round(target*max(0,end-start))))
        kept,selection=select_supported(group,desired,minimum,hands,_hit_times(chosen)[-hands:])
        for p in kept:p['targetNps']=round(target,3)
        # Both hands must be recovered around a double. Check kept hits plus
        # adjacent phrase evidence, preventing an accent at a phrase edge from
        # silently consuming recovery needed by the next musical attack.
        neighbors=_hit_times(chosen)[-hands:]+[p['time'] for p in kept]
        neighbors += [p['time'] for p in merged if p['time']>=end and p['time']<end+minimum]
        accent_carry,accents=_accent_doubles(kept,desired,minimum,hands,intent,model,
            result.get('phrases',[]),neighbors,accent_carry)
        chosen.extend(kept)
        # These boundaries are consumed by planning and note ownership, not just
        # drawn in the UI. Rounding upward can exclude a real first attack and
        # falsely report that composition removed it (or assign it to a neighbor).
        density_rows.append({'start':start,'end':end,'requestedNps':goal,'targetNps':round(target,3),
            **({'npsRange':list(intent['npsRange'])} if intent.get('npsRange') is not None else {}),
            'availableEvents':len(group),'chosenEvents':len(kept),**accents,
            'actualCandidateNps':round(accents['chosenHits']/max(.1,end-start),3),
            'backfilledEvents':selection['backfilledEvents'],
            'limitedBy':(selection['limitedBy'] or 'attention') if accents['chosenHits']<desired else None,
            'requestedTargetNps':goal, 'densityGoalPolicy':'activity_resolved_range' if intent.get('npsRange') is not None else 'dynamic_target',
            'targetLimit':'hand_recovery' if goal>limit else 'density_ceiling' if float(result.get('settings',{}).get('maxDensity',1.))<1 else None})
    chosen.sort(key=lambda p:p['time'])
    analysis['_arcAudioMapperReport']={'mode':mode,'version':result.get('version','2.0'),'detectedEvents':len(result.get('events',[])),
        'selectedByAttention':len(result.get('selectedEvents',[])),'eligibleForegroundEvents':len(events),
        'mapperCandidates':len(chosen),'mapperCandidateNotes':len(_hit_times(chosen)),
        'accentDoubles':sum(p.get('chord',False) for p in chosen),
        'eligibleAccents':sum(r['eligibleAccents'] for r in density_rows),
        'accentRejections':dict(sum((collections.Counter(r['accentRejections']) for r in density_rows),collections.Counter())),
        'accentPolicy':'Learned phase/energy double frequency, admitted observed foreground accents, spare note quota, and two-hand recovery; no extra timings.',
        'summary':result.get('summary',{}),
        'mappingFocus':result.get('mappingFocus',{}),
        'recognition':result.get('recognition',{}),'sourceAnalysisMethod':result.get('method'),
        'densityCurve':density_rows,'npsPolicy':'Requested NPS targets active musical phrases; quiet sections and human hand limits can reduce it.',
        'limits':result.get('limitations',[]),'timing':'Observed source events; only close beat matches are snapped.'}
    return chosen,minimum,limit
