"""Shared musical evidence for authored lighting, with independent visual density.

No analyzer is rerun here. Source roles, phrase clock, attention and song sections
come from the same analysis used by gameplay. Styles control visual articulation.
"""
from __future__ import annotations
import bisect,copy

STYLES={
 'AI':dict(spacing=.075,threshold=.20,motion=1.,fade=False,flickers=True,phraseOnly=False),
 'Minimal':dict(spacing=.65,threshold=.55,motion=.15,fade=True,flickers=False,phraseOnly=True),
 'Classic':dict(spacing=.18,threshold=.30,motion=.6,fade=False,flickers=False,phraseOnly=False),
 'Modern':dict(spacing=.10,threshold=.24,motion=1.,fade=False,flickers=True,phraseOnly=False),
 'Energetic':dict(spacing=.065,threshold=.20,motion=1.25,fade=False,flickers=True,phraseOnly=False),
 'Cinematic':dict(spacing=.4,threshold=.35,motion=.4,fade=True,flickers=False,phraseOnly=False),
 'Aggressive':dict(spacing=.055,threshold=.24,motion=1.5,fade=False,flickers=True,phraseOnly=False),
 'Tech':dict(spacing=.10,threshold=.30,motion=1.15,fade=False,flickers=False,phraseOnly=False),
 'High-BPM':dict(spacing=.045,threshold=.20,motion=1.3,fade=False,flickers=True,phraseOnly=False),
 'Reactive':dict(spacing=.08,threshold=.15,motion=.8,fade=False,flickers=False,phraseOnly=False),
}
SOURCE_ROLES={'kick':['lowPercussion'],'bass':['lowPercussion'],'snare':['midTransient'],
 'drums':['midTransient'],'percussion':['highPercussion'],'vocals':['tonalChange'],
 'spoken':['tonalChange'],'piano':['tonalChange'],'melody':['tonalChange'],'lead':['tonalChange'],'chords':['tonalChange']}


def lighting_evidence(analysis,config):
    style=config.get('lightingStyle','AI')
    if not isinstance(style,str) or style not in STYLES:raise ValueError('Choose a supported lighting style.')
    attention=analysis.get('_arcAudioAttention',{})
    music=analysis.get('musicAnalysis',{})
    sections=copy.deepcopy(music.get('sections',analysis.get('sections',[])))
    frames=attention.get('frames',[]);times=[f['time'] for f in frames]
    triggers=[]
    if attention.get('enabled') and frames:
        for event in attention.get('events',[]):
            frame=frames[max(0,min(len(frames)-1,bisect.bisect_right(times,event['time'])-1))]
            # Lighting can articulate more foreground accents than gameplay,
            # but never turns rejected accompaniment into an independent show.
            foreground=event.get('attentionSource')==frame.get('primarySource')
            if not event.get('selected') and not (foreground and event.get('reason') in ['attention_budget','redundant_nearby','low_importance'] and event.get('importance',0)>=.18):continue
            triggers.append({**event,'roles':SOURCE_ROLES.get(event['source'],event.get('roles',[])),
                'musicalSource':event['source'],'attentionImportance':event.get('importance',0)})
        origin='Shared ArcAudio foreground events and phrasing; independent visual articulation'
    else:
        triggers=copy.deepcopy(music.get('accents') or [{'time':t,'strength':.5,'roles':[]} for t in analysis.get('onsets',[])])
        origin='Shared legacy musical accents and sections'
    plan=config.get('_arcForgeMapPlan',{})
    plan_sections=plan.get('sections',[])
    plan_times=[s['start'] for s in plan_sections]
    for trigger in triggers:
        if not plan_times:continue
        section=plan_sections[max(0,bisect.bisect_right(plan_times,trigger['time'])-1)]
        if not section['start']<=trigger['time']<section['end']:continue
        trigger['mapIntent']={'targetNps':section.get('targetNps',0),
            'requestedNps':section.get('requestedNps',0),'goals':section.get('goals',{}),
            'lightingIntent':section.get('lightingIntent'),'mappingStyle':section.get('mappingStyle')}
    return {'style':style,'policy':STYLES[style],'triggers':triggers,'sections':sections,
        'report':{'style':style,'evidence':origin,'candidateTriggers':len(triggers),
            'sharedMapPlanSections':len(plan.get('sections',[])),
            'planConditionedTriggers':sum('mapIntent' in t for t in triggers),
            'tempoTimeline':analysis.get('tempoTimeline',[]),
            'learnedLighting':bool(config.get('learnLighting')),
            'limitations':['Native event lanes vary by environment; unsupported lanes may have no visible effect. Advanced imported group choreography is preserved, not reconstructed from basic lighting statistics.']}}
