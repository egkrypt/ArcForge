"""Optional phrase accents, with explicit clearance from the real note corridor."""
import bisect, collections, math, random
from modding import BeatClock
from engine.diagnostics import analyze_playability, repair_playability, recommend_jump_settings

def add_events(chart,analysis,config):
    events=config.get('events',{});rng=random.Random(int(config.get('seed',42))+31)
    clock=BeatClock(config.get('bpm') or analysis['bpm'],chart.get('bpmEvents',[]))
    notes=chart.get('colorNotes',[]);times=[clock.time(n['b']) for n in notes]
    # Avoidance objects occur only in audible breaks with a clear recovery zone.
    # Decorative walls are handled separately by Noodle and are non-interactable.
    grouped=collections.defaultdict(list)
    for note in notes:grouped[note['c']].append(note)
    if events.get('bombs'):
        for section in analysis.get('sections',[]):
            t=section['end']-.3
            if t<1 or t>analysis['duration']-.5:continue
            i=bisect.bisect_left(times,t)
            near=notes[max(0,i-5):min(len(notes),i+6)]
            if any(abs(clock.time(n['b'])-t)<.30 for n in near):continue
            x=0 if rng.random()<.5 else 3
            chart['bombNotes'].append({'b':clock.beat(t),'x':x,'y':2})
    if events.get('obstacles'):
        for section in analysis.get('sections',[]):
            if section.get('energy',1)>.55 or section['end']-section['start']<2:continue
            start=section['start']+.4;end=min(section['end']-.4,start+.75)
            if end<=start:continue
            i=bisect.bisect_left(times,start-.35);j=bisect.bisect_right(times,end+.35)
            occupied={n['x'] for n in notes[i:j]}
            # A narrow outer wall, with at least three lanes left open, never
            # covers a note or forces a duck during a rapid stream.
            free=[x for x in [0,3] if x not in occupied]
            if free:chart['obstacles'].append({'b':clock.beat(start),'x':rng.choice(free),'y':0,'d':clock.beat(end)-clock.beat(start),'w':1,'h':5})
    # Arcs follow existing same-hand phrases rather than adding scoring burden.
    # Chains are only attached to unoccupied sustained audio gaps on enabled maps.
    if config.get('arcs',True) and config.get('mode')!='NoArrows':
        for hand,sequence in grouped.items():
            last_end=-10
            for head,tail in zip(sequence,sequence[1:]):
                a=clock.time(head['b']);z=clock.time(tail['b']);gap=z-a
                if .3<=gap<=.7 and a-last_end>2.5 and head['d']!=8 and tail['d']!=8 and rng.random()<.25:
                    chart['sliders'].append({'b':head['b'],'c':hand,'x':head['x'],'y':head['y'],'d':head['d'],'mu':1,
                        'tb':tail['b'],'tx':tail['x'],'ty':tail['y'],'tc':tail['d'],'tmu':1,'m':0})
                    last_end=z
    if config.get('chains',False) and config.get('mode')!='NoArrows':
        for hand,sequence in grouped.items():
            for head,tail in zip(sequence,sequence[1:]):
                gap=clock.time(tail['b'])-clock.time(head['b'])
                if gap<.6 or head['d'] not in [0,1] or rng.random()>.12:continue
                direction=1 if head['d']==0 else -1
                target=head['y']+direction*.8
                if not 0<=target<=2:continue
                chart['burstSliders'].append({'b':head['b'],'c':hand,'x':head['x'],'y':head['y'],'d':head['d'],
                    'tb':round(head['b']+.25,6),'tx':head['x'],'ty':head['y']+direction,'sc':3,'s':.8})
    return chart
