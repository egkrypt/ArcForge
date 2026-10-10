"""Audio-triggered Chroma/Noodle/Vivify choreography and precision export.

Schemas follow https://heck.aeroluna.dev/ and MappingExtensions' official README.
Only peripheral scenery is animated automatically; playable notes remain readable.
"""
import bisect, colorsys, copy, math, random

class BeatClock:
    def __init__(self, bpm, events=()):
        self.changes = [(0., float(bpm))]
        for e in sorted(events, key=lambda e:e['b']):
            if e['b'] == 0:
                self.changes[0] = (0., float(e['m']))
            else:
                self.changes.append((float(e['b']), float(e['m'])))
        self.seconds = [0.]
        for (a, tempo), (b, _) in zip(self.changes, self.changes[1:]):
            self.seconds.append(self.seconds[-1] + (b-a)*60/tempo)
    def beat(self, seconds):
        i = max(0, bisect.bisect_right(self.seconds, seconds)-1)
        return round(self.changes[i][0] + (seconds-self.seconds[i])*self.changes[i][1]/60, 6)
    def time(self, beat):
        i = max(0, bisect.bisect_right([c[0] for c in self.changes], beat)-1)
        return self.seconds[i] + (beat-self.changes[i][0])*60/self.changes[i][1]

def mod_names(mods):
    known = {'chroma':'Chroma','noodle':'Noodle Extensions','noodleextensions':'Noodle Extensions',
        'noodle extensions':'Noodle Extensions','vivify':'Vivify','mappingextensions':'Mapping Extensions',
        'mapping extensions':'Mapping Extensions','mapping':'Mapping Extensions','customjsondata':'CustomJSONData'}
    return {known.get(str(m).lower(), str(m)) for m in mods}

def precision(coordinate):
    value = round(float(coordinate)*1000)
    return value+1000 if value >= 0 else value-1000

def decode_precision(coordinate):
    return (coordinate-1000)/1000 if coordinate >= 1000 else (coordinate+1000)/1000 if coordinate <= -1000 else coordinate

def apply_mods(chart, analysis, config, asset=None):
    chart = copy.deepcopy(chart)
    mods = mod_names(config.get('mods', []))
    clock = BeatClock(config.get('bpm') or analysis['bpm'], chart.get('bpmEvents', []))
    duration = float(analysis['duration'])
    seed = int(config.get('seed', 42))
    rng = random.Random(seed)
    hue = (seed * .61803398875) % 1
    learned=config.get('_learnedLighting',{})
    palette=[]
    for value in learned.get('commonColors',[]):
        rgb=value['rgb'];peak=max(rgb)
        if peak>.1 and colorsys.rgb_to_hsv(*(v/peak for v in rgb))[1]>.35:
            palette.append((rgb,value['count']))
    if palette:
        rgb=rng.choices([p[0] for p in palette],weights=[p[1] for p in palette])[0]
        hue=colorsys.rgb_to_hsv(*(v/max(rgb) for v in rgb))[0]
    # Training may discover an all-blue show. It must not silently recolor both
    # hands, or remove the paired palette users expect in an ordinary map.
    left = list(colorsys.hsv_to_rgb(hue, .85, 1)) + [1] if config.get('effects',{}).get('learnedSaberColors',False) else [1,.13,.22,1]
    right = list(colorsys.hsv_to_rgb((hue+.46) % 1, .85, 1)) + [1] if config.get('effects',{}).get('learnedSaberColors',False) else [.12,.40,1,1]
    white = [.85,.93,1,1]
    cd = chart.setdefault('customData', {})
    cd.setdefault('arcforge',{})['lighting']='Percussive-role and section choreography with learned palette statistics' if learned else 'Percussive-role and section choreography'
    env = cd.setdefault('environment', [])
    custom = cd.setdefault('customEvents', [])
    lights = []
    events_enabled = config.get('events', {})
    dynamic = events_enabled.get('lighting', True)
    chroma = 'Chroma' in mods
    noodle = 'Noodle Extensions' in mods
    # Twelve discrete light fixtures leave the entire note corridor open.
    ids = {'left':[], 'right':[], 'top':[]}
    if chroma:
        cd.setdefault('materials', {})['arcforgeLight'] = {'shader':'TransparentLight','color':[1,1,1,1]}
        for side in [-1,1]:
            for depth in range(4):
                li = 100+depth*2+(0 if side<0 else 1)
                ids['left' if side<0 else 'right'].append(li)
                env.append({'geometry':{'type':'Cube','material':'arcforgeLight','collision':False},
                    'position':[side*(6+depth*1.5),4+depth*1.5,18+depth*8],
                    'scale':[.1,7+depth*2,.1],'rotation':[0,0,side*(14+depth*9)],
                    'track':f'forgeWing{side}_{depth}',
                    'components':{'ILightWithId':{'type':1,'lightID':li},
                        'TubeBloomPrePassLight':{'colorAlphaMultiplier':1,'bloomFogIntensityMultiplier':.20}}})
        for i in range(4):
            li = 108+i;ids['top'].append(li)
            env.append({'geometry':{'type':'Cube','material':'arcforgeLight','collision':False},
                'position':[0,11+i*2,40+i*8],'scale':[18+i*4,.08,.08],
                'rotation':[0,0,(-1 if i%2 else 1)*8],'track':f'forgeSky{i}',
                'components':{'ILightWithId':{'type':1,'lightID':li}}})
    for et in range(5):
        lights.append({'b':0,'et':et,'i':0,'f':0})
    all_ids = ids['left']+ids['right']+ids['top']
    if chroma:
        lights.append({'b':0,'et':1,'i':0,'f':0,'customData':{'lightID':all_ids,'color':white}})
    frames = analysis.get('frames', [])
    music = analysis.get('musicAnalysis', {})
    from engine.lighting_plan import lighting_evidence
    lighting=lighting_evidence(analysis,config);policy=lighting['policy'];lighting_style=lighting['style']
    sections=lighting['sections']
    cd['arcforge']['lightingPlan']=lighting['report']
    frame_times = [f['time'] for f in frames]
    composition_cache={};composition_colors={}
    def composition_at(time):
        if not config.get('learnLighting',False) or not config.get('_compositionModelDir'):return {'ready':False}
        from engine.composition import composition_profile
        key=int(time*float(config.get('bpm') or analysis['bpm'])/240)
        if key not in composition_cache:composition_cache[key]=composition_profile(config['_compositionModelDir'],analysis,time,config)
        return composition_cache[key]
    def feature(time):
        return frames[max(0,min(len(frames)-1,bisect.bisect_right(frame_times,time)-1))] if frames else {'bass':1,'onset':1,'energy':.5,'high':.5}
    def section_at(time):
        return next((s for s in sections if s['start']<=time<s['end']), {'label':'Drive','activity':.5,'energy':.5})
    if dynamic:
        previous = -100; last_flicker = -100; last_rotation=-100; last_side=None
        triggers=lighting['triggers']
        for i, trigger in enumerate(triggers):
            time=trigger['time'];roles=trigger.get('roles',[]);section=section_at(time);label=section.get('label','Drive')
            f = feature(time)
            profile=composition_at(time)
            human_light=profile.get('lighting',{}) if profile.get('ready') else {}
            # Refrain from lighting every weak vocal transient; keep strong kick,
            # snare and high-frequency flourishes audible through the scene.
            melodic = roles == ['tonalChange']
            local_period=60/(next((s['bpm'] for s in analysis.get('tempoTimeline',[]) if s['start']<=time<s['end']),config.get('bpm') or analysis['bpm']))
            spacing = max(policy['spacing'],.24 if label in ['Intro','Break','Outro'] or melodic else local_period*.18)
            if lighting_style=='Cinematic':spacing=max(spacing,local_period)
            if policy['phraseOnly'] and (clock.beat(time)%4)>.35:continue
            if lighting_style=='Classic' and (clock.beat(time)%.5)>.15:continue
            if human_light.get('eventTypes'):
                rate=max(.1,min(20.,float(human_light.get('eventsPerSecond',2))))
                spacing=max(spacing,min(.5,1/rate))
            if trigger.get('strength',f['onset']) < policy['threshold'] or time-previous < spacing:
                continue
            previous = time
            low = 'lowPercussion' in roles if roles else f['bass']>.5
            high = 'highPercussion' in roles if roles else f['high']>.35
            color = left if low else right if high else white
            contrast = .22 if label in ['Intro','Break','Outro'] else .55 if label=='Build' else 1. if label=='Drop' else .7
            intent=trigger.get('mapIntent',{})
            if intent:
                contrast_goal=float(intent.get('goals',{}).get('contrast',65))/100
                contrast=contrast**(.6+.8*max(0,min(1,contrast_goal)))
            intensity = min(1.25, (.12+trigger.get('strength',f['onset'])*.45+f['energy']*.25)*contrast)
            if intent.get('requestedNps',0)>0:
                density_ratio=min(1.,float(intent['targetNps'])/float(intent['requestedNps']))
                intensity*=.70+.30*density_ratio
            decision={'ready':False}
            if human_light.get('eventTypes'):
                from engine.learned_lighting import light_decision
                decision=light_decision(profile,rng)
            if decision['ready']:
                phase=int(round(clock.beat(time)*4))%16
                phase_weights=human_light.get('beatPhases',{}).get(str(decision['eventType']),[])
                if phase_weights and max(phase_weights)>0 and trigger.get('strength',f['onset'])<.25+.15*(1-phase_weights[phase]/max(phase_weights)):continue
                intensity=min(1.25,intensity*(.6+.4*decision['brightness']))
                if decision['value']==0:intensity=0
                palette=decision['palette'];color_key=(profile.get('phraseStart'),low,high)
                if color_key not in composition_colors and palette:
                    valid=[p for p in palette if isinstance(p.get('rgb'),list) and len(p['rgb'])>=3 and max(p['rgb'])>.1]
                    if valid:
                        rgb=rng.choices([p['rgb'] for p in valid],weights=[max(1,p.get('count',1)) for p in valid])[0]
                        peak=max(rgb);composition_colors[color_key]=[max(0,min(1,v/peak)) for v in rgb[:3]]+[1]
                color=composition_colors.get(color_key,color)
                cd['arcforge']['learnedLighting']={'method':'Audio-conditioned human phrase/event distributions',
                    'evidence':decision['evidence'],'features':['event types','event values','beat phases','brightness','fade behavior','event spacing','phrase colors','color boosts'],
                    'limitation':'Basic events are learned; imported event-box/custom-shader choreography is not reconstructed.'}
            beat = clock.beat(time)
            group = 'left' if low else 'right' if high else 'top'
            # Preserve the learned on/flash/fade behavior while balancing native
            # color banks over musical phrases. Chroma also receives both sides
            # of the paired palette; secondary fixtures don't inherit one tint.
            value=decision['value'] if decision['ready'] else 3 if policy['fade'] or melodic or label in ['Intro','Break','Outro'] else 2
            side=(int(beat//8)%2) ^ (0 if group=='left' else 1 if group=='right' else int(beat//2)%2)
            if value in [1,2,3,5,6,7]:value=(value-1)%4+1+(4 if side==0 else 0)
            if chroma and group!='top':
                color=left if side==0 else right
            last_side=side
            if chroma:
                lights.append({'b':beat,'et':1,'i':value,'f':round(intensity,3),
                    'customData':{'lightID':ids[group],'color':color}})
                # The native environment participates as well as custom geometry.
                # This also provides a faithful basic-light fallback in ArcViewer.
                native_lane=0 if melodic else 1 if low else 2+side if high else 4
                lights.append({'b':beat,'et':decision['eventType'] if decision['ready'] else native_lane,'i':value,'f':round(intensity*.55,3),
                    'customData':{'color':left if side==0 else right}})
            else:
                native_lane=0 if melodic else 1 if low else 2+side if high else 4
                lights.append({'b':beat,'et':decision['eventType'] if decision['ready'] else native_lane,'i':value,'f':round(intensity,3)})
            rotation_spacing=max(.28,local_period*(2 if label=='Drop' else 4)/policy['motion'])
            if label in ['Drop','Build'] and low and trigger.get('strength',0)>.55 and time-last_rotation>rotation_spacing:
                last_rotation=time;direction=int(beat//4)%2
                human_motion=decision.get('motion',{})
                ring_phase=human_motion.get('8',{}).get('beatPhases',[])
                if ring_phase and max(ring_phase)>0:direction=int(ring_phase[phase if decision['ready'] else int(round(beat*4))%16]>=max(ring_phase)*.5)
                lights.append({'b':beat,'et':8,'i':1,'f':1,**({'customData':{
                    'rotation':round((24 if label=='Drop' else 10)*policy['motion'],2),
                    'step':4+int(beat//8)%5,'prop':1.1,'speed':.8+float(section.get('activity',.5)),
                    'direction':direction}} if chroma else {})})
                for et in [12,13]:
                    observed_speed=human_motion.get(str(et),{}).get('value')
                    speed=max(.3,min(5.,float(observed_speed))) if observed_speed is not None else 1.2+float(section.get('activity',.5))*3
                    lights.append({'b':beat,'et':et,'i':max(1,round(speed)),'f':1,
                        **({'customData':{'speed':speed,'direction':(direction+et)%2,'lockRotation':True}} if chroma else {})})
            # Rare accent flurries are tied to pronounced treble hits.
            if high and label=='Drop' and trigger.get('strength',0)>.85 and time-last_flicker>2 and policy['flickers'] and config.get('effects',{}).get('flickers',True) and int(beat*4)%8<2:
                last_flicker=time
                for delta in [local_period/8,local_period/4,local_period*3/8]:
                    if time+delta<duration:
                        off=abs(delta-local_period/4)>.00001
                        e={'b':clock.beat(time+delta),'et':1 if chroma else 4,'i':0 if off else 2,'f':0 if off else .8}
                        if chroma:e['customData']={'lightID':ids['top'],'color':white}
                        lights.append(e)
        for i, sec in enumerate(sections):
            time=sec['start'];b=clock.beat(time);energy=sec.get('energy',.5);activity=sec.get('activity',energy);label=sec.get('label','Drive')
            brightness=.045 if label in ['Intro','Break','Outro'] else .12+activity*.25
            human_light=composition_at(time).get('lighting',{})
            if float(human_light.get('boostFraction',0))>0:
                chart.setdefault('colorBoostBeatmapEvents',[]).append({'b':b,'o':label in ['Drop','Build'] and activity>.55})
            if chroma:
                for group,color in [('left',left),('right',right),('top',white)]:
                    lights.append({'b':b,'et':1,'i':1,'f':round(brightness,3),
                        'customData':{'lightID':ids[group],'color':color}})
            if label in ['Drop','Build']:
                lights.append({'b':b,'et':8,'i':1,'f':1,'customData':{'rotation':15 if label=='Drop' else 5,'step':3,'prop':1,'speed':round(.2+activity,2)}})
            for et in [12,13]:
                lights.append({'b':b,'et':et,'i':max(0,round(activity*4)) if label not in ['Intro','Break','Outro'] else 0,'f':1})
    else:
        # Static low brightness is still a visible environment.
        for et in range(5):lights.append({'b':0,'et':et,'i':1,'f':.3})
        if chroma:lights.append({'b':0,'et':1,'i':1,'f':.25,'customData':{'lightID':all_ids,'color':white}})
    if noodle:
        # Musical phrase entrances can arc into view, but reach the native grid
        # well before the cut. No time warp, dissolve or hit position change is
        # applied to real notes. Arc/chain heads stay on their original paths.
        heads={(round(s['b'],5),s.get('c')) for key in ['sliders','burstSliders'] for s in chart.get(key,[])}
        if config.get('effects',{}).get('noteEntrances',True):
            for note in chart.get('colorNotes',[]):
                time=clock.time(note['b']);sec=section_at(time)
                if sec.get('label') not in ['Drop','Build'] or (round(note['b'],5),note.get('c')) in heads:continue
                side=-1 if note['c']==0 else 1;phrase=int(note['b']//4)
                amount=.18+.08*(phrase%3);direction=1 if phrase%2 else -1
                note.setdefault('customData',{}).setdefault('animation',{}).update({
                    'offsetPosition':[[side*amount,direction*.16,0,0],[0,0,0,.28,'easeOutSine'],[0,0,0,1]],
                    'localRotation':[[0,0,side*direction*12,0],[0,0,0,.28,'easeOutSine'],[0,0,0,1]]})
        cd['arcforge']['noteEntrancePolicy']='Small phrase-dependent approach motion; zero transform from 28% of jump life through the hit. Sustain heads unchanged.'
        fake_walls=cd.setdefault('fakeObstacles', [])
        for i,sec in enumerate(sections):
            b=clock.beat(sec['start']);energy=float(sec.get('energy',.5));activity=float(sec.get('activity',energy));label=sec.get('label','Drive')
            if chroma:
                for side in [-1,1]:
                    custom.append({'b':b,'t':'AnimateTrack','d':{
                        'track':[f'forgeWing{side}_{depth}' for depth in range(4)],
                        'duration':min(8,max(1,clock.beat(sec['end'])-b)),
                        'localRotation':[[0,0,0,0],[0,0,side*(2+activity*16),1,'easeOutSine']]}})
            if label=='Drop' and activity>.65:
                for side in [-1,1]:
                    fake_walls.append({'b':b,'x':0,'y':0,'d':1,'w':1,'h':1,'customData':{
                        'coordinates':[side*(8+i%3),3+i%2],'size':[.1,3,.2],
                        'uninteractable':True,'color':left if side<0 else right,
                        'noteJumpMovementSpeed':12,'noteJumpStartBeatOffset':0,
                        'animation':{'offsetPosition':[[0,0,0,0],[side*.5,2,0,1,'easeOutSine']],
                            'dissolve':[[0,0],[.45,.15],[.45,.65],[0,1]],
                            'localRotation':[[0,0,side*20,0],[0,0,side*50,1,'easeOutSine']]}}})
    if 'Mapping Extensions' in mods:
        # Precision placement makes a small controllable lane expansion. All
        # directional notes, arc endpoints and chains retain their swing paths.
        for n in chart.get('colorNotes', []):
            n['x']=precision(n['x']+(-.08 if n['c']==0 else .08))
            n['y']=precision(n['y'])
        for key in ['sliders','burstSliders']:
            for a in chart.get(key, []):
                delta=-.08 if a['c']==0 else .08
                for p in ['x','tx']:a[p]=precision(a[p]+delta)
                for p in ['y','ty']:a[p]=precision(a[p])
    if 'Vivify' in mods:
        if not asset:
            raise ValueError('Vivify needs an imported Unity .vivify bundle, its checksum, and a prefab path. Add one in Assets first.')
        for i, sec in enumerate(sections):
            if sec.get('label')!='Drop' and sec.get('activity',sec.get('energy',0))<.70:
                continue
            identity=f'forgePrefab{i}'
            b=clock.beat(sec['start']);end=clock.beat(sec['end'])
            custom.append({'b':b,'t':'InstantiatePrefab','d':{
                'asset':asset['prefab'].lower(),'id':identity,'track':identity,
                'position':[0,8,32],'scale':[1,1,1]}})
            custom.append({'b':end,'t':'DestroyObject','d':{'id':identity}})
    lights.append({'b':clock.beat(max(0,duration-.04)),'et':1,'i':0,'f':0,
        **({'customData':{'lightID':all_ids,'color':white}} if chroma else {})})
    chart['basicBeatmapEvents']=sorted(lights,key=lambda e:(e['b'],e['et']))
    cd['arcforge']['lightingPlan'].update(eventCount=len(lights),
        rotationCount=sum(e['et']==8 for e in lights),
        nativeColorBanks=sorted(set('red' if e['i'] in [5,6,7] else 'blue' for e in lights if e['et']<5 and e['i'] in [1,2,3,5,6,7])))
    custom.sort(key=lambda e:e['b'])
    # Framework is tracked in dependencies, rather than emitted as an unknown
    # gameplay requirement that some custom-song loaders cannot resolve.
    requirements=sorted(mods-{'CustomJSONData','Heck','Synapse'})
    return chart, {'requirements':requirements,'dependencies':['CustomJSONData','Heck'] if mods & {'Chroma','Noodle Extensions','Vivify'} else ['CustomJSONData'] if 'CustomJSONData' in mods else [],
        'colorLeft':dict(zip(['r','g','b'],left[:3])),'colorRight':dict(zip(['r','g','b'],right[:3])),
        'visualScope':'Peripheral choreography; playable notes are not hidden or moved at hit time.'}
