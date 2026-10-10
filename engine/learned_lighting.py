"""Human light-event behavior conditions new audio hits; no chart is copied."""
from __future__ import annotations

def _weighted(distribution,rng,allowed=None,default=1):
    values=[];weights=[]
    items=distribution.items() if isinstance(distribution,dict) else []
    for key,weight in items:
        try:key=int(key);weight=float(weight)
        except (ValueError,TypeError):continue
        if weight>0 and (allowed is None or key in allowed):values.append(key);weights.append(weight)
    return rng.choices(values,weights=weights,k=1)[0] if values else default

def light_decision(profile,rng):
    """Return native event choices only when conditioned training has evidence."""
    light=profile.get('lighting',{})
    if not profile.get('ready') or not light.get('eventTypes') or not light.get('eventValues'):
        return {'ready':False}
    if not any(float(v)>0 for k,v in light['eventTypes'].items() if str(k) in {'0','1','2','3','4'}):return {'ready':False}
    rate=max(.1,min(20.,float(light.get('eventsPerSecond',2))))
    kind=_weighted(light['eventTypes'],rng,set(range(5)),1)
    values=light['eventValues'].get(str(kind),light['eventValues'])
    value=_weighted(values,rng,set(range(9)),2)
    motion_counts={k:float(light['eventTypes'].get(str(k),0)) for k in [8,9,12,13]}
    native_total=sum(float(v) for v in light['eventTypes'].values())
    motion={str(kind):{'weight':motion_counts[kind]/max(native_total,1e-8),
        'value':_weighted(light['eventValues'].get(str(kind),{}),rng,set(range(16)),1),
        'beatPhases':light.get('beatPhases',{}).get(str(kind),[])} for kind in motion_counts if motion_counts[kind]>0}
    # Explicit safety/effect bounds stay in the caller; learned flash frequency
    # does not bypass flicker opt-out or create unsupported environment events.
    return {'ready':True,'eventType':kind,
        'value':value,'brightness':max(.1,min(1.6,float(light.get('brightnessByType',{}).get(str(kind),light.get('meanBrightness',1))))),
        'minimumSpacing':max(.05,min(.5,1/rate)),
        'fadeFraction':max(0,min(1,float(light.get('fadeFraction',0)))),
        'boostFraction':max(0,min(1,float(light.get('boostFraction',0)))),
        'palette':light.get('commonColors',[]),'evidence':{'confidence':profile.get('confidence'),'method':profile.get('method')},
        'motion':motion,
        'matches':profile.get('matches',[])}
