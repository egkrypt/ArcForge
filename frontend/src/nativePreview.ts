/** Native V4 preview helpers. Browser geometry is an approximation, not the game.
 * Schema: https://bsmg.wiki/mapping/map-format/beatmap.html#njs-events
 */
type Clock={seconds:(beat:number)=>number,beat?:(seconds:number)=>number};
const finite=(n:any)=>typeof n==='number'&&Number.isFinite(n);
const supportedEasings=new Set([0,1,2,3,4,5,6,7,8,9]);
const ease=(type:number,t:number)=>{
  if(type===1)return t*t;if(type===2)return 1-(1-t)**2;
  if(type===3)return t<.5?2*t*t:1-(-2*t+2)**2/2;
  if(type===4)return 1-Math.cos(t*Math.PI/2);if(type===5)return Math.sin(t*Math.PI/2);
  if(type===6)return (1-Math.cos(Math.PI*t))/2;
  if(type===7)return t**3;if(type===8)return 1-(1-t)**3;
  if(type===9)return t<.5?4*t**3:1-(-2*t+2)**3/2;
  return t;
};

export function arcViewerLimitations(chart:any):string[]{
  const reasons=new Set<string>();
  if(chart?.njsEvents?.length)reasons.add('native V4.1 NJS events');
  for(const key of ['colorNotes','bombNotes','obstacles','waypoints'])
    if((chart?.[key]||[]).some((o:any)=>o.r))reasons.add('per-object lane rotations');
  for(const key of ['sliders','burstSliders'])
    if((chart?.[key]||[]).some((o:any)=>o.hr||o.tr))reasons.add('independent arc/chain lane rotations');
  // Red is encoded as zero. A blue-head/red-tail arc is still V4-only,
  // even though its tail metadata is falsey.
  if((chart?.sliders||[]).some((o:any)=>finite(o._v4TailColor)&&o._v4TailColor!==(o.c??0)))
    reasons.add('independent arc tail colors');
  const seen=new Set<any>();
  const visit=(value:any)=>{if(!value||typeof value!=='object'||seen.has(value))return;seen.add(value);
    if(Array.isArray(value)){value.forEach(visit);return;}
    for(const [key,item] of Object.entries(value)){if(key.startsWith('_v4')&&item&&(!Array.isArray(item)||item.length)&&
      (typeof item!=='object'||Object.keys(item).length))reasons.add('V4-only metadata');visit(item);}
  };visit(chart);return [...reasons];
}

export function createNativeNjsSampler(chart:any,clock:Clock,baseNjs:number){
  const warnings=new Set<string>();let previous=baseNjs;
  const rows=(chart?.njsEvents||[]).filter((e:any)=>finite(e.b)&&finite(e.d)).slice().sort((a:any,b:any)=>a.b-b.b)
    .map((e:any)=>{const value=e.p?previous:Math.max(.1,baseNjs+e.d);previous=value;
      const easing=finite(e.e)?e.e:0;
      if(!e.p&&!supportedEasings.has(easing))warnings.add(`NJS easing ${easing} is not reproduced; preview holds the preceding speed until that event.`);
      return {time:clock.seconds(e.b),value,extend:!!e.p,easing};});
  return {hasEvents:rows.length>0,warnings:[...warnings],speedAt(seconds:number){
    let before={time:0,value:baseNjs};
    for(const row of rows){
      if(seconds>=row.time){before=row;continue;}
      if(row.extend||!supportedEasings.has(row.easing))return before.value;
      const t=Math.max(0,Math.min(1,(seconds-before.time)/Math.max(.000001,row.time-before.time)));
      return before.value+(row.value-before.value)*ease(row.easing,t);
    }return before.value;
  }};
}

export function createLaneRotationSampler(chart:any){
  let cumulative=0;
  const events=(chart?.rotationEvents||[]).filter((e:any)=>finite(e.b)&&finite(e.r)).slice().sort((a:any,b:any)=>a.b-b.b||a.e-b.e)
    .map((e:any)=>({...e,value:(cumulative+=e.r)}));
  return (object:any,endpoint:'head'|'tail'='head')=>{
    const field=endpoint==='tail'?'tr':finite(object.hr)?'hr':'r';
    if(finite(object[field]))return object[field];
    const beat=endpoint==='tail'?object.tb:object.b;
    let lo=0,hi=events.length;
    while(lo<hi){const mid=(lo+hi)>>>1,e=events[mid];if(e.b<beat||e.b===beat&&e.e===0)lo=mid+1;else hi=mid;}
    return lo?events[lo-1].value:0;
  };
}

/** Positive game lane rotation is clockwise: a forward point moves right. */
export function rotateLanePoint(point:number[],degrees:number):number[]{
  const angle=degrees*Math.PI/180,c=Math.cos(angle),s=Math.sin(angle);
  return [point[0]*c-point[2]*s,point[1],point[0]*s+point[2]*c];
}

export function createLocalBpmSampler(chart:any,baseBpm:number,clock:Clock){
  const rows=(chart?.bpmEvents||[]).filter((e:any)=>finite(e.b)&&e.m>0).slice().sort((a:any,b:any)=>a.b-b.b)
    .map((e:any)=>({time:clock.seconds(e.b),bpm:e.m}));
  return (seconds:number)=>{let lo=0,hi=rows.length;
    while(lo<hi){const mid=(lo+hi)>>>1;if(rows[mid].time<=seconds)lo=mid+1;else hi=mid;}
    return lo?rows[lo-1].bpm:baseBpm;
  };
}

/** Native NJS preserves JD below base, and reaction time above base. */
export function nativeJumpParameters(baseNjs:number,baseJumpDistance:number,speed:number){
  const jumpDistance=baseJumpDistance*Math.max(1,speed/baseNjs);
  return {jumpDistance,reactionSeconds:jumpDistance/(2*speed)};
}
