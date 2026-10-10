/** Original, deterministic preview of the documented Heck/Chroma data subset.
 * This is not the game mod runtime. Unknown data is preserved by the editor.
 * Reference: https://heck.aeroluna.dev/ and MappingExtensions README.
 */
export const MOD_CAPABILITIES = [
 {name:'Chroma',url:'https://github.com/Aeroluna/Chroma',preview:'Colors, independent light groups/IDs, ring and laser rotation, primitive environment geometry.',arc:'Colors, basic lights/IDs, ring and laser cues. Environment enhancement is limited.'},
 {name:'Noodle Extensions',url:'https://github.com/Aeroluna/Heck',preview:'Coordinates, local/world rotation, scale, dissolve, AnimateTrack, AssignPathAnimation, track parenting and time curves.',arc:'Precision placement. Advanced track choreography requires the Mod preview.'},
 {name:'Mapping Extensions',url:'https://github.com/Kylemc1413/MappingExtensions',preview:'Extended lanes/layers, precision coordinates and 360° cut directions.',arc:'Precision placement and rotation. Use separately from Noodle Extensions.'},
 {name:'Vivify',url:'https://github.com/Aeroluna/Vivify',preview:'Asset/event data inspector and export. Compiled Unity AssetBundles and shaders require the PC game runtime.',arc:'Compiled custom assets are not reproduced by this WebGL player.'},
 {name:'CustomJSONData',url:'https://github.com/Aeroluna/CustomJSONData',preview:'Preserves arbitrary custom data in maps. Framework dependency; no independent visual effect.',arc:'Recognized custom map data is loaded; unknown fields are retained in export.'},
 {name:'Heck',url:'https://github.com/Aeroluna/Heck',preview:'Shared track/keyframe data and dependency metadata. Preview covers the listed animation subset.',arc:'The player is not a complete Heck runtime.'},
 {name:'Synapse',url:'https://github.com/Aeroluna/Synapse',preview:'Live event/network integration reference. It is not a visual beatmap extension.',arc:'Requires its own server and in-game client. No simulated connection.'},
];
export const clamp=(v:number,min=0,max=1)=>Math.min(max,Math.max(min,v));
export const decodeCoordinate=(v:number)=>Math.abs(v)>=1000?(v-Math.sign(v)*1000)/1000:v;
export const cutRotation=(direction:number,angle=0)=>direction>=1000&&direction<=1360?180-(direction-1000):direction>=2000&&direction<=2360?180-(direction-2000):([0,180,90,-90,45,-45,135,-135,0][direction]||0)-angle;
export const isDot=(direction:number)=>direction===8||direction>=2000&&direction<=2360;
const lookup=(obj:any,key:string)=>obj?.[key]??obj?.[`_${key}`];
export const custom=(obj:any)=>obj?.customData??obj?._customData??{};
export function easing(name:string,t:number):number {
 if(name==='easeStep')return t<1?0:1;
 if(name==='easeInOutSine')return -(Math.cos(Math.PI*t)-1)/2;
 if(name==='easeInSine')return 1-Math.cos(t*Math.PI/2);
 if(name==='easeOutSine')return Math.sin(t*Math.PI/2);
 if(name==='easeInQuad')return t*t;
 if(name==='easeOutQuad')return 1-(1-t)*(1-t);
 if(name==='easeInOutQuad')return t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2;
 if(name==='easeInCubic')return t*t*t;
 if(name==='easeOutCubic')return 1-Math.pow(1-t,3);
 if(name==='easeInOutCubic')return t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
 if(name==='easeInExpo')return t===0?0:Math.pow(2,10*t-10);
 if(name==='easeOutExpo')return t===1?1:1-Math.pow(2,-10*t);
 return t;
}
export function samplePoint(value:any,progress:number,dimensions=3,definitions:any={}):number[]|undefined {
 if(typeof value==='string')value=definitions[value];
 if(!Array.isArray(value)||!value.length)return;
 if(typeof value[0]==='number')return value.slice(0,dimensions);
 const keys=value.filter((p:any)=>Array.isArray(p)&&p.slice(0,dimensions+1).every((v:any)=>typeof v==='number'&&Number.isFinite(v))).sort((a:any,b:any)=>a[dimensions]-b[dimensions]);
 if(!keys.length)return;
 if(progress<=keys[0][dimensions])return keys[0].slice(0,dimensions);
 if(progress>=keys[keys.length-1][dimensions])return keys[keys.length-1].slice(0,dimensions);
 let i=0;while(i<keys.length-2&&progress>keys[i+1][dimensions])i++;
 const a=keys[i],b=keys[i+1];let t=clamp((progress-a[dimensions])/(b[dimensions]-a[dimensions]||1));t=easing(b.find((v:any)=>typeof v==='string'&&v.startsWith('ease'))||'easeLinear',t);
 if(b.includes('splineCatmullRom')&&dimensions===3){const prev=keys[Math.max(0,i-1)],next=keys[Math.min(keys.length-1,i+2)];return a.slice(0,dimensions).map((_:number,j:number)=>.5*((2*a[j])+(-prev[j]+b[j])*t+(2*prev[j]-5*a[j]+4*b[j]-next[j])*t*t+(-prev[j]+3*a[j]-3*b[j]+next[j])*t*t*t));}
 return a.slice(0,dimensions).map((v:number,j:number)=>v+(b[j]-v)*t);
}
type Animation={position:number[],rotation:number[],scale:number[],color?:number[],dissolve:number,arrow:number,time?:number,definite?:number[]};
const neutral=():Animation=>({position:[0,0,0],rotation:[0,0,0],scale:[1,1,1],dissolve:1,arrow:1});
export function createModRuntime(chart:any,clock:{seconds:(b:number)=>number}) {
 const cd=custom(chart),events=(lookup(cd,'customEvents')||[]).slice().sort((a:any,b:any)=>(a.b??a._time??0)-(b.b??b._time??0));
 const defs=lookup(cd,'pointDefinitions')||{};const definitions=Array.isArray(defs)?Object.fromEntries(defs.map((d:any)=>[d.name??d._name,d.points??d._points])):defs;
 const trackEvents=new Map<string,any[]>();for(const event of events){const data=event.d??event._data??{},tracks=lookup(data,'track'),type=event.t??event._type,list=type==='AssignTrackParent'?lookup(data,'childrenTracks')||[]:Array.isArray(tracks)?tracks:tracks?[tracks]:[];for(const track of list){if(!trackEvents.has(track))trackEvents.set(track,[]);trackEvents.get(track)!.push(event);}}
 const apply=(target:Animation,data:any,p:number)=>{
  const v=(key:string,n=3)=>samplePoint(lookup(data,key),p,n,definitions);
  const pos=v('offsetPosition')??v('position'),rot=v('offsetWorldRotation')??v('localRotation')??v('rotation'),scale=v('scale'),color=v('color',4),dissolve=v('dissolve',1),arrow=v('dissolveArrow',1),time=v('time',1),definite=v('definitePosition');
  if(pos)target.position=pos;if(rot)target.rotation=rot;if(scale)target.scale=scale;if(color)target.color=color;if(dissolve)target.dissolve=clamp(dissolve[0]);if(arrow)target.arrow=clamp(arrow[0]);if(time)target.time=time[0];if(definite)target.definite=definite;
 };
 const match=(event:any,track:string)=>{const data=event.d??event._data;const t=lookup(data,'track');return t===track||Array.isArray(t)&&t.includes(track);};
 const trackState=(track:string,seconds:number,pathProgress:number,visited=new Set<string>()):Animation=>{
  const result=neutral(),pathData:any={};let parentState:Animation|undefined;if(visited.has(track))return result;visited.add(track);
  for(const event of trackEvents.get(track)||[]){const beat=event.b??event._time??0,start=clock.seconds(beat);if(start>seconds)break;const type=event.t??event._type,data=event.d??event._data??{};
   if(type==='AssignTrackParent'&&(lookup(data,'childrenTracks')||[]).includes(track)){parentState=trackState(lookup(data,'parentTrack'),seconds,pathProgress,new Set(visited));}
   if(!match(event,track))continue;
   if(type==='AssignPathAnimation')for(const [key,value] of Object.entries(data))if(!['track','_track','duration','_duration','easing','_easing'].includes(key))pathData[key]=value;
   if(type==='AnimateTrack'){const duration=Number(lookup(data,'duration')||0),end=clock.seconds(beat+duration),repeat=Math.max(0,Number(lookup(data,'repeat')||0));let progress=duration?clamp((seconds-start)/Math.max(.001,end-start),0,repeat+1):1;if(repeat&&progress<repeat+1)progress%=1;else progress=clamp(progress);apply(result,data,easing(lookup(data,'easing')||'easeLinear',progress));}
  }const layer=neutral();apply(layer,pathData,pathProgress);result.position=result.position.map((x,i)=>x+layer.position[i]);result.rotation=result.rotation.map((x,i)=>x+layer.rotation[i]);result.scale=result.scale.map((x,i)=>x*layer.scale[i]);result.dissolve*=layer.dissolve;result.arrow*=layer.arrow;if(layer.color)result.color=layer.color;if(layer.definite)result.definite=layer.definite;if(parentState){const parent=parentState;result.position=result.position.map((x,i)=>x+parent.position[i]);result.rotation=result.rotation.map((x,i)=>x+parent.rotation[i]);result.scale=result.scale.map((x,i)=>x*parent.scale[i]);result.dissolve*=parent.dissolve;result.arrow*=parent.arrow;}return result;
 };
 return {object(object:any,seconds:number,lifeProgress:number):Animation {
  const data=custom(object),tracks=lookup(data,'track'),list=Array.isArray(tracks)?tracks:tracks?[tracks]:[],result=neutral();
  for(const track of list){const layer=trackState(track,seconds,lifeProgress);result.position=result.position.map((x,i)=>x+layer.position[i]);result.rotation=result.rotation.map((x,i)=>x+layer.rotation[i]);result.scale=result.scale.map((x,i)=>x*layer.scale[i]);result.dissolve*=layer.dissolve;result.arrow*=layer.arrow;if(layer.color)result.color=layer.color;if(layer.time!==undefined)result.time=layer.time;if(layer.definite)result.definite=layer.definite;}
  const individual=lookup(data,'animation');if(individual){const layer=neutral();apply(layer,individual,result.time??lifeProgress);result.position=result.position.map((x,i)=>x+layer.position[i]);result.rotation=result.rotation.map((x,i)=>x+layer.rotation[i]);result.scale=result.scale.map((x,i)=>x*layer.scale[i]);result.dissolve*=layer.dissolve;result.arrow*=layer.arrow;if(layer.color)result.color=layer.color;if(layer.time!==undefined)result.time=layer.time;if(layer.definite)result.definite=layer.definite;}
  return result;
 },track:trackState,definitions};
}
export function validateModChart(chart:any):string[] {
 const errors=new Set<string>(),record=(v:any)=>v!==null&&typeof v==='object'&&!Array.isArray(v);
 const fail=(message:string)=>{if(errors.size<20)errors.add(message);};
 if(!record(chart))return ['The chart must be a JSON object.'];
 // Check JSON safety first: Worker results may contain cycles or non-JSON values.
 const ancestors=new Set<any>();let visited=0;
 const walk=(v:any,depth=0)=>{if(++visited>1_000_000){fail('The chart contains too many JSON values.');return;}if(depth>40){fail('JSON nesting is too deep.');return;}if(typeof v==='number'&&!Number.isFinite(v))fail('All numbers must be finite.');if(['bigint','function','symbol','undefined'].includes(typeof v))fail('The chart must contain only JSON values.');if(v&&typeof v==='object'){if(ancestors.has(v)){fail('The chart contains a circular reference.');return;}ancestors.add(v);for(const [k,a] of Object.entries(v)){if(['__proto__','prototype','constructor'].includes(k))fail(`Unsafe object key: ${k}.`);walk(a,depth+1);if(visited>1_000_000)break;}ancestors.delete(v);}};walk(chart);
 if(errors.size)return [...errors];
 try{if(JSON.stringify(chart).length>12_000_000)fail('The script output exceeds the 12 MB chart limit.');}catch{fail('The chart must contain serializable JSON.');}
 if(!String(chart.version||'').startsWith('3')||!Array.isArray(chart.colorNotes))fail('Return a complete V3 chart with version and colorNotes.');
 const numeric=(o:any,fields:string[],path:string)=>{for(const field of fields)if(!Number.isFinite(o[field]))fail(`${path}.${field} must be a finite number.`);};
 const direction=(v:any)=>Number.isInteger(v)&&(v>=0&&v<=8||v>=1000&&v<=1360||v>=2000&&v<=2360);
 for(const key of ['colorNotes','bombNotes','obstacles','sliders','burstSliders','basicBeatmapEvents','rotationEvents','bpmEvents','colorBoostBeatmapEvents','lightColorEventBoxGroups','lightRotationEventBoxGroups','lightTranslationEventBoxGroups']){
  if(chart[key]===undefined)continue;
  if(!Array.isArray(chart[key])){fail(`${key} must be an array.`);continue;}
  for(const [i,o] of chart[key].entries()){
   const path=`${key}[${i}]`;if(!record(o)){fail(`${path} must be an object.`);continue;}
   if(!Number.isFinite(o.b)||o.b<0)fail(`${path} needs a finite, non-negative beat.`);
   if(['colorNotes','bombNotes','obstacles','sliders','burstSliders'].includes(key))numeric(o,['x','y'],path);
   if(['colorNotes','sliders','burstSliders'].includes(key)){if(o.c!==0&&o.c!==1)fail(`${path}.c must be 0 (left/red) or 1 (right/blue).`);if(!direction(o.d))fail(`${path}.d is not a supported cut direction.`);}
   if(['sliders','burstSliders'].includes(key)){numeric(o,['tb','tx','ty'],path);if(o.tb<o.b)fail(`${path} ends before its head.`);if(key==='burstSliders'&&(!Number.isInteger(o.sc)||o.sc<1))fail(`${path}.sc must be a positive slice count.`);}
   if(key==='obstacles'){numeric(o,['d','w','h'],path);const size=custom(o).size??custom(o)._scale;if(o.d<0||o.w<=0&&!(Array.isArray(size)&&size[0]>0)||o.h<=0&&!(Array.isArray(size)&&size[1]>0))fail(`${path} needs non-negative duration and positive native or custom dimensions.`);}
   if(key==='basicBeatmapEvents'){numeric(o,['et','i'],path);if(o.f!==undefined&&(!Number.isFinite(o.f)||o.f<0))fail(`${path}.f must be non-negative brightness.`);}
   if(key==='bpmEvents'&&(!Number.isFinite(o.m)||o.m<=0))fail(`${path}.m must be positive BPM.`);
   if(key==='colorBoostBeatmapEvents'&&typeof o.o!=='boolean')fail(`${path}.o must be a boolean.`);
   if(key.endsWith('EventBoxGroups')&&!Array.isArray(o.e))fail(`${path}.e must be an event-box array.`);
   if(o.customData!==undefined&&!record(o.customData))fail(`${path}.customData must be an object.`);
  }
 }
 for(const key of ['customData','_customData'])if(chart[key]!==undefined&&!record(chart[key]))fail(`${key} must be an object.`);
 const cd=custom(chart);
 for(const key of ['customEvents','_customEvents','environment','_environment','fakeColorNotes','fakeBombNotes','fakeObstacles'])if(cd[key]!==undefined&&!Array.isArray(cd[key]))fail(`${key} must be an array.`);
 for(const key of ['customEvents','_customEvents'])for(const [i,e] of (Array.isArray(cd[key])?cd[key]:[]).entries()){
  if(!record(e)){fail(`${key}[${i}] must be an object.`);continue;}
  const b=e.b??e._time,t=e.t??e._type,d=e.d??e._data;
  if(!Number.isFinite(b)||b<0||typeof t!=='string'||!t||!record(d))fail(`${key}[${i}] requires a non-negative beat, type and data object.`);
  if(record(d)&&((d.duration??d._duration)!==undefined)&&((d.duration??d._duration)<0||!Number.isFinite(d.duration??d._duration)))fail(`${key}[${i}] duration must be non-negative.`);
 }
 return [...errors];
}
