export async function api(path:string, options:RequestInit={}) {const r=await fetch(path, options); let data:any; const type=r.headers.get('content-type')||''; if(type.includes('json')) data=await r.json(); else data=await r.text(); if(!r.ok) throw new Error(data.detail||data.error||data||`Request failed (${r.status})`); return data;}
export function post(path:string,data:any){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});}
export function put(path:string,data:any){return api(path,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});}
export const seconds=(n:number=0)=>`${Math.floor(n/60)}:${String(Math.floor(n%60)).padStart(2,'0')}`;
export const arrows=['↑','↓','←','→','↖','↗','↙','↘','•'];
export function chartTime(beat:number,bpm:number,events:any[]=[]) {if(!events.length)return beat*60/bpm;let time=0,last=0,rate=bpm;for(const e of [...events].sort((a,b)=>a.b-b.b)){if(e.b>beat)break;time+=(e.b-last)*60/rate;last=e.b;rate=e.m;}return time+(beat-last)*60/rate;}
export function timeBeat(sec:number,bpm:number,events:any[]=[]) {if(!events.length)return sec*bpm/60;let time=0,last=0,rate=bpm;for(const e of [...events].sort((a,b)=>a.b-b.b)){const t=(e.b-last)*60/rate;if(time+t>sec)break;time+=t;last=e.b;rate=e.m;}return last+(sec-time)*rate/60;}
