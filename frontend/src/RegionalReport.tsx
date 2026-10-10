import {seconds} from './api';

export default function RegionalReport({chart}:any){
 const rows=chart?.report?.regionReport?.regions||[],focus=chart?.report?.attention?.mappingFocus?.regions||[];
 if(!rows.length&&!focus.length)return null;
 return <div className="regional-generation-report"><h4>Regional generation evidence</h4>{rows.length>0&&<div className="regional-report-scroll"><table><thead><tr><th>Region</th><th>Mapping</th><th>NPS goal</th><th>Actual NPS</th><th>Attention</th></tr></thead><tbody>{rows.map((r:any)=><tr key={r.id}><td><strong>{r.name}</strong><small>{seconds(r.start)}–{seconds(r.end)}</small></td><td>{r.style}</td><td>{r.npsRange?`${r.npsRange[0]}–${r.npsRange[1]}`:Number(r.requestedNps).toFixed(1)}</td><td>{Number(r.actualNps).toFixed(2)}<small>{r.noteCount} notes</small></td><td>{r.analysisMode==='all'?'Full Mix':r.analysisMode==='custom'?`${r.sourcePolicy==='primary'?'Primary: ':'Only: '}${Object.entries(r.sources||{}).filter(([,weight])=>Number(weight)>0).map(([source])=>source).join(', ')}`:'Auto'}</td></tr>)}</tbody></table></div>}
  <p className="subtle">Actual NPS measures the whole region, including rests. Timing, available musical attacks, hand recovery and density ceilings can limit the target.</p>
  {focus.length>0&&<details><summary>Source selection and missing-evidence warnings</summary>{focus.map((r:any)=><div key={`${r.id}-${r.start}`}><strong>{r.name} · {seconds(r.start)}–{seconds(r.end)}</strong><p className="subtle">{r.observedAttacks} observed attacks · {r.eligibleEvents} eligible · {r.selectedEvents} selected. {Object.entries(r.selectedSourceCounts||{}).map(([source,count])=>`${source}: ${count}`).join(' · ')}</p>{r.warnings?.map((warning:string,i:number)=><p className="arc-muted" key={i}>{warning}</p>)}</div>)}</details>}
 </div>;
}
