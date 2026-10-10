import {useEffect,useState} from 'react';
import {api} from './api';

export const BALANCED_COMPUTE={quality:'balanced',device:'auto',cpuThreads:0};
export const DEEP_COMPUTE={quality:'deep',device:'auto',cpuThreads:0};

export default function AnalysisCompute({value,onChange,performance,cached=false}:any){
 const [runtime,setRuntime]=useState<any>(null),[error,setError]=useState('');
 useEffect(()=>{let active=true;api('/api/analysis/runtime').then(r=>{if(active)setRuntime(r)}).catch((e:any)=>{if(active)setError(e.message)});return()=>{active=false}},[]);
 const config={...BALANCED_COMPUTE,...value},cores=runtime?.cpu?.logicalCores||1;
 const change=(patch:any)=>onChange?.({...config,...patch});
 return <div className="analysis-compute" data-testid="analysis-compute">
  <div className="arc-panel-title"><div><h3>Analysis compute</h3><p className="arc-caption">Deep examines four times as many overlapping recognition windows. It uses available CPU threads and larger GPU batches while keeping observed attack timing.</p></div></div>
  <div className="analysis-compute-fields">
   <label className="field"><span>Analysis quality</span><select aria-label="Analysis quality" value={config.quality} onChange={e=>change({quality:e.target.value})}><option value="balanced">Balanced · standard temporal coverage</option><option value="deep">Deep · 4× temporal coverage</option></select></label>
   <label className="field"><span>Analysis device</span><select aria-label="Analysis device" value={config.device} onChange={e=>change({device:e.target.value})}><option value="auto">Auto · GPU when available</option><option value="cpu">CPU</option><option value="cuda" disabled={runtime&&!runtime.gpu?.available}>NVIDIA GPU · CUDA{runtime&&!runtime.gpu?.available?' unavailable':''}</option></select></label>
   <label className="field"><span>CPU threads</span><select aria-label="Analysis CPU threads" value={config.cpuThreads} onChange={e=>change({cpuThreads:Number(e.target.value)})}><option value={0}>{config.quality==='deep'?'Automatic · all available threads':'Automatic · balanced policy'}</option>{Array.from({length:cores},(_,i)=><option key={i+1} value={i+1}>{i+1}</option>)}</select></label>
  </div>
  {runtime&&<p className="arc-caption">{cores} CPU threads · {runtime.gpu?.available?`${runtime.gpu.name} · ${(runtime.gpu.memoryBytes/1024**3).toFixed(1)} GB VRAM`:'CUDA unavailable; CPU analysis supported'}. GPU batches reserve memory for the editor and other models. Deep overlaps CPU source-feature analysis with CUDA inference.</p>}
  {error&&<p className="arc-caption">Hardware status unavailable: {error}</p>}
  {performance&&<p className="arc-caption" data-testid="analysis-compute-result">{cached||performance.cacheHit?'Cached measurements':'Measured'}: {performance.device||'local'} · {performance.patchCount?.toLocaleString()||'—'} windows{performance.hopSeconds!=null?` · ${performance.hopSeconds.toFixed(2)} s step`:''}{performance.cpuThreads!=null?` · ${performance.cpuThreads} CPU threads`:''}{performance.batchSize!=null?` · batch ${performance.batchSize}`:''}{performance.inferenceSeconds!=null?` · ${performance.inferenceSeconds.toFixed(2)} s inference`:''}.</p>}
  <p className="arc-caption">More temporal evidence can help short passages. Instrument accuracy still depends on the trained mixed-audio model; Deep does not separate stems or invent labels.</p>
 </div>;
}
