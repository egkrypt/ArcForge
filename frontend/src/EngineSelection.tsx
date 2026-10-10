/** User-facing engine names stay separate from their API identifiers. */
export const AUDIO_ENGINES=[
 {id:'off',name:'Off',description:'Disable optional musical attention; use the existing DSP timing pipeline.'},
 {id:'legacy',name:'Legacy',description:'Existing DSP timing and rhythmic coverage, without source-attention filtering.'},
 {id:'arcaudio',name:'ArcAudio 2.0',description:'DSP source evidence with phrase context and a competing attention budget.'},
 {id:'hybrid',name:'Hybrid',description:'Legacy rhythmic opportunities combined with ArcAudio source attention.'},
 {id:'arcaudio3',name:'ArcAudio 3.0',description:'Local sound-class recognition combined with measured musical attacks and attention.'},
];
export const PLACEMENT_ENGINES=[
 {id:'local',name:'ArcForge local',description:'Use your locally trained audio-conditioned placement network.'},
 {id:'hybrid',name:'Inferno + ArcForge',description:'Blend upstream InfernoSaber predictions with local learned placement. Reports identify any local fallback.'},
 {id:'inferno',name:'Inferno-guided',description:'Prioritize upstream InfernoSaber predictions, with local adaptation and physical guards.'},
];
export const audioEngineName=(id:string)=>AUDIO_ENGINES.find(e=>e.id===id)?.name||id||'Not recorded';
export const placementEngineName=(id:string)=>id==='inferno+local'?'Inferno + ArcForge':PLACEMENT_ENGINES.find(e=>e.id===id)?.name||id||'Not recorded';

/** A diagnostics refresh must not replace the level's pending analysis choice. */
export function withAnalysisSelection(loaded:any,active:any){
 if(loaded?.id!==active?.id)return loaded;
 const selected:Record<string,unknown>={};
 for(const key of ['arcAudioMode','arcAudioSettings','backend'])if(Object.prototype.hasOwnProperty.call(active?.config||{},key))selected[key]=active.config[key];
 return {...loaded,config:{...loaded.config,...selected}};
}

export function EngineSelection({options,setOptions}:any){
 const audioMode=options.arcAudioMode||'legacy',backend=options.backend||'hybrid';
 return <div className="generation-engine-selection">
  <div className="field-grid"><label className="field"><span>Audio analysis engine</span><select aria-label="Audio analysis engine" value={audioMode} onChange={e=>setOptions((v:any)=>({...v,arcAudioMode:e.target.value}))}>{AUDIO_ENGINES.map(e=><option key={e.id} value={e.id}>{e.name}</option>)}</select><small>{AUDIO_ENGINES.find(e=>e.id===audioMode)?.description}</small></label>
   <label className="field"><span>Note placement engine</span><select aria-label="Note placement engine" value={backend} onChange={e=>setOptions((v:any)=>({...v,backend:e.target.value}))}>{PLACEMENT_ENGINES.map(e=><option key={e.id} value={e.id}>{e.name}</option>)}</select><small>{PLACEMENT_ENGINES.find(e=>e.id===backend)?.description}</small></label></div>
  {['off','legacy'].includes(audioMode)&&<p className="subtle">Off and Legacy intentionally use the same existing timing pipeline. The placement engine chooses note positions and directions independently of audio analysis.</p>}
 </div>;
}

export function ChartEngineOrigin({chart}:any){
 const report=chart?.report||{},selection=report.engineSelection||{};
 const attention=selection.audioAnalysis||{};
 const mode=attention.effectiveMode||attention.mode||report.attention?.mode;
 const requested=attention.requestedMode;
 const legacyKnown=!mode&&Boolean(report.musicAnalysisVersion);
 const placement=selection.placement||{};
 const effective=placement.effectiveBackend||report.backend;
 const savedAttention=report.attention||{},viability=report.attentionViability;
 const detected=Number(savedAttention.detectedEvents),eligible=Number(savedAttention.selectedByAttention),lowRejected=Number(savedAttention.summary?.rejectionReasons?.low_importance||0);
 const hasDensity=Number.isFinite(Number(report.requestedNps))&&report.requestedNps!=null&&Number.isFinite(Number(report.averageNps));
 return <div className="chart-engine-origin" data-chart-engine={mode||(legacyKnown?'legacy':'unknown')}>
  <strong>{chart?.name||'Chart'} · generated chart evidence</strong>
  <p>Audio analysis: {mode?audioEngineName(mode):legacyKnown?'Legacy / Off (historical report)':report.backend==='Manual blank chart'?'Manual blank chart':'Not recorded for this imported or historical chart'}{requested&&requested!==mode?` · requested ${audioEngineName(requested)}`:''}</p>
  <p>Placement: {effective?placementEngineName(effective):'Not recorded'}{placement.requestedBackend?` · requested ${placementEngineName(placement.requestedBackend)}`:''}{placement.fallbackReason?` · ${placement.fallbackReason}`:''}</p>
  {hasDensity&&<p data-testid="chart-density-evidence">Saved generation: requested {Number(report.requestedNps).toFixed(1).replace(/\.0$/,'')} active NPS; generated {Number(report.averageNps).toFixed(2)} average NPS{report.peakFourSecondNps!=null?` · ${Number(report.peakFourSecondNps).toFixed(2)} peak four-second NPS`:''}. NJS {Number(chart.njs).toFixed(1).replace(/\.0$/,'')} m/s is approach speed, separate from note density.</p>}
  {Number.isFinite(detected)&&detected>0&&Number.isFinite(eligible)&&<p data-testid="chart-attention-evidence">Saved attention result: {eligible.toLocaleString()} / {detected.toLocaleString()} detected attacks passed attention{lowRejected?`; ${lowRejected.toLocaleString()} rejected for low importance`:''}.{viability?` Minimum importance ${Math.round(Number(viability.minimumImportance)*100)}%; density ceiling ${Math.round(Number(viability.densityCeiling)*100)}% (at most ${Number(viability.activeNpsCeiling).toFixed(1).replace(/\.0$/,'')} active NPS).`:''} Changing analysis settings applies to future generation and does not change this saved result.</p>}
  {report.generationReportStale&&<p className="subtle">This chart was edited after generation. Its generation report describes the earlier draft.</p>}
 </div>;
}
