import {useEffect, useMemo, useRef, useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {createBeatClock, halfJumpBeats} from '../chromapper';
import {createModRuntime,custom,cutRotation,isDot,MOD_CAPABILITIES} from '../modRuntime';
import {arcViewerLimitations,createNativeNjsSampler,createLaneRotationSampler,createLocalBpmSampler,nativeJumpParameters} from '../nativePreview';

export type ViewerProps = {chart:any,audioUrl:string,bpm:number,njs:number,environment:string,time:number,playing:boolean,projectId?:string,project?:any,chartId?:string,audioOffset?:number,jumpOffset?:number,characteristic?:string,difficulty?:string,ghost?:any,onGridHover?:(p:{x:number,y:number}|null)=>void,onGridPlace?:(x:number,y:number)=>void,onNoteSelect?:(index:number)=>void,selectedNote?:any,editEnabled?:boolean,zoomScale?:number,savedRevision?:number,dirty?:boolean,onSaveReload?:()=>void,palette?:{hand:number,direction:number,tool:string,beat:number}};
const rotations = [0,180,90,-90,45,-45,135,-135,0];
const rgb = (value:any, fallback:string) => Array.isArray(value) ? new THREE.Color(value[0],value[1],value[2]) : new THREE.Color(fallback);
const colorOf = (n:any) => rgb(n.customData?.color ?? n.customData?._color, n.c === 0 ? '#fb3868' : '#21bfff');
const xyz = (n:any) => {
  const p = n.customData?.coordinates ?? n.customData?._position;
  const x = Array.isArray(p) ? p[0]+1.5 : Math.abs(n.x)>=1000 ? (n.x-Math.sign(n.x)*1000)/1000 : n.x;
  const y = Array.isArray(p) ? p[1] : Math.abs(n.y)>=1000 ? (n.y-Math.sign(n.y)*1000)/1000 : n.y;
  return [(x-1.5)*.6,.75+y*.6];
};

function normalize(c:any) {
  if (!c) return {colorNotes:[],bombNotes:[],obstacles:[],sliders:[],burstSliders:[],basicBeatmapEvents:[],bpmEvents:[],customData:{}};
  if (!c._notes) return c;
  return {
    ...c,colorNotes:c._notes.filter((n:any)=>n._type<2).map((n:any)=>({b:n._time,x:n._lineIndex,y:n._lineLayer,c:n._type,d:n._cutDirection,a:0,customData:n._customData})),
    bombNotes:c._notes.filter((n:any)=>n._type===3).map((n:any)=>({b:n._time,x:n._lineIndex,y:n._lineLayer,customData:n._customData})),
    obstacles:(c._obstacles||[]).map((o:any)=>({b:o._time,x:o._lineIndex,y:o._type===1?2:0,d:o._duration,w:o._width,h:o._type===1?3:5,customData:o._customData})),
    basicBeatmapEvents:(c._events||[]).map((e:any)=>({b:e._time,et:e._type,i:e._value,f:e._floatValue??1,customData:e._customData})),
    bpmEvents:[],sliders:[],burstSliders:[],customData:c._customData||{}
  };
}

export default function Viewer(props:ViewerProps) {
  const host = useRef<HTMLDivElement>(null), canvasWrap=useRef<HTMLDivElement>(null);
  const latest = useRef(props); latest.current=props;
  const playClock=useRef({time:props.time,wall:performance.now(),playing:props.playing});
  if(playClock.current.time!==props.time||playClock.current.playing!==props.playing)playClock.current={time:props.time,wall:performance.now(),playing:props.playing};
  const reset=useRef<()=>void>(()=>{});
  const [mode,setMode]=useState<'studio'|'arcviewer'|'modstudio'>(props.projectId?'arcviewer':'modstudio');
  const [revision,setRevision]=useState(0),[error,setError]=useState(''),[arcState,setArcState]=useState('Starting environment viewer…'),[arcAvailable,setArcAvailable]=useState<boolean|null>(null);
  const [arcTime,setArcTime]=useState(props.time||0),[previewUrl,setPreviewUrl]=useState(''),[previewBusy,setPreviewBusy]=useState(false),[showSupport,setShowSupport]=useState(false),[autoRefresh,setAutoRefresh]=useState(false);
  const [nativeInfo,setNativeInfo]=useState({njs:props.njs,bpm:props.bpm,warnings:[] as string[]});
  const arcIssues=useMemo(()=>[...new Set([props.chart,...(props.project?.charts||[]).filter((c:any)=>c.id!==props.chartId).map((c:any)=>c.data)].flatMap(arcViewerLimitations))],[props.chart,props.project?.charts,props.chartId]);
  const arcBlocked=arcIssues.length>0;
  useEffect(()=>{if(arcBlocked&&mode==='arcviewer'){setMode('modstudio');setArcState('Native V4 content requires the Mod preview. Export remains available.');}},[arcBlocked,mode]);
  const activePreview=useRef(false),lastPreviewChart=useRef<any>(null);
  const loadDraft=async()=>{if(!props.projectId||activePreview.current||arcBlocked)return;activePreview.current=true;const sourceChart=props.chart;setPreviewBusy(true);try{const r=await fetch(`/api/projects/${encodeURIComponent(props.projectId)}/preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({project:props.project,chartId:props.chartId,chart:props.chart})});const value=await r.json();if(!r.ok)throw new Error(value.detail||'Draft preview failed.');lastPreviewChart.current=sourceChart;setPreviewUrl(new URL(value.zipUrl,location.origin).href);setArcTime(props.time||0);setRevision(v=>v+1);setArcState('Loading unsaved draft · saved map remains unchanged');}catch(e:any){setArcState(e.message);}finally{activePreview.current=false;setPreviewBusy(false)}};
  useEffect(()=>{window.dispatchEvent(new CustomEvent('arcforge:viewer-mode',{detail:mode}));},[mode]);
  useEffect(()=>{const controller=new AbortController();fetch('/api/capabilities',{signal:controller.signal}).then(r=>{if(!r.ok)throw new Error('Viewer availability could not be checked.');return r.json()}).then(value=>{const available=value.preview?.arcViewer!==false;setArcAvailable(available);if(!available){setMode(current=>current==='arcviewer'?'modstudio':current);setArcState('ArcViewer build is not installed. Mod preview and Legacy 3D are available.');}}).catch(error=>{if(error.name==='AbortError')return;setArcAvailable(false);setMode(current=>current==='arcviewer'?'modstudio':current);setArcState('ArcViewer availability could not be checked. Mod preview and Legacy 3D are available.');});return()=>controller.abort()},[]);
  useEffect(()=>{if(!autoRefresh||mode!=='arcviewer'||!props.dirty||previewBusy||lastPreviewChart.current===props.chart)return;const timer=setTimeout(loadDraft,1800);return()=>clearTimeout(timer)},[autoRefresh,mode,props.chart,props.dirty,previewBusy]);
  const [cameraMode,setCameraMode]=useState<'player'|'orbit'>('player'),[editOverlay,setEditOverlay]=useState(true);
  const [quality,setQuality]=useState(()=>{try{return localStorage.getItem('arcforge:viewerQuality')||'auto'}catch{return 'auto'}}),[rendererInfo,setRendererInfo]=useState<any>(null),[showRendererInfo,setShowRendererInfo]=useState(false);
  useEffect(()=>{if(mode==='arcviewer'){setArcTime(props.time||0);setRevision(r=>r+1);setArcState('Reloading the saved revision…');}},[props.savedRevision]);
  useEffect(()=>{setPreviewUrl('');lastPreviewChart.current=null;setArcTime(props.time||0);setRevision(r=>r+1);},[props.projectId,props.chartId,props.project?.revision,props.savedRevision]);

  useEffect(()=>{
    const receive=(e:MessageEvent)=>{
      if(e.origin!==location.origin) return;
      if(e.data?.type==='arcforge:viewer-ready') setArcState('ArcViewer ready · use its playback controls');
      if(e.data?.type==='arcforge:viewer-error') setArcState('ArcViewer could not start. Studio view is available.');
    };
    window.addEventListener('message',receive);return()=>window.removeEventListener('message',receive);
  },[]);

  useEffect(()=>{
    if(!canvasWrap.current || mode==='arcviewer') return;
    const target=canvasWrap.current;
    let renderer:THREE.WebGLRenderer;
    try {renderer=new THREE.WebGLRenderer({antialias:quality!=='performance',alpha:false,powerPreference:quality==='performance'?'low-power':'high-performance'});}
    catch {setError('3D acceleration is unavailable. The editor and ArcViewer remain accessible.');return;}
    setError('');
    const pixelRatio=quality==='performance'?1:Math.min(devicePixelRatio,quality==='high'?2.5:2);
    renderer.setPixelRatio(pixelRatio);
    renderer.outputColorSpace=THREE.SRGBColorSpace;
    target.appendChild(renderer.domElement);
    const gl=renderer.getContext();let gpu=String(gl.getParameter(gl.RENDERER)),vendor=String(gl.getParameter(gl.VENDOR));
    try{const debug=gl.getExtension('WEBGL_debug_renderer_info');if(debug){gpu=String(gl.getParameter(debug.UNMASKED_RENDERER_WEBGL));vendor=String(gl.getParameter(debug.UNMASKED_VENDOR_WEBGL));}}catch{}
    const info={gpu,vendor,api:renderer.capabilities.isWebGL2?'WebGL2':'WebGL',quality,pixelRatio,antialias:gl.getContextAttributes()?.antialias||false,browserControlled:true};setRendererInfo(info);window.dispatchEvent(new CustomEvent('arcforge:renderer',{detail:info}));
    const scene=new THREE.Scene();scene.background=new THREE.Color('#060914');scene.fog=new THREE.FogExp2('#060914',.022);
    const camera=new THREE.PerspectiveCamera(61,1,.05,170);camera.position.set(0,1.6,4.1);
    const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,1.5,-9);controls.enableDamping=true;controls.dampingFactor=.08;controls.maxDistance=40;controls.minDistance=2;controls.maxPolarAngle=Math.PI*.7;controls.enableRotate=cameraMode==='orbit';controls.enablePan=cameraMode==='orbit';controls.enableZoom=false;
    reset.current=()=>{camera.position.set(0,1.6,4.1);controls.target.set(0,1.5,-9);controls.update();};
    scene.add(new THREE.AmbientLight('#b4c8ff',1.45));
    const key=new THREE.DirectionalLight('#ffffff',3);key.position.set(-3,5,4);scene.add(key);
    const floor=new THREE.Mesh(new THREE.PlaneGeometry(140,150),new THREE.MeshStandardMaterial({color:'#080e1b',metalness:.7,roughness:.3}));floor.rotation.x=-Math.PI/2;floor.position.set(0,-.015,-40);scene.add(floor);
    const stage=new THREE.Mesh(new THREE.BoxGeometry(3,.045,3),new THREE.MeshStandardMaterial({color:'#111a2a',metalness:.7,roughness:.32}));stage.position.set(0,.015,.5);scene.add(stage);
    const trackLines:THREE.Mesh[]=[];
    const matLeft=new THREE.MeshBasicMaterial({color:'#ee457e',transparent:true,opacity:.55});
    const matRight=new THREE.MeshBasicMaterial({color:'#36c9ff',transparent:true,opacity:.55});
    [-1.35,1.35].forEach((x,i)=>{const rail=new THREE.Mesh(new THREE.BoxGeometry(.02,.008,125),i?matRight:matLeft);rail.position.set(x,.055,-50);scene.add(rail);trackLines.push(rail);});
    const gridPoints:THREE.Vector3[]=[];for(let i=0;i<=4;i++)gridPoints.push(new THREE.Vector3(-1.2+i*.6,.45,.005),new THREE.Vector3(-1.2+i*.6,2.25,.005));for(let i=0;i<=3;i++)gridPoints.push(new THREE.Vector3(-1.2,.45+i*.6,.005),new THREE.Vector3(1.2,.45+i*.6,.005));
    const grid=new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(gridPoints),new THREE.LineBasicMaterial({color:'#627597',transparent:true,opacity:.48}));scene.add(grid);
    const laserMats:THREE.MeshBasicMaterial[]=[],laserMeshes:THREE.Mesh[]=[],ringMeshes:THREE.Line[]=[];
    const envName=props.environment||'DefaultEnvironment';
    const triangular=/Triangle|Timbaland|Panic|Pyro|TheSecond/.test(envName),mirrored=/Mirror|KDA|Lattice|Kaleidoscope/.test(envName);
    for(let i=0;i<12;i++) {
      const z=-6-i*8, radius=5.2+i*.09,shape:THREE.Vector3[]=[];
      const sides=triangular?3:mirrored?4:8;
      for(let j=0;j<=sides;j++){const angle=(j/sides)*Math.PI*2+Math.PI/2;shape.push(new THREE.Vector3(Math.cos(angle)*radius,Math.sin(angle)*radius+3,z));}
      const mat=new THREE.LineBasicMaterial({color:i%2?'#175789':'#58294f',transparent:true,opacity:.48});
      const ring=new THREE.Line(new THREE.BufferGeometry().setFromPoints(shape),mat);scene.add(ring);ringMeshes.push(ring);
      for(const side of [-1,1]) {
        const material=new THREE.MeshBasicMaterial({color:side<0?'#ee457e':'#36c9ff',transparent:true,opacity:.18,blending:THREE.AdditiveBlending,depthWrite:false});laserMats.push(material);
        const beam=new THREE.Mesh(new THREE.BoxGeometry(.065,10,.065),material);beam.position.set(side*(4+i*.32),4,z);beam.rotation.z=side*.42;beam.userData={side,lightID:i+1,type:side<0?2:3,baseRotation:side*.42};scene.add(beam);laserMeshes.push(beam);
      }
    }
    for(const [type,count] of [[0,4],[4,3]])for(let i=0;i<count;i++){const material=new THREE.MeshBasicMaterial({color:type===0?'#ff3868':'#21bfff',transparent:true,opacity:.1,blending:THREE.AdditiveBlending,depthWrite:false});const beam=new THREE.Mesh(new THREE.BoxGeometry(type===0?8:.08,type===0?.07:5,.06),material);beam.position.set(type===0?0:(i-1)*2,type===0?3+i*.75:3,-15-i*12);beam.userData={side:0,type,lightID:i+1,baseRotation:0};scene.add(beam);laserMats.push(material);laserMeshes.push(beam);}
    const chart=normalize(props.chart),clock=createBeatClock(props.bpm||190,chart.bpmEvents||[]);
    const modRuntime=createModRuntime(chart,clock),njsSampler=createNativeNjsSampler(chart,clock,props.njs||20),laneRotation=createLaneRotationSampler(chart),tempoAt=createLocalBpmSampler(chart,props.bpm||190,clock),laneAxis=new THREE.Vector3(0,1,0);
    const notes=[...(chart.colorNotes||[]),...(mode==='modstudio'?(custom(chart).fakeColorNotes||[]):[])].map((n:any,i:number)=>({...n,seconds:clock.seconds(n.b),_editorIndex:i<(chart.colorNotes||[]).length?i:-1})).sort((a:any,b:any)=>a.seconds-b.seconds);
    const bombs=(chart.bombNotes||[]).map((n:any)=>({...n,seconds:clock.seconds(n.b)}));
    const walls=(chart.obstacles||[]).map((n:any)=>({...n,seconds:clock.seconds(n.b),end:clock.seconds(n.b+n.d)}));
    const events=(chart.basicBeatmapEvents||[]).map((e:any)=>({...e,seconds:clock.seconds(e.b)})).sort((a:any,b:any)=>a.seconds-b.seconds);
    const arrowShape=new THREE.Shape();arrowShape.moveTo(0,.13);arrowShape.lineTo(-.105,-.015);arrowShape.lineTo(-.035,-.015);arrowShape.lineTo(-.035,-.10);arrowShape.lineTo(.035,-.10);arrowShape.lineTo(.035,-.015);arrowShape.lineTo(.105,-.015);arrowShape.closePath();
    const ghostGroup=new THREE.Group(),ghostMaterial=new THREE.MeshBasicMaterial({color:'#e77992',transparent:true,opacity:.32,depthWrite:false});
    ghostGroup.add(new THREE.Mesh(new THREE.BoxGeometry(.45,.45,.45),ghostMaterial));
    const ghostOutline=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(.45,.45,.45)),new THREE.LineBasicMaterial({color:'#f0ddeb',transparent:true,opacity:.85}));ghostGroup.add(ghostOutline);
    const ghostArrow=new THREE.Mesh(new THREE.ShapeGeometry(arrowShape),new THREE.MeshBasicMaterial({color:'#fff7ef',side:THREE.DoubleSide,transparent:true,opacity:.8}));ghostArrow.position.z=.235;ghostGroup.add(ghostArrow);
    const ghostDot=new THREE.Mesh(new THREE.CircleGeometry(.04,12),new THREE.MeshBasicMaterial({color:'#fff7ef',side:THREE.DoubleSide}));ghostDot.position.z=.235;ghostGroup.add(ghostDot);ghostGroup.visible=false;scene.add(ghostGroup);
    const selectedOutline=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(.49,.49,.49)),new THREE.LineBasicMaterial({color:'#fff2b6',transparent:true,opacity:.95}));selectedOutline.visible=false;scene.add(selectedOutline);
    const capacity=2048;
    const blocks=new THREE.InstancedMesh(new THREE.BoxGeometry(.42,.42,.42),new THREE.MeshStandardMaterial({roughness:.28,metalness:.26}),capacity);
    const arrows=new THREE.InstancedMesh(new THREE.ShapeGeometry(arrowShape),new THREE.MeshBasicMaterial({color:'#fff8f3',side:THREE.DoubleSide}),capacity);
    const dots=new THREE.InstancedMesh(new THREE.SphereGeometry(.038,10,8),new THREE.MeshBasicMaterial({color:'#fff8f3'}),capacity);
    const bombMesh=new THREE.InstancedMesh(new THREE.IcosahedronGeometry(.22,1),new THREE.MeshStandardMaterial({color:'#181926',emissive:'#91233f',emissiveIntensity:.4,metalness:.6,roughness:.2}),capacity);
    for(const mesh of [blocks,arrows,dots,bombMesh]){mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);mesh.frustumCulled=false;scene.add(mesh);}
    const opacityAttributes=new Map<THREE.InstancedMesh,THREE.InstancedBufferAttribute>();
    const enableOpacity=(mesh:THREE.InstancedMesh)=>{const attribute=new THREE.InstancedBufferAttribute(new Float32Array(capacity).fill(1),1);attribute.setUsage(THREE.DynamicDrawUsage);mesh.geometry.setAttribute('instanceOpacity',attribute);opacityAttributes.set(mesh,attribute);const material=mesh.material as THREE.Material;material.transparent=true;material.onBeforeCompile=(shader:any)=>{shader.vertexShader=shader.vertexShader.replace('#include <common>','#include <common>\nattribute float instanceOpacity; varying float vInstanceOpacity;').replace('#include <begin_vertex>','#include <begin_vertex>\nvInstanceOpacity = instanceOpacity;');shader.fragmentShader=shader.fragmentShader.replace('#include <common>','#include <common>\nvarying float vInstanceOpacity;').replace('#include <color_fragment>','#include <color_fragment>\ndiffuseColor.a *= vInstanceOpacity;');};};
    for(const mesh of [blocks,arrows,dots,bombMesh])enableOpacity(mesh);
    const visibleNotes:any[]=[],ray=new THREE.Raycaster(),pointer=new THREE.Vector2(),gridPlane=new THREE.Plane(new THREE.Vector3(0,0,1),0),intersection=new THREE.Vector3();let downPoint:any=null;
    const pointGrid=(e:PointerEvent)=>{const r=renderer.domElement.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);ray.setFromCamera(pointer,camera);if(!ray.ray.intersectPlane(gridPlane,intersection))return null;const x=Math.floor((intersection.x+1.2)/.6),y=Math.floor((intersection.y-.45)/.6);return x>=0&&x<=3&&y>=0&&y<=2?{x,y}:null;};
    const onMove=(e:PointerEvent)=>{if(cameraMode==='orbit'&&e.buttons)return;latest.current.onGridHover?.(pointGrid(e));};
    const onLeave=()=>latest.current.onGridHover?.(null);
    const onDown=(e:PointerEvent)=>{downPoint={x:e.clientX,y:e.clientY};};
    const onUp=(e:PointerEvent)=>{if(!downPoint||Math.hypot(e.clientX-downPoint.x,e.clientY-downPoint.y)>5)return;const p=pointGrid(e);const hit=ray.intersectObject(blocks,false)[0];if(hit?.instanceId!==undefined){const n=visibleNotes[hit.instanceId];if(n&&(!latest.current.editEnabled||Math.abs(n.seconds-(latest.current.time-(latest.current.audioOffset||0)))<.04)){if(n._editorIndex>=0)latest.current.onNoteSelect?.(n._editorIndex);return;}}if(p&&latest.current.editEnabled)latest.current.onGridPlace?.(p.x,p.y);};
    renderer.domElement.addEventListener('pointermove',onMove);renderer.domElement.addEventListener('pointerleave',onLeave);renderer.domElement.addEventListener('pointerdown',onDown);renderer.domElement.addEventListener('pointerup',onUp);
    const wallMeshes=walls.map((w:any)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshBasicMaterial({color:rgb(w.customData?.color??w.customData?._color,'#a878fa'),transparent:true,opacity:.19,wireframe:true}));m.visible=false;scene.add(m);return m;});
    const arcMeshes=(chart.sliders||[]).map((a:any)=>{const line=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:colorOf(a),transparent:true,opacity:.7}));line.visible=false;scene.add(line);return {a,line};});
    const chains=(chart.burstSliders||[]).flatMap((a:any)=>Array.from({length:Math.min(128,a.sc||1)},(_,i)=>({b:a.b+(a.tb-a.b)*i/Math.max(1,a.sc-1),x:a.x+(a.tx-a.x)*i/Math.max(1,a.sc-1),y:a.y+(a.ty-a.y)*i/Math.max(1,a.sc-1),c:a.c,d:a.d,r:laneRotation(a)+(laneRotation(a,'tail')-laneRotation(a))*i/Math.max(1,a.sc-1),customData:a.customData,seconds:clock.seconds(a.b+(a.tb-a.b)*i/Math.max(1,a.sc-1))})));
    const chainMesh=new THREE.InstancedMesh(new THREE.BoxGeometry(.38,.07,.15),new THREE.MeshStandardMaterial({roughness:.3}),capacity);chainMesh.frustumCulled=false;scene.add(chainMesh);enableOpacity(chainMesh);
    const enhanced:{group:THREE.Group,base:any,track:any}[]=[];
    for(const env of (chart.customData?.environment||chart.customData?._environment||[]).slice(0,300)) {
      if(!env.geometry && !env._geometry) continue;
      const geometry=env.geometry||env._geometry,type=geometry.type||geometry._type;
      const g=/Sphere/.test(type)?new THREE.SphereGeometry(.5,14,10):/Cylinder/.test(type)?new THREE.CylinderGeometry(.5,.5,1,12):new THREE.BoxGeometry(1,1,1);
      const group=new THREE.Group(),mesh=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:rgb(geometry.material?.color??geometry._material?._color,'#46d4f2'),transparent:true,opacity:.55,wireframe:false}));
      group.add(mesh);const p=env.position||env._position||[0,0,0],s=env.scale||env._scale||[1,1,1],r=env.rotation||env._rotation||[0,0,0];
      group.position.set(p[0],p[1],-p[2]);group.scale.set(s[0],s[1],s[2]);group.rotation.set(r[0]*Math.PI/180,-r[1]*Math.PI/180,-r[2]*Math.PI/180);scene.add(group);
      enhanced.push({group,base:{p,s,r},track:env.track||env._track});
    }
    const lowerBound=(items:any[],seconds:number)=>{let low=0,high=items.length;while(low<high){const mid=(low+high)>>>1;if(items[mid].seconds<seconds)low=mid+1;else high=mid;}return low;};
    const eventGroups=new Map<number,any[]>();for(const event of events){if(!eventGroups.has(event.et))eventGroups.set(event.et,[]);eventGroups.get(event.et)!.push(event);}
    const object=new THREE.Object3D();let frame=0,lastNativeUpdate=0;
    const pose=(n:any,dt:number,hjd:number,speed:number,t:number)=>{
      const [x,y]=xyz(n),cd=custom(n),atCursor=!latest.current.playing&&Math.abs(n.b-(latest.current.palette?.beat??-10))<.001;
      object.position.set(x,y,atCursor?0:-dt*speed);object.rotation.set(0,0,0);object.scale.setScalar(1);
      let alpha=1,arrow=1,color=colorOf(n);
      if(mode==='modstudio'){
        const life=Math.max(0,Math.min(1,.5-dt/(2*hjd))),state=modRuntime.object(n,t,life);
        const lr=cd.localRotation||cd._localRotation||[0,0,0],wr=cd.worldRotation||cd._rotation||[0,0,0],world=typeof wr==='number'?[0,wr,0]:wr;
        object.position.add(new THREE.Vector3(state.position[0]*.6,state.position[1]*.6,-state.position[2]*.6));
        if(state.definite)object.position.set(state.definite[0]*.6,.75+state.definite[1]*.6,-state.definite[2]*.6);
        if(state.time!==undefined&&!atCursor)object.position.z=(state.time-.5)*2*hjd*speed;
        object.rotation.set((lr[0]+state.rotation[0])*Math.PI/180,-(lr[1]+state.rotation[1])*Math.PI/180,-(lr[2]+state.rotation[2])*Math.PI/180);
        if(world.some((v:number)=>v)){const e=new THREE.Euler(world[0]*Math.PI/180,-world[1]*Math.PI/180,-world[2]*Math.PI/180);object.position.applyEuler(e);object.rotation.x+=e.x;object.rotation.y+=e.y;object.rotation.z+=e.z;}
        object.scale.set(...state.scale.map((v:number)=>Math.max(0,Math.min(50,v))) as [number,number,number]);
        alpha=state.dissolve;arrow=state.arrow;if(state.color)color=rgb(state.color,'#ffffff');
        if(cd.scale||cd._scale){const sc=cd.scale||cd._scale;object.scale.multiply(new THREE.Vector3(...sc));}
        // The paused cursor remains on the editable frontal grid. Away from
        // it, native lanes rotate objects around the player origin.
        if(!atCursor){const yaw=-laneRotation(n)*Math.PI/180;object.position.applyAxisAngle(laneAxis,yaw);object.rotation.y+=yaw;}
      }
      return{color,arrow,alpha};
    };
    const size=()=>{const w=target.clientWidth,h=target.clientHeight;renderer.setSize(w,h,false);camera.aspect=w/Math.max(1,h);camera.updateProjectionMatrix();};
    const observer=new ResizeObserver(size);observer.observe(target);size();
    const animate=()=>{
      frame=requestAnimationFrame(animate);
      if(!target.clientWidth||!target.clientHeight)return;
      const elapsed=latest.current.playing?Math.min(.4,(performance.now()-playClock.current.wall)/1000):0;
      const t=(latest.current.time||0)+elapsed-(latest.current.audioOffset||0),baseSpeed=latest.current.njs||20,localBpm=tempoAt(t);
      const speed=mode==='modstudio'?njsSampler.speedAt(t):baseSpeed;
      const baseHjd=halfJumpBeats(baseSpeed,latest.current.jumpOffset??chart.customData?.noteJumpStartBeatOffset??0,localBpm)*60/localBpm;
      const hjd=mode==='modstudio'&&njsSampler.hasEvents?nativeJumpParameters(baseSpeed,2*baseSpeed*baseHjd,speed).reactionSeconds:baseHjd;
      if(performance.now()-lastNativeUpdate>250){lastNativeUpdate=performance.now();setNativeInfo({njs:speed,bpm:localBpm,warnings:njsSampler.warnings});}
      const fov=Math.max(32,Math.min(85,61/Math.sqrt(latest.current.zoomScale||1)));if(Math.abs(camera.fov-fov)>.01){camera.fov=fov;camera.updateProjectionMatrix();}
      const ghost=latest.current.ghost;ghostGroup.visible=!!ghost&&latest.current.editEnabled&&ghost.tool!=='arc';if(ghostGroup.visible){const [gx,gy]=xyz(ghost);ghostGroup.position.set(gx,gy,0);ghostMaterial.color.copy(colorOf(ghost));ghostArrow.rotation.z=rotations[ghost.d||0]*Math.PI/180;ghostArrow.visible=ghost.d!==8&&ghost.tool!=='bomb';ghostDot.visible=ghost.d===8||ghost.tool==='bomb';}
      const selected=latest.current.selectedNote;selectedOutline.visible=!!selected;if(selected){pose(selected,clock.seconds(selected.b)-t,hjd,speed,t);selectedOutline.position.copy(object.position);selectedOutline.rotation.copy(object.rotation);selectedOutline.scale.copy(object.scale);}
      let count=0,arrowCount=0,dotCount=0,bombCount=0,chainCount=0;
      visibleNotes.length=0;
      for(let noteIndex=lowerBound(notes,t-.12);noteIndex<notes.length;noteIndex++) {
        const n=notes[noteIndex],dt=n.seconds-t;if(dt>hjd+.35)break;if(count>=capacity)break;
        const explicitSpeed=Number(custom(n).noteJumpMovementSpeed||0),objectSpeed=mode==='modstudio'&&explicitSpeed>0?explicitSpeed:speed;
        const objectHjd=mode==='modstudio'&&explicitSpeed>0?halfJumpBeats(objectSpeed,custom(n).noteJumpStartBeatOffset??latest.current.jumpOffset??0,tempoAt(n.seconds))*60/tempoAt(n.seconds):hjd;
        const state=pose(n,dt,objectHjd,objectSpeed,t);object.updateMatrix();blocks.setMatrixAt(count,object.matrix);blocks.setColorAt(count,state.color);opacityAttributes.get(blocks)!.setX(count,state.alpha);visibleNotes.push(n);count++;
        object.translateZ(.218);object.rotateZ(cutRotation(n.d,n.a||0)*Math.PI/180);object.updateMatrix();
        if(isDot(n.d)){opacityAttributes.get(dots)!.setX(dotCount,state.alpha*state.arrow);dots.setMatrixAt(dotCount++,object.matrix);}else{opacityAttributes.get(arrows)!.setX(arrowCount,state.alpha*state.arrow);arrows.setMatrixAt(arrowCount++,object.matrix);}
      }
      for(const n of bombs) {const dt=n.seconds-t;if(dt<-.12||dt>hjd+.35||bombCount>=capacity)continue;const state=pose(n,dt,hjd,speed,t);object.updateMatrix();opacityAttributes.get(bombMesh)!.setX(bombCount,state.alpha);bombMesh.setMatrixAt(bombCount++,object.matrix);}
      for(const n of chains) {const dt=n.seconds-t;if(dt<-.12||dt>hjd+.35||chainCount>=capacity)continue;const state=pose(n,dt,hjd,speed,t);object.rotateZ(cutRotation(n.d)*Math.PI/180);object.updateMatrix();chainMesh.setMatrixAt(chainCount,object.matrix);chainMesh.setColorAt(chainCount,state.color);opacityAttributes.get(chainMesh)!.setX(chainCount++,state.alpha);}
      for(const [mesh,n] of [[blocks,count],[arrows,arrowCount],[dots,dotCount],[bombMesh,bombCount],[chainMesh,chainCount]] as [THREE.InstancedMesh,number][]){mesh.count=n;mesh.instanceMatrix.needsUpdate=true;if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true;opacityAttributes.get(mesh)!.needsUpdate=true;}
      walls.forEach((w:any,i:number)=>{const m=wallMeshes[i],duration=w.end-w.seconds;m.visible=w.end>=t-.1&&w.seconds<=t+2;if(!m.visible)return;const size=w.customData?.size||w.customData?._scale;const coord=w.customData?.coordinates||w.customData?._position;const width=(size?.[0]??w.w)*.6,height=(size?.[1]??w.h)*.6,depth=Math.max(.04,(size?.[2]??duration*speed));m.scale.set(width,height,depth);m.position.set(coord?.[0]!==undefined?coord[0]*.6:((w.x-2)*.6+width/2),coord?.[1]!==undefined?.45+coord[1]*.6+height/2:.45+w.y*.6+height/2,-(w.seconds-t)*speed-depth/2);m.rotation.set(0,0,0);if(mode==='modstudio'){const state=modRuntime.object(w,t,Math.max(0,Math.min(1,.5-(w.seconds-t)/(2*hjd)))),cd=custom(w),lr=cd.localRotation||[0,0,0];m.position.add(new THREE.Vector3(state.position[0]*.6,state.position[1]*.6,-state.position[2]*.6));m.scale.multiply(new THREE.Vector3(...state.scale as [number,number,number]));m.rotation.set((lr[0]+state.rotation[0])*Math.PI/180,-(lr[1]+state.rotation[1])*Math.PI/180,-(lr[2]+state.rotation[2])*Math.PI/180);(m.material as THREE.MeshBasicMaterial).opacity=.25*state.dissolve;if(state.color)(m.material as THREE.MeshBasicMaterial).color.copy(rgb(state.color,'#ffffff'));}});
      if(mode==='modstudio')wallMeshes.forEach((mesh,i)=>{if(!mesh.visible)return;const yaw=-laneRotation(walls[i])*Math.PI/180;mesh.position.applyAxisAngle(laneAxis,yaw);mesh.rotation.y+=yaw;});
      arcMeshes.forEach(({a,line}:any)=>{
        const st=clock.seconds(a.b),et=clock.seconds(a.tb);line.visible=et>=t-.1&&st<=t+hjd+.35;if(!line.visible)return;
        const p=xyz(a),q=xyz({x:a.tx,y:a.ty}),start=new THREE.Vector3(p[0],p[1],-(st-t)*speed),end=new THREE.Vector3(q[0],q[1],-(et-t)*speed);
        if(mode==='modstudio'){start.applyAxisAngle(laneAxis,-laneRotation(a)*Math.PI/180);end.applyAxisAngle(laneAxis,-laneRotation(a,'tail')*Math.PI/180);}
        const midpoint=start.clone().lerp(end,.5);midpoint.y+=.28;const curve=new THREE.QuadraticBezierCurve3(start,midpoint,end);line.geometry.setFromPoints(curve.getPoints(24));
      });
      const latestEvent=(type:number,id?:number)=>{const group=eventGroups.get(type)||[];let index=lowerBound(group,t+.010001)-1;while(index>=0){const event=group[index--],ids=custom(event).lightID??custom(event)._lightID;if(id===undefined||ids===undefined||(Array.isArray(ids)?ids.includes(id):ids===id))return event;}return undefined;};
      const cue=(e:any)=>{const brightness=e?.f??1,value=e?.i??0,age=e?t-e.seconds:0,modeValue=(value-1)%4;const pulse=!e?.i?.valueOf()?0:value===0?0:modeValue===1?Math.max(.2,Math.exp(-age*5)):modeValue===2?Math.max(0,1-age*1.4):1;const color=e?rgb(custom(e).color??custom(e)._color,value>=9?'#ffffff':value>=5?'#ff3868':'#21bfff'):new THREE.Color('#21bfff');return{pulse:Math.min(1,brightness*pulse),color};};
      laserMeshes.forEach((mesh,i)=>{const side=mesh.userData.side,event=latestEvent(mesh.userData.type,mesh.userData.lightID),light=cue(event);laserMats[i].opacity=.03+light.pulse*.65;laserMats[i].color.copy(light.color);const rotation=side?latestEvent(side<0?12:13):undefined,cd=custom(rotation),spin=Number(cd.speed??rotation?.i??0)*(cd.direction===0?-1:1);mesh.rotation.z=mesh.userData.baseRotation+(rotation?(t-rotation.seconds)*spin*.22:0);});
      const ringLight=cue(latestEvent(1)),ringRotate=latestEvent(8),ringZoom=latestEvent(9),rcd=custom(ringRotate),zcd=custom(ringZoom);
      ringMeshes.forEach((ring,i)=>{const material=ring.material as THREE.LineBasicMaterial;material.color.copy(ringLight.color);material.opacity=.12+.7*ringLight.pulse;if(ringRotate){const progress=Math.min(1,Math.max(0,(t-ringRotate.seconds-i*.025)*Number(rcd.speed??1)));ring.rotation.z=(Number(rcd.rotation??45)+i*Number(rcd.step??8))*Math.PI/180*progress*(rcd.direction===0?-1:1);}if(ringZoom)ring.position.z=Number(zcd.step??1)*i*Math.min(1,(t-ringZoom.seconds)*Number(zcd.speed??1));});
      trackLines.forEach((mesh:any,i)=>{const light=cue(latestEvent(i?3:2));mesh.material.color.copy(light.color);mesh.material.opacity=.3+.5*light.pulse;});
      for(const item of enhanced){const state=mode==='modstudio'&&item.track?modRuntime.track(item.track,t,.5):null;const p=item.base.p,s=item.base.s,r=item.base.r;item.group.position.set(p[0]+(state?.position[0]||0),p[1]+(state?.position[1]||0),-p[2]-(state?.position[2]||0));item.group.scale.set(s[0]*(state?.scale[0]??1),s[1]*(state?.scale[1]??1),s[2]*(state?.scale[2]??1));item.group.rotation.set((r[0]+(state?.rotation[0]||0))*Math.PI/180,-(r[1]+(state?.rotation[1]||0))*Math.PI/180,-(r[2]+(state?.rotation[2]||0))*Math.PI/180);item.group.traverse((o:any)=>{if(o.material){o.material.opacity=.55*(state?.dissolve??1);if(state?.color)o.material.color.copy(rgb(state.color,'#ffffff'))}});}
      controls.update();renderer.render(scene,camera);
    };
    animate();
    return()=>{cancelAnimationFrame(frame);observer.disconnect();controls.dispose();renderer.domElement.removeEventListener('pointermove',onMove);renderer.domElement.removeEventListener('pointerleave',onLeave);renderer.domElement.removeEventListener('pointerdown',onDown);renderer.domElement.removeEventListener('pointerup',onUp);scene.traverse((o:any)=>{o.geometry?.dispose();if(Array.isArray(o.material))o.material.forEach((m:any)=>m.dispose());else o.material?.dispose();});renderer.dispose();renderer.domElement.remove();};
  },[props.chart,props.bpm,props.njs,props.environment,mode,cameraMode,quality]);

  const changeMode=(next:'studio'|'arcviewer'|'modstudio')=>{if(next==='arcviewer'&&(arcAvailable!==true||arcBlocked))return;setMode(next);if(next==='arcviewer'){setArcTime(props.time||0);setArcState('Starting environment viewer…');}};
  const mapUrl=previewUrl||(props.projectId?`${location.origin}/api/projects/${encodeURIComponent(props.projectId)}/export?version=3`:'');
  const arcUrl=`/viewer/?${new URLSearchParams({...(mapUrl?{url:mapUrl,noProxy:'true',t:String(arcTime),...(props.characteristic?{mode:props.characteristic}:{}),...(props.difficulty?{difficulty:props.difficulty}:{})}:{}),arcforgeRevision:String(revision)})}`;
  const glyphs=['↑','↓','←','→','↖','↗','↙','↘','•'];
  return <div ref={host} className="arcforge-edit-viewer" style={{position:'relative',height:'100%',minHeight:320,background:'#060914',borderRadius:0,overflow:'hidden'}}>
    <div style={{position:'absolute',inset:0,display:mode!=='arcviewer'?'block':'none'}} ref={canvasWrap}/>
    {mode==='arcviewer'&&arcAvailable===true&&<iframe title="ArcViewer environment player" src={arcUrl} allow="autoplay; fullscreen" style={{position:'absolute',inset:'48px 0 0',width:'100%',height:'calc(100% - 48px)',border:0}}/>}
    <div style={{position:'absolute',top:12,left:12,right:12,display:'flex',gap:6,alignItems:'center',zIndex:3}}>
      <button onClick={()=>changeMode('modstudio')} className={mode==='modstudio'?'primary':''} style={{fontSize:11,padding:'6px 10px'}}>Mod preview</button><button onClick={()=>changeMode('studio')} className={mode==='studio'?'primary':''} style={{fontSize:10,padding:'6px 8px'}}>Legacy 3D</button>
      <button disabled={arcAvailable!==true||arcBlocked} title={arcBlocked?`This map contains ${arcIssues.join(', ')}. Use Mod preview; export as V4 to preserve these features.`:arcAvailable===false?'Optional ArcViewer build is not installed; use Mod preview or Legacy 3D.':arcAvailable===null?'Checking optional ArcViewer build…':'Reconstructed game environments'} onClick={()=>changeMode('arcviewer')} className={mode==='arcviewer'?'primary':''} style={{fontSize:11,padding:'6px 10px'}}>ArcViewer environments</button>
      <div style={{flex:1}}/>
      {mode!=='arcviewer'?<><button onClick={()=>setCameraMode(cameraMode==='player'?'orbit':'player')} style={{fontSize:10,padding:'6px 8px'}}>{cameraMode==='player'?'Player camera':'Orbit camera'}</button><select aria-label="Viewer quality" value={quality} onChange={e=>{setQuality(e.target.value);try{localStorage.setItem('arcforge:viewerQuality',e.target.value)}catch{}}} style={{fontSize:9,padding:'4px 6px',maxWidth:96}}><option value="auto">Auto quality</option><option value="high">High quality</option><option value="performance">Performance</option></select><button title="Actual browser renderer information" onClick={()=>setShowRendererInfo(!showRendererInfo)} style={{fontSize:10,padding:'6px 8px'}}>GPU</button><button onClick={()=>reset.current()} style={{fontSize:10,padding:'6px 8px'}}>Reset</button></>:<><button onClick={()=>setEditOverlay(!editOverlay)} style={{fontSize:10,padding:'6px 8px'}}>{editOverlay?'Hide edit grid':'Edit grid'}</button><button disabled={previewBusy} onClick={loadDraft} style={{fontSize:10,padding:'6px 8px'}}>{previewBusy?'Preparing draft…':'Preview current draft'}</button><button aria-pressed={autoRefresh} title="Refresh unsaved snapshot after 1.8 seconds idle; off by default" onClick={()=>setAutoRefresh(!autoRefresh)} style={{fontSize:10,padding:'6px 8px'}}>{autoRefresh?'Auto-refresh on':'Auto-refresh off'}</button></>}
      <button style={{fontSize:10,padding:'6px 8px'}} onClick={()=>setShowSupport(!showSupport)}>Support</button><button aria-label="Fullscreen viewer" onClick={()=>host.current?.requestFullscreen()} style={{fontSize:13,padding:'6px 10px'}}>⛶</button>
    </div>
    {mode!=='arcviewer'&&<><div className="live-view-caption"><span className="left-hand-caption">LEFT / RED</span><span>{cameraMode==='player'?'CLICK GRID TO PLACE · CTRL+WHEEL ZOOMS':'DRAG TO ORBIT · SWITCH TO PLAYER FOR FRONTAL EDITING'}</span><span className="right-hand-caption">RIGHT / BLUE</span></div></>}
    {mode!=='arcviewer'&&showRendererInfo&&rendererInfo&&<div className="viewer-renderer-info"><strong>ACTUAL WEBGL RENDERER</strong><span>{rendererInfo.gpu}</span><small>{rendererInfo.api} · {rendererInfo.pixelRatio}× pixel ratio · Antialias {rendererInfo.antialias?'on':'off'}</small><p>The browser selects drawing hardware. Training hardware is managed separately.</p></div>}
    {mode==='arcviewer'&&editOverlay&&props.onGridPlace&&<div className="arc-edit-overlay"><div><strong>LIVE EDIT GRID</strong><span>Beat {(props.palette?.beat||0).toFixed(3)}</span></div><div className="arc-edit-cells">{[2,1,0].flatMap(y=>[0,1,2,3].map(x=>{const at=(props.chart?.colorNotes||[]).find((n:any)=>Math.abs(n.b-(props.palette?.beat||0))<.001&&n.x===x&&n.y===y);return <button key={`${x}:${y}`} title={`Place ${props.palette?.hand===0?'left red':'right blue'} at column ${x}, row ${y}`} className={`${props.palette?.hand===0?'red':'blue'} ${at?'occupied':''}`} onPointerEnter={()=>props.onGridHover?.({x,y})} onPointerLeave={()=>props.onGridHover?.(null)} onClick={()=>props.onGridPlace?.(x,y)}>{at?<span className={at.c===0?'red':'blue'}>{glyphs[at.d]}</span>:<span className="arc-cell-ghost">{props.ghost?.x===x&&props.ghost?.y===y?glyphs[props.palette?.direction??1]:'+'}</span>}<small>{x},{y}</small></button>;}))}</div><p>Edits use the studio cursor. Preview current draft loads unsaved changes without changing the saved map.</p>{props.dirty&&<button disabled={previewBusy} className="arc-edit-sync" onClick={loadDraft}>Preview unsaved draft</button>}</div>}
    {mode==='arcviewer'&&<div style={{position:'absolute',top:54,right:12,fontSize:10,color:'#8d9cb3',pointerEvents:'none',background:'#060914cc',padding:'5px 9px',borderRadius:6}}>{arcState}</div>}
    {mode==='modstudio'&&<div className="mod-runtime-badge">{arcBlocked&&<><strong>Native V4 preview</strong> · {arcIssues.join(', ')}. ArcViewer cannot display these features. </>}{arcAvailable===false&&<>Optional ArcViewer build is unavailable. </>}<span data-testid="native-preview-clock">{nativeInfo.njs.toFixed(2)} NJS · {nativeInfo.bpm.toFixed(2)} BPM. </span>Documented transforms and lighting preview. Environment geometry and lane trajectories are approximations. Unity assets/shaders require the game runtime.{nativeInfo.warnings.map(w=><p key={w}>{w}</p>)}</div>}
    {showSupport&&<div className="viewer-support-popover"><strong>PREVIEW CAPABILITIES</strong>{MOD_CAPABILITIES.map(m=><p key={m.name}><b>{m.name}:</b> {mode==='arcviewer'?m.arc:m.preview}</p>)}</div>}
    {error&&mode!=='arcviewer'&&<div style={{position:'absolute',inset:'42% 15%',textAlign:'center',color:'#becbe0',fontSize:13}}>{error}</div>}
  </div>;
}
