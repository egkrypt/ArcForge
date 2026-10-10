import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url),ts=require('../frontend/node_modules/typescript');
const source=fs.readFileSync('frontend/src/modRuntime.ts','utf8');
const output=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext,target:ts.ScriptTarget.ES2022}}).outputText;
const {samplePoint,createModRuntime,decodeCoordinate,cutRotation,isDot,validateModChart}=await import(`data:text/javascript;base64,${Buffer.from(output).toString('base64')}`);
assert.deepEqual(samplePoint([[0,0,0,0],[2,4,6,1]],.5),[1,2,3]);
assert.deepEqual(samplePoint([[2,4,6,.4],[5,8,9,.7]],.1),[2,4,6]);
assert.deepEqual(samplePoint('movement',.5,3,{movement:[[0,0,0,0],[2,0,0,1,'easeInQuad']]}),[.5,0,0]);
assert.equal(decodeCoordinate(2000),1);assert.equal(decodeCoordinate(-2500),-1.5);assert.equal(decodeCoordinate(3),3);
assert.equal(cutRotation(1000),180);assert.equal(cutRotation(1090),90);assert.equal(isDot(2180),true);
const clock={seconds:b=>b/2};
const chart={version:'3.3.0',colorNotes:[],customData:{pointDefinitions:{swing:[[0,0,0,0],[2,0,0,1]]},customEvents:[
 {b:0,t:'AnimateTrack',d:{track:'note',duration:4,offsetPosition:'swing',color:[[1,0,0,1,0],[0,0,1,1,1]]}},
 {b:0,t:'AssignPathAnimation',d:{track:'note',offsetPosition:[[0,2,0,0],[0,0,0,.5]],scale:[[1,1,1,0],[2,2,2,1]]}},
 {b:1,t:'AssignPathAnimation',d:{track:'note',offsetPosition:[[0,4,0,0],[0,0,0,.5]]}},
 {b:0,t:'AnimateTrack',d:{track:'scenery',duration:4,localRotation:[[0,0,0,0],[0,0,90,1]],dissolve:[[1,0],[0,1]]}},
 ]}};
const runtime=createModRuntime(chart,clock);let p=runtime.object({customData:{track:'note'}},1,.25);
assert.deepEqual(p.position,[1,2,0]);assert.deepEqual(p.scale,[1.25,1.25,1.25]);assert.deepEqual(p.color,[.5,0,.5,1]);
p=runtime.object({customData:{track:'note'}},.25,.25);assert.deepEqual(p.position,[.25,1,0]); // Look backwards after a later preview; no leaked state.
p=runtime.object({customData:{track:'note',animation:{offsetPosition:[[0,1,0,0],[0,0,0,.5]],dissolve:[[0,0],[1,1]]}}},1,.25);assert.deepEqual(p.position,[1,2.5,0]);assert.equal(p.dissolve,.25);
p=runtime.object({customData:{animation:{time:[[0,0],[.15,.35],[.5,.5],[1,1]]}}},1,.35);assert.equal(p.time,.15); // Individual approach curves must reach the renderer, not just track curves.
assert.equal(runtime.object({customData:{animation:{time:[[0,0],[.15,.35],[.5,.5],[1,1]]}}},1,.5).time,.5);
const scenery=runtime.track('scenery',1,.5);assert.deepEqual(scenery.rotation,[0,0,45]);assert.equal(scenery.dissolve,.5);
assert.deepEqual(validateModChart(chart),[]);assert.ok(validateModChart({...chart,colorNotes:[{b:NaN}]}).length>0);assert.ok(validateModChart({...chart,customData:{customEvents:[{b:1,t:'AnimateTrack'}]}}).length>0);
for(const customData of [{customEvents:{}},{customEvents:[null]},{environment:{}},[]])assert.ok(validateModChart({...chart,customData}).length>0);
assert.ok(validateModChart({...chart,customData:{customEvents:[{b:0,t:'AnimateTrack',d:[]}]}}).length>0);
assert.ok(validateModChart({...chart,bpmEvents:[{b:0,m:0}]}).length>0);
assert.ok(validateModChart({...chart,lightColorEventBoxGroups:[{b:0,e:{}}]}).length>0);
assert.ok(validateModChart({...chart,colorNotes:[{b:0,x:0,y:0,c:2,d:1}]}).length>0);
assert.deepEqual(validateModChart({...chart,colorNotes:[{b:0,x:-2500,y:5000,c:0,d:1090}],customData:{customEvents:[{_time:0,_type:'AnimateTrack',_data:{_track:'legacy',_duration:2}}]}}),[]);
assert.deepEqual(validateModChart({...chart,obstacles:[{b:0,x:0,y:0,w:0,h:0,d:0,customData:{size:[2,3,4],uninteractable:true}}]}),[]); // Modded scenery can replace native wall dimensions.
const cyclic={...chart};cyclic.customData={cycle:cyclic};assert.ok(validateModChart(cyclic).some(e=>e.includes('circular')));
assert.ok(validateModChart({...chart,customData:{bad:1n}}).length>0);
assert.ok(validateModChart({}).length>0);
const parent=createModRuntime({customData:{customEvents:[{b:0,t:'AnimateTrack',d:{track:'parent',offsetPosition:[2,0,0]}},{b:0,t:'AssignTrackParent',d:{childrenTracks:['child'],parentTrack:'parent'}},{b:0,t:'AnimateTrack',d:{track:'child',offsetPosition:[1,0,0]}}]}},clock);assert.deepEqual(parent.track('child',1,.5).position,[3,0,0]);
fs.writeFileSync('data/mod-runtime-test.json',JSON.stringify({passed:true,checks:['interpolation/easing/clamping','named point definitions','Mapping Extensions precise coordinates/360 rotations','source animation replacement','rewind reproducibility','individual+track animation','individual time curve reaches renderer','environment rotation+dissolve','parent translation','malformed JSON and circular/non-JSON script result validation','native object fields and positive tempo validation','legacy custom event preservation']},null,2));
console.log('Mod preview runtime tests passed.');
