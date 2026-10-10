import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import {createRequire} from 'node:module';

// The API, audio and ArcViewer iframe are controlled fixtures. No live maps
// are loaded or saved; WebGL uses the real editor renderer in Edge.
const {chromium}=createRequire(import.meta.url)(process.env.ARCFORGE_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(process.env.ARCFORGE_FRONTEND_DIST||'frontend/dist');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png'};
const web=http.createServer((request,response)=>{
 let file=path.join(root,decodeURIComponent(new URL(request.url,'http://localhost').pathname));
 if(!file.startsWith(root+path.sep)&&file!==root){response.writeHead(403).end();return;}
 if(!fs.existsSync(file)||fs.statSync(file).isDirectory())file=path.join(root,'index.html');
 response.setHeader('content-type',mime[path.extname(file)]||'application/octet-stream');response.end(fs.readFileSync(file));
});
await new Promise(resolve=>web.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({channel:process.env.ARCFORGE_BROWSER_CHANNEL||'msedge',headless:true,args:['--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1600,height:1050},baseURL:`http://127.0.0.1:${web.address().port}`});
const id='a'.repeat(32),audioId='b'.repeat(32),chartId='c'.repeat(32);
const audio={id:audioId,name:'Native preview fixture',bpm:120,duration:30,offset:0,confidence:.8,waveform:[{min:-.2,max:.3}],beatTimes:[0,.5,1],sections:[]};
const data={version:'3.3.0',colorNotes:[],bombNotes:[],obstacles:[],sliders:[],burstSliders:[],basicBeatmapEvents:[],bpmEvents:[],rotationEvents:[],customData:{}};
let project={id,name:'Native preview fixture',bpm:120,revision:1,environment:'DefaultEnvironment',songTimeOffset:0,audio,config:{arcAudioMode:'legacy'},charts:[{id:chartId,name:'Expert+',mode:'Standard',difficulty:'ExpertPlus',rank:9,njs:20,jumpDistance:24,data}]};
const errors=[],writes=[],iframeUrls=[];
page.on('pageerror',error=>errors.push(error.message));
const send=(route,value)=>route.fulfill({contentType:'application/json',body:JSON.stringify(value)});
await page.route('**/api/**',async route=>{
 const req=route.request(),url=new URL(req.url()),endpoint=url.pathname;
 if(req.method()!=='GET')writes.push({endpoint,method:req.method()});
 if(endpoint==='/api/preferences')return send(route,{theme:'Midnight',accent:'#bda8ff',editor:{wheelBeats:.25},ai:{arcAudioMode:'legacy'},storage:{rememberImportedAudio:false},workflow:{autoOpenGenerated:true}});
 if(endpoint==='/api/library')return send(route,{projects:[{id,name:project.name,bpm:120,duration:30,chartCount:1,audioId}],audio:[audio]});
 if(endpoint==='/api/dashboard')return send(route,{recent:[],mapCount:1,training:{archives:1,maps:1},activity:[]});
 if(endpoint==='/api/capabilities')return send(route,{environments:[{id:'DefaultEnvironment',name:'The First',modelVersion:'V2'}],preview:{arcViewer:true},beatmapVersions:['3','4'],mods:[]});
 if(endpoint==='/api/status')return send(route,{version:'1.3.0'});
 if(endpoint==='/api/training/status')return send(route,{dataset:{songs:1,maps:1},runtime:{},recommendation:{epochs:8}});
 if(endpoint===`/api/projects/${id}`)return send(route,project);
 if(endpoint===`/api/projects/${id}/versions`)return send(route,{versions:[],current:project.revision});
 if(endpoint===`/api/audio/${audioId}`)return send(route,audio);
 if(endpoint.endsWith('/waveform'))return send(route,{id:audioId,start:Number(url.searchParams.get('start')),end:Number(url.searchParams.get('end')),waveform:audio.waveform});
 if(endpoint.endsWith('/attention'))return send(route,{version:'2.0',frames:[],events:[],phrases:[],summary:{},limitations:[]});
 if(endpoint==='/api/recognition/status')return send(route,{ready:false,installed:false,runtimeAvailable:true});
 return send(route,{});
});
await page.route('**/viewer/**',route=>{iframeUrls.push(route.request().url());return route.fulfill({contentType:'text/html',body:'<!doctype html><title>Controlled ArcViewer fixture</title><body>Environment viewer fixture</body>'});});
const open=async()=>{await page.goto('/');await page.getByRole('button',{name:'Levels',exact:true}).click();await page.getByRole('button',{name:'Open in Studio',exact:true}).click();await page.locator('.editor-workspace').waitFor();};
try{
 await open();await page.locator('iframe[title="ArcViewer environment player"]').waitFor();
 const savedUrl=new URL(await page.locator('iframe[title="ArcViewer environment player"]').getAttribute('src'),'http://localhost');
 const exportUrl=new URL(savedUrl.searchParams.get('url'));
 assert.equal(exportUrl.searchParams.get('version'),'3','ArcViewer must request a validated V3 conversion.');
 assert.equal(exportUrl.pathname,`/api/projects/${id}/export`);
 project={...project,revision:2,charts:[{...project.charts[0],data:{...structuredClone(data),colorNotes:[{b:2,x:1,y:1,c:1,d:1,r:90}],bpmEvents:[{b:4,m:240}],njsEvents:[{b:0,d:0,p:0,e:0},{b:4,d:4,p:0,e:6},{b:8,d:123,p:1,e:0}]}}]};
 await open();await page.locator('.viewer-wrap canvas').waitFor();
 const arcButton=page.getByRole('button',{name:'ArcViewer environments',exact:true});
 assert.equal(await arcButton.isDisabled(),true);assert.match(await arcButton.getAttribute('title'),/native V4.1 NJS events/);
 assert.equal(await page.locator('iframe[title="ArcViewer environment player"]').count(),0);
 assert.match(await page.locator('.mod-runtime-badge').innerText(),/Native V4 preview.*per-object lane rotations/);
 await page.getByLabel('Cursor beat').fill('2');
 await page.getByTestId('native-preview-clock').filter({hasText:'22.00 NJS · 120.00 BPM'}).waitFor();
 // A rotated note at the paused cursor remains on the frontal edit plane.
 // Select it by raycasting at the known projection of (-.3,1.35,0).
 await page.getByRole('button',{name:'Note palette +',exact:true}).click();
 const canvas=page.locator('.viewer-wrap canvas'),bounds=await canvas.boundingBox();
 await canvas.click({position:{x:bounds.width/2-bounds.height*.0621,y:bounds.height*.5453}});
 await page.locator('.inspector').getByText('Selection',{exact:true}).waitFor();
 assert.equal(await page.locator('.inspector-fields label').filter({hasText:'Column'}).locator('input').inputValue(),'1');
 assert.equal(await page.locator('.save-state.dirty').count(),0,'Selecting the existing paused note must not create another note.');
 await page.getByLabel('Cursor beat').fill('6');
 await page.getByTestId('native-preview-clock').filter({hasText:'24.00 NJS · 240.00 BPM'}).waitFor();
 await page.getByLabel('Cursor beat').fill('2');
 await page.getByTestId('native-preview-clock').filter({hasText:'22.00 NJS · 120.00 BPM'}).waitFor();
 assert.deepEqual(writes,[]);assert.deepEqual(errors,[]);
 fs.mkdirSync('data/test-artifacts',{recursive:true});
 await page.screenshot({path:'data/test-artifacts/browser-native-preview.png',fullPage:true});
 const report={passed:true,savedArcViewerUsesV3:true,nativeFeaturesRouteToModPreview:true,backwardNjsEasing:true,njsExtendPrevious:true,dynamicTempo:true,rewindStable:true,pausedRotatedNoteStaysEditable:true,serverWrites:writes,pageErrors:errors};
 fs.writeFileSync('data/test-artifacts/browser-native-preview.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
}catch(error){fs.mkdirSync('data/test-artifacts',{recursive:true});await page.screenshot({path:'data/test-artifacts/browser-native-preview-failure.png',fullPage:true});throw error;}
finally{await browser.close();await new Promise(resolve=>web.close(resolve));}
