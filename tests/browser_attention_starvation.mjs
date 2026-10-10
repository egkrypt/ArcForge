// Isolated API fixture: restrictive attention preferences must be visible and
// reset only by the user. No requests reach the installed ArcForge service.
import {createRequire} from 'node:module';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';

const {chromium}=createRequire(import.meta.url)(process.env.ARCFORGE_PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(process.env.ARCFORGE_FRONTEND_DIST||'frontend/dist');
const output=path.resolve('data/test-artifacts/attention-starvation-fix');
fs.mkdirSync(output,{recursive:true});
const types={'.html':'text/html','.js':'text/javascript','.css':'text/css','.png':'image/png','.svg':'image/svg+xml'};
const server=http.createServer((req,res)=>{
 const filePath=path.resolve(root,'.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
 if(filePath!==root&&!filePath.startsWith(root+path.sep)){res.writeHead(403);res.end();return;}
 const file=fs.existsSync(filePath)&&!fs.statSync(filePath).isDirectory()?filePath:path.join(root,'index.html');
 res.setHeader('content-type',types[path.extname(file)]||'application/octet-stream');res.end(fs.readFileSync(file));
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const baseURL=`http://127.0.0.1:${server.address().port}`;
const browser=await chromium.launch({channel:process.env.ARCFORGE_BROWSER_CHANNEL||'msedge',headless:true,args:['--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1600,height:1050},baseURL});
const errors=[],requests=[];
page.on('pageerror',error=>errors.push(error.message));
const audioId='b'.repeat(32);
const audio={id:audioId,name:'Attention starvation regression',bpm:182.5,duration:301.03,offset:0,confidence:.8,waveform:[{min:-.2,max:.4}],beatMarkers:[0,.3288,.6575],onsets:[0,.3288,.6575],sections:[]};
const projectId='a'.repeat(32),chartId='c'.repeat(32);
const chart={id:chartId,name:'Expert+',difficulty:'ExpertPlus',rank:9,mode:'Standard',njs:14,jumpDistance:15.29,data:{version:'3.3.0',colorNotes:Array.from({length:26},(_,i)=>({b:i*20,x:i%2?3:0,y:i%2,c:i%2,d:i%2})),bombNotes:[],obstacles:[],sliders:[],burstSliders:[],basicBeatmapEvents:[],rotationEvents:[],bpmEvents:[],colorBoostBeatmapEvents:[],lightColorEventBoxGroups:[],lightRotationEventBoxGroups:[],lightTranslationEventBoxGroups:[],customData:{}},report:{requestedNps:13,averageNps:.086,peakFourSecondNps:.75,engineSelection:{audioAnalysis:{requestedMode:'arcaudio3',effectiveMode:'arcaudio3'},placement:{requestedBackend:'hybrid',effectiveBackend:'local',fallbackReason:'Controlled Inferno unavailable fixture.'}},attention:{mode:'arcaudio3',detectedEvents:8407,selectedByAttention:25,summary:{rejectionReasons:{low_importance:6511,supporting_layer:1845}}}}};
const project={id:projectId,name:'Saved sparse generation',bpm:182.5,revision:1,environment:'DefaultEnvironment',audio,charts:[chart],config:{arcAudioMode:'arcaudio3',arcAudioSettings:{},backend:'hybrid'}};
const strictSettings={vocalPriority:.8,drumPriority:.7,attentionStrength:.6,smoothingSeconds:.5,competition:.8,maxDensity:.7,minImportance:.75,secondaryAllowance:.7,phraseAwareness:.8,backgroundSuppression:.8,repetitionSuppression:.8};
const prefs={theme:'Midnight',accent:'#bda8ff',density:'comfortable',editor:{wheelBeats:.25},ai:{style:'Tech',arcAudioMode:'arcaudio3',backend:'hybrid',arcAudioSettings:strictSettings},profile:{},workflow:{autoOpenGenerated:false},storage:{rememberImportedAudio:false}};
const send=(route,data,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(data)});
await page.route('**/api/**',async route=>{
 const request=route.request(),url=new URL(request.url()),endpoint=url.pathname,method=request.method();
 let body;if(method!=='GET'&&request.headers()['content-type']?.includes('application/json'))body=request.postDataJSON();
 requests.push({endpoint,method,body});
 if(endpoint==='/api/preferences')return send(route,prefs);
 if(endpoint==='/api/status')return send(route,{version:'1.3.0',device:'CPU'});
 if(endpoint==='/api/capabilities')return send(route,{environments:[{id:'DefaultEnvironment',name:'The First',modelVersion:'V2'}],mods:[],preview:{arcViewer:false}});
 if(endpoint==='/api/library')return send(route,{projects:[{id:projectId,name:project.name,bpm:182.5,duration:301.03,chartCount:1,audioId}],audio:[]});
 if(endpoint==='/api/dashboard')return send(route,{recent:[],mapCount:0,training:{archives:0,maps:0},activity:[]});
 if(endpoint==='/api/training/status')return send(route,{dataset:{maps:0,songs:0,samples:0},runtime:{cpu:{logicalCores:4},gpu:{available:false}}});
 if(endpoint==='/api/trash')return send(route,{projects:[]});
 if(endpoint==='/api/recognition/status')return send(route,{ready:true,model:'YAMNet',device:'CPU',version:'3.0'});
 if(endpoint==='/api/assets')return send(route,{assets:[]});
 if(endpoint===`/api/projects/${projectId}`)return send(route,project);
 if(endpoint===`/api/projects/${projectId}/versions`)return send(route,{versions:[],current:1});
 if(endpoint===`/api/audio/${audioId}`)return send(route,audio);
 if(endpoint==='/api/audio')return send(route,audio);
 if(endpoint.endsWith('/waveform'))return send(route,{id:audioId,start:Number(url.searchParams.get('start')),end:Number(url.searchParams.get('end')),waveform:audio.waveform});
 if(endpoint.endsWith('/attention')){
  const restrictive=Number(body?.settings?.minImportance)>=.7;
  const events=Array.from({length:100},(_,i)=>({time:i/10,source:'melody',importance:restrictive?.3:.6,selected:restrictive?i===0:i<70,reason:restrictive&&i>0?'low_importance':'primary_event'}));
  return send(route,{version:'3.0',mode:body.mode,frames:[{time:0,state:'music_focused',primarySource:'melody',sources:{melody:.5},attention:{melody:1},instrumentEvidence:{piano:.02,keyboard:.03,guitar:.11,synth:.06,plucked:.7,bowed:.04,brass:.01}}],events,phrases:[],summary:{rejectionReasons:restrictive?{low_importance:99}:{}},limitations:['Controlled browser fixture; no audio is classified.']});
 }
 if(endpoint==='/api/generate'){
  if(Number(body?.arcAudioSettings?.minImportance)>=.7)return send(route,{detail:'Attention filtering left too few eligible attacks for the requested difficulty. Review minimum event importance before generating.'},422);
  return send(route,{jobId:'d'.repeat(32)});
 }
 if(endpoint.startsWith('/api/jobs/'))return send(route,{status:'queued',progress:0,message:'Controlled fixture; generation not executed.'});
 return send(route,{});
});

try {
 await page.goto('/');
 await page.locator('.sidebar').getByRole('button',{name:'Settings',exact:true}).click();
 await page.getByTestId('strict-attention-warning').waitFor();
 assert.match(await page.getByTestId('strict-attention-warning').innerText(),/75%/);
 assert.equal(await page.getByLabel('Minimum event importance',{exact:true}).inputValue(),'0.75');
 await page.getByRole('button',{name:'Reset attention defaults',exact:true}).click();
 assert.equal(await page.getByLabel('Minimum event importance',{exact:true}).inputValue(),'0.24');
 assert.equal(await page.getByLabel('Density ceiling',{exact:true}).inputValue(),'1');
 assert.equal(await page.getByTestId('strict-attention-warning').count(),0);
 assert.equal(requests.filter(r=>r.endpoint==='/api/preferences'&&r.method!=='GET').length,0,'Reset must not silently save global preferences.');

 // Reload reuses the saved restrictive preferences: nothing was silently reset.
 await page.reload();
 await page.locator('.sidebar').getByRole('button',{name:'Levels',exact:true}).click();
 await page.getByRole('button',{name:'New level',exact:true}).click();
 await page.getByRole('button',{name:/AI first draft/}).click();
 await page.locator('input[type=file][accept^="audio/"]').setInputFiles({name:'fixture.ogg',mimeType:'audio/ogg',buffer:Buffer.from('Mocked upload; not real audio.')});
 await page.locator('.analyzed-audio').waitFor();
 await page.locator('.creation-audio-analysis>summary').click();
 await page.getByTestId('attention-filter-starvation').waitFor();
 assert.match(await page.getByTestId('attention-filter-starvation').innerText(),/1 of 100/);
 assert.match(await page.getByTestId('attention-filter-starvation').innerText(),/99 were rejected for low importance/);
 const hypotheses=page.getByTestId('instrument-hypotheses');
 await hypotheses.waitFor();assert.match(await hypotheses.innerText(),/Windowed model evidence; uncertain/);
 assert.equal(await hypotheses.locator('.arc-attention-share').count(),4);
 assert.match(await hypotheses.locator('.arc-attention-share').first().innerText(),/Plucked\s+0\.70/);
 assert.doesNotMatch(await hypotheses.innerText(),/Brass/,'Inspector should show only the top four hypotheses.');
 assert.equal(await page.getByLabel('Minimum event importance',{exact:true}).inputValue(),'0.75');
 await page.locator('.step-strip').getByRole('button',{name:/Difficulties/}).click();
 await page.getByLabel('Expert+ target NPS',{exact:true}).fill('13');
 await page.locator('.step-strip').getByRole('button',{name:/Fine tuning/}).click();
 await page.getByTestId('strict-attention-warning').waitFor();
 assert.match(await page.getByTestId('attention-density-ceiling').innerText(),/13 NPS request to at most 9\.1 active NPS/);
 assert.match(await page.getByTestId('attention-density-ceiling').innerText(),/NJS controls approach speed separately/);
 await page.screenshot({path:path.join(output,'strict-attention-review.png'),fullPage:true});
 await page.getByRole('button',{name:'Generate & playtest',exact:true}).click();
 await page.getByText('Attention filtering left too few eligible attacks for the requested difficulty. Review minimum event importance before generating.',{exact:true}).waitFor();
 const strictRequest=requests.find(r=>r.endpoint==='/api/generate');
 assert.equal(strictRequest.body.difficulties.find(d=>d.enabled).nps,13);
 assert.equal(strictRequest.body.arcAudioSettings.minImportance,.75);
 assert.equal(strictRequest.body.arcAudioSettings.maxDensity,.7);
 assert.equal(strictRequest.body.njs,null);

 await page.getByRole('button',{name:'Reset attention defaults',exact:true}).click();
 assert.equal(await page.getByTestId('strict-attention-warning').count(),0);
 assert.match(await page.getByTestId('attention-density-ceiling').innerText(),/13 NPS request to at most 13 active NPS/);
 await page.getByRole('button',{name:'Generate & playtest',exact:true}).click();
 await page.locator('.job-banner').waitFor();
 const generations=requests.filter(r=>r.endpoint==='/api/generate');
 assert.equal(generations.length,2);
 const resetRequest=generations[1].body;
 assert.equal(resetRequest.arcAudioSettings.minImportance,.24);
 assert.equal(resetRequest.arcAudioSettings.maxDensity,1);
 assert.equal(resetRequest.difficulties.find(d=>d.enabled).nps,13);
 assert.equal(resetRequest.njs,null);
 assert.equal(resetRequest.arcAudioMode,'arcaudio3');
 assert.equal(resetRequest.backend,'hybrid');
 await page.locator('.sidebar').getByRole('button',{name:'Levels',exact:true}).click();
 await page.getByRole('button',{name:`Actions for ${project.name}`,exact:true}).click();
 await page.getByRole('menuitem',{name:'Map Details',exact:true}).click();
 await page.getByTestId('chart-density-evidence').waitFor();
 assert.match(await page.getByTestId('chart-density-evidence').innerText(),/requested 13 active NPS; generated 0\.09 average NPS/);
 assert.match(await page.getByTestId('chart-density-evidence').innerText(),/NJS 14 m\/s is approach speed/);
 assert.match(await page.getByTestId('chart-attention-evidence').innerText(),/25 \/ 8,407 detected attacks/);
 assert.match(await page.getByTestId('chart-attention-evidence').innerText(),/6,511 rejected for low importance/);
 assert.match(await page.getByTestId('chart-attention-evidence').innerText(),/does not change this saved result/);
 assert.deepEqual(errors,[]);
 const report={passed:true,mockedApi:true,isolatedPort:server.address().port,strictSavedPreferencesVisible:true,noSilentReset:true,explicitResetRestoresDefaults:true,lowEligibilityExplained:true,uncertainInstrumentHypothesesVisible:true,finalReviewWarningVisible:true,densityCeilingExplained:true,npsSeparateFromNjs:true,backendFilterErrorVisible:true,resetUsedByGeneration:true,globalDefaultsNotSilentlySaved:true,savedDensityAndFilterEvidence:true,errors};
 fs.writeFileSync(path.join(output,'browser-regression.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify(report));
} catch(error) {
 await page.screenshot({path:path.join(output,'browser-regression-failure.png'),fullPage:true});throw error;
} finally {
 await browser.close();await new Promise(resolve=>server.close(resolve));
}
