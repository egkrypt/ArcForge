// Read-only smoke test against the installed local service. Never saves a map.
import {createRequire} from 'node:module';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const require=createRequire(import.meta.url);
const {chromium}=require(process.env.ARCFORGE_PLAYWRIGHT_MODULE||'playwright');
const browser=await chromium.launch({channel:process.env.ARCFORGE_BROWSER_CHANNEL||'msedge',headless:true,args:['--enable-unsafe-swiftshader']});
const page=await browser.newPage({viewport:{width:1600,height:1050},baseURL:process.env.ARCFORGE_URL||'http://127.0.0.1:8769'});
const errors=[],writes=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>{if(['POST','PUT','PATCH','DELETE'].includes(r.method()))writes.push({method:r.method(),url:new URL(r.url()).pathname})});
try {
 await page.goto('/',{waitUntil:'domcontentloaded'});
 await page.getByRole('heading',{name:'The next great phrase starts here.'}).waitFor({timeout:45000});
 await page.locator('.sidebar').getByRole('button',{name:'Levels',exact:true}).click();
 await page.getByRole('heading',{name:'Your levels.'}).waitFor();
 await page.getByRole('button',{name:'Open in Studio',exact:true}).first().click();
 await page.locator('.studio-heading').waitFor({timeout:30000});
 const capabilities=await (await page.request.get('/api/capabilities')).json();
 let unityReady=false;
 if(capabilities.preview?.arcViewer) {
  await page.getByText('ArcViewer ready · use its playback controls',{exact:false}).waitFor({timeout:90000});
  unityReady=true;
 }
 await page.getByRole('button',{name:'Mod preview',exact:true}).click();
 await page.locator('.viewer-wrap canvas').first().waitFor({state:'visible',timeout:15000});
 await page.getByRole('button',{name:'Light grid',exact:true}).click();
 await page.getByRole('heading',{name:'Light grid',exact:true}).waitFor();
 await page.getByRole('tab',{name:'Analytics',exact:true}).click();
 await page.locator('.nps-graph').waitFor({timeout:30000});
 await page.locator('.sidebar').getByRole('button',{name:'Settings',exact:true}).click();
 await page.getByRole('heading',{name:'Workflow & storage',exact:true}).waitFor();
 assert.equal(await page.getByRole('button',{name:'Restart ArcForge',exact:true}).count(),1);
 assert.equal(await page.getByRole('switch',{name:'Remember imported audio'}).count(),1);
 assert.deepEqual(writes,[],'Read-only live smoke test unexpectedly wrote to the service.');
 assert.deepEqual(errors,[]);
 const report={passed:true,realServer:true,unityReady,modPreview:true,lightGrid:true,analytics:true,settings:true,writes,errors};
 fs.mkdirSync('data/test-artifacts',{recursive:true});fs.writeFileSync('data/test-artifacts/browser13-live.json',JSON.stringify(report,null,2));
 console.log(JSON.stringify(report));
} finally {await browser.close()}
