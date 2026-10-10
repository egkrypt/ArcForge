import fs from 'node:fs';import path from 'node:path';
import {createRequire} from 'node:module';import {pathToFileURL} from 'node:url';
const schemaRoot=process.env.ARCFORGE_BSMAP_ROOT;
if(!schemaRoot)throw new Error('Set ARCFORGE_BSMAP_ROOT to an installed bsmap 3.0.3 package directory.');
const require=createRequire(import.meta.url);
const {safeParse}=await import(pathToFileURL(require.resolve('valibot',{paths:[schemaRoot]})).href);
const {InfoSchema}=await import(pathToFileURL(path.join(schemaRoot,'esm/beatmap/schema/v2/declaration/info.js')).href);
const {DifficultySchema}=await import(pathToFileURL(path.join(schemaRoot,'esm/beatmap/schema/v3/declaration/difficulty.js')).href);
const {InfoSchema:Info4Schema}=await import(pathToFileURL(path.join(schemaRoot,'esm/beatmap/schema/v4/declaration/info.js')).href);
const {DifficultySchema:Difficulty4Schema}=await import(pathToFileURL(path.join(schemaRoot,'esm/beatmap/schema/v4/declaration/difficulty.js')).href);
const {LightshowSchema}=await import(pathToFileURL(path.join(schemaRoot,'esm/beatmap/schema/v4/declaration/lightshow.js')).href);
const {AudioDataSchema}=await import(pathToFileURL(path.join(schemaRoot,'esm/beatmap/schema/v4/declaration/audioData.js')).href);
const root=path.resolve(process.argv[2]);const info=JSON.parse(fs.readFileSync(path.join(root,'Info.dat'),'utf8'));
const result=[];
const isV4=String(info.version||info._version).startsWith('4');
const files=isV4?[['Info.dat',Info4Schema],[info.audio.audioDataFilename,AudioDataSchema],...info.difficultyBeatmaps.flatMap(d=>[[d.beatmapDataFilename,Difficulty4Schema],[d.lightshowDataFilename,LightshowSchema]])]:[['Info.dat',InfoSchema],...info._difficultyBeatmapSets.flatMap(s=>s._difficultyBeatmaps.map(d=>[d._beatmapFilename,DifficultySchema]))];
for(const [file,schema] of files){
 const parsed=safeParse(schema(),JSON.parse(fs.readFileSync(path.join(root,file),'utf8')));
 result.push({file,success:parsed.success,issues:parsed.issues?.map(i=>({message:i.message,path:i.path?.map(p=>p.key).join('.')}))||[]});
}
fs.writeFileSync(path.join(root,'schema-validation.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));if(result.some(r=>!r.success))process.exitCode=1;
