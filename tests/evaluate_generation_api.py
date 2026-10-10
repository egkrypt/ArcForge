"""Real local generation through the HTTP contract, using an isolated workspace.

Usage: python tests/evaluate_generation_api.py PROJECT_ID
Reads the selected local song/checkpoint. Never writes that project or library.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,pathlib,shutil,sys,time
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap
import server,projects,studio,attention_api,services_api
from fastapi.testclient import TestClient


def evaluate(project_id,mode='arcaudio3',dynamic_njs=False,config_override=None,use_project_config=False):
    original=projects.read_project(server.identity(project_id))
    original_path=ROOT/'data/projects'/project_id/'project.json'
    original_sha=hashlib.sha256(original_path.read_bytes()).hexdigest()
    audio_id=original['audio']['id']
    source=ROOT/'data/audio'/audio_id
    analysis=json.loads((source/'analysis.json').read_text('utf-8'))
    folder=ROOT/'data/test-artifacts/master-end-to-end'/str(time.time_ns())
    audio=folder/'audio'/audio_id;audio.mkdir(parents=True)
    shutil.copy2(source/'song.ogg',audio/'song.ogg')
    analysis['audioPath']=str(audio/'song.ogg')
    projects.write_json(audio/'analysis.json',analysis)
    result={}
    def synchronous_job(kind,work,resources=None):
        result['value']=work(lambda *args:None)
        return {'jobId':'a'*32}
    patches=[patch.object(module,'DATA',folder) for module in [server,projects,studio,attention_api,services_api]]
    patches.append(patch.object(server,'start_job',synchronous_job))
    for item in patches:item.start()
    started=time.perf_counter()
    try:
        with TestClient(server.app) as client:
            payload={'audioId':audio_id,'name':'Isolated full-pipeline evaluation',
                'backend':'local','arcAudioMode':mode,'beatmapVersion':'4','dynamicNjs':dynamic_njs,'seed':197,'style':'Modern',
                'learnLighting':True,'compatibility':'Modded','mods':['Chroma','Noodle Extensions'],
                'events':{'lighting':True,'arcs':True,'chains':True,'obstacles':True,'bombs':False},
                'difficulties':[{'name':'Easy','rank':1,'nps':2},{'name':'Hard','rank':5,'nps':5},{'name':'Expert+','rank':9,'nps':12}],
                'modes':['Standard'],'bpm':analysis['bpm'],'startOffset':analysis['offset']}
            if use_project_config:
                payload=copy.deepcopy(original['config'])
                payload.update(audioId=audio_id,manual=False)
            payload.update(copy.deepcopy(config_override or {}))
            response=client.post('/api/generate',json=payload)
            projects.write_json(folder/'request.json',payload)
            if response.status_code!=200:
                projects.write_json(folder/'rejection.json',{'statusCode':response.status_code,'detail':response.json(),
                    'originalProjectUnchanged':hashlib.sha256(original_path.read_bytes()).hexdigest()==original_sha})
            if response.status_code!=200:raise AssertionError(response.text)
            generated=result['value']['project'];key=generated['id']
            export=client.get(f'/api/projects/{key}/export?version=4')
            if export.status_code!=200:raise AssertionError(export.text)
            (folder/'native-map.zip').write_bytes(export.content)
            rows=[]
            for chart in generated['charts']:
                report=chart['report']
                rows.append({'name':chart['name'],'notes':len(chart['data']['colorNotes']),
                    'requestedNps':report['requestedNps'],'averageNps':report.get('averageNps'),
                    'peakFourSecondNps':report.get('peakFourSecondNps'),
                    'engineSelection':report['engineSelection'],'critic':report.get('critic'),
                    'attentionViability':report.get('attentionViability'),
                    'attentionSummary':report.get('attention',{}).get('summary'),
                    'playtestErrors':report['playtest'].get('errors'),
                    'njs':chart['njs'],'jumpDistance':chart['jumpDistance'],
                    'nativeNjsEvents':len(chart['data'].get('njsEvents',[])),'jumpPlan':report.get('jumpPlan'),
                    'lighting':chart['data'].get('customData',{}).get('arcforge',{}).get('lightingPlan'),
                    'nativeMapSha256':hashlib.sha256(json.dumps(chart['data'],sort_keys=True).encode()).hexdigest()})
            summary={'seconds':round(time.perf_counter()-started,3),'sourceAudioId':audio_id,
                'sameDisplayedBpmPreservedTempo':generated['audio'].get('tempoTimeline')==server.audio_record(audio_id).get('tempoTimeline'),
                'tempoAnalysisVersion':generated['audio'].get('tempoAnalysis',{}).get('version'),
                'originalProjectUnchanged':hashlib.sha256(original_path.read_bytes()).hexdigest()==original_sha,
                'charts':rows,'validation':generated['validation'],'nativeExport':str(folder/'native-map.zip')}
            projects.write_json(folder/'evaluation.json',summary)
            print(json.dumps({'artifact':str(folder/'evaluation.json'),'seconds':summary['seconds'],
                'charts':[{k:v for k,v in r.items() if k not in ['critic','lighting','jumpPlan']} for r in rows]},indent=2),flush=True)
            return summary
    finally:
        for item in reversed(patches):item.stop()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('project_id')
    parser.add_argument('--mode',choices=['off','legacy','arcaudio','hybrid','arcaudio3'],default='arcaudio3')
    parser.add_argument('--dynamic-njs',action='store_true')
    parser.add_argument('--use-project-config',action='store_true',help='Reproduce all saved generation choices in isolation')
    parser.add_argument('--balanced-attention',action='store_true',help='Explicitly reset minimum importance/density ceiling for comparison')
    parser.add_argument('--backend',choices=['local','inferno','hybrid','auto'])
    args=parser.parse_args();overrides={}
    if args.backend:overrides['backend']=args.backend
    if args.balanced_attention:
        config=projects.read_project(server.identity(args.project_id)).get('config',{})
        overrides['arcAudioSettings']=dict(config.get('arcAudioSettings',{}),minImportance=.24,maxDensity=1.)
    evaluate(args.project_id,args.mode,args.dynamic_njs,overrides,args.use_project_config)
