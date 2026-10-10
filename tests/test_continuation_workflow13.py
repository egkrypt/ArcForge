"""Real neural generation → preview → regenerate → accept → undo → native ZIP."""
import copy, io, json, pathlib, sys, tempfile, time, unittest, zipfile
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import bootstrap, server, projects, studio, attention_api, services_api
from fastapi.testclient import TestClient
from test_music_mapping import fixture
from audio import public_analysis

class ContinuationWorkflow(unittest.TestCase):
    def test_real_job_and_reversible_region_edit(self):
        (ROOT/'data/test-artifacts').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='continuation13-',dir=ROOT/'data/test-artifacts') as temporary:
            folder=pathlib.Path(temporary);analysis=fixture();key='a'*32;audio='b'*32;chart_id='c'*32;analysis['id']=audio
            chart={'version':'3.3.0','colorNotes':[{'b':t*190/60,'x':0 if i%2==0 else 3,'y':0,'c':i%2,'d':1 if i<2 else 0,'a':0,'customData':{'authored':True}}
                for i,t in enumerate([5,6,18,19])], 'bpmEvents':[],'basicBeatmapEvents':[{'b':10,'et':1,'i':6,'f':1}],
                'bombNotes':[],'obstacles':[],'sliders':[],'burstSliders':[]}
            project={'id':key,'name':'Continuation QA','created':1,'updated':1,'revision':1,'bpm':190,
                'environment':'DefaultEnvironment','audio':public_analysis(analysis),
                'config':{'arcAudioMode':'hybrid','backend':'local','style':'Flow'},
                'charts':[{'id':chart_id,'name':'Expert+','mode':'Standard','rank':9,'njs':20,'jumpDistance':24,'data':chart,'report':{'requestedNps':10}}]}
            replacements=[patch.object(m,'DATA',folder) for m in [server,projects,studio,attention_api,services_api]]
            replacements+=[patch.object(server,'JOBS',{}),patch.object(server,'CANCEL',{}),patch.object(server,'audio_record',lambda x:copy.deepcopy(analysis))]
            for p in replacements:p.start()
            try:
                projects.write_json(folder/'projects'/key/'project.json',project)
                audio_folder=folder/'audio'/audio;audio_folder.mkdir(parents=True);(audio_folder/'song.ogg').write_bytes(b'OggS fixture')
                with TestClient(server.app) as client:
                    def wait(job_id):
                        for _ in range(400):
                            response=client.get('/api/jobs/'+job_id);self.assertEqual(response.status_code,200)
                            job=response.json()
                            if job['status'] in ['completed','failed','cancelled']:
                                self.assertEqual(job['status'],'completed',job);return job['result']
                            time.sleep(.05)
                        self.fail('Continuation job timed out')
                    body={'chartId':chart_id,'ranges':[{'start':8,'end':12},{'start':13,'end':16}],
                        'operation':'fill','matchDensity':False,'matchFlow':True,'matchRhythm':True}
                    response=client.post(f'/api/projects/{key}/continue',json=body);self.assertEqual(response.status_code,200,response.text)
                    preview=wait(response.json()['jobId'])
                    self.assertEqual(projects.read_project(key),project)
                    data=preview['previewProject']['charts'][0]['data']
                    self.assertGreater(len(data['colorNotes']),len(chart['colorNotes']))
                    self.assertEqual(data['colorNotes'][:len(chart['colorNotes'])],chart['colorNotes'])
                    self.assertEqual(data['basicBeatmapEvents'],chart['basicBeatmapEvents'])
                    self.assertTrue(preview['report']['preservedOutside'])
                    response=client.post(f"/api/projects/{key}/continue/{preview['previewId']}/regenerate",json={})
                    self.assertEqual(response.status_code,200,response.text);regenerated=wait(response.json()['jobId'])
                    accepted=client.post(f"/api/projects/{key}/continue/{regenerated['previewId']}/accept",json={})
                    self.assertEqual(accepted.status_code,200,accepted.text);self.assertEqual(accepted.json()['revision'],2)
                    exported=client.get(f'/api/projects/{key}/export');self.assertEqual(exported.status_code,200,exported.text[:200])
                    with zipfile.ZipFile(io.BytesIO(exported.content)) as z:
                        self.assertEqual(json.loads(z.read('ExpertPlusStandard.dat'))['colorNotes'],regenerated['previewProject']['charts'][0]['data']['colorNotes'])
                    undone=client.post(f"/api/projects/{key}/continue/{regenerated['previewId']}/undo",json={})
                    self.assertEqual(undone.status_code,200,undone.text);self.assertEqual(undone.json()['charts'][0]['data'],chart)
                    report={'passed':True,'previewDidNotSave':True,'authoredNotesPreserved':True,'lightingPreserved':True,
                        'generatedNotes':len(data['colorNotes'])-len(chart['colorNotes']),'regenerateAcceptUndo':True,'nativeZip':True}
                    (ROOT/'data/test-artifacts/continuation13-result.json').write_text(json.dumps(report,indent=2),'utf-8')
            finally:
                for p in reversed(replacements):p.stop()
if __name__=='__main__':unittest.main()
