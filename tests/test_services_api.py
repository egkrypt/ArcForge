"""Lifecycle/settings/export contracts must preserve authored work."""
import copy, json, pathlib, sys, tempfile, unittest, zipfile
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap, server, services_api, studio, projects
from fastapi.testclient import TestClient

class ServicesApi(unittest.TestCase):
    def setUp(self):
        (ROOT/'data/test-artifacts').mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='services-',dir=ROOT/'data/test-artifacts');self.data=pathlib.Path(self.temp.name)
        self.patches=[patch.object(m,'DATA',self.data) for m in [services_api,studio,projects,server]]
        self.patches += [patch.object(studio,'SETTINGS',self.data/'preferences.json'),patch.object(services_api,'PREVIEWS',self.data/'previews')]
        for p in self.patches:p.start()
        self.key='a'*32;self.audio='b'*32;self.chart='c'*32
        self.project={'id':self.key,'name':'Authored fixture','created':1,'updated':1,'revision':1,'bpm':120,
            'environment':'DefaultEnvironment','audio':{'id':self.audio,'duration':12,'bpm':120},'config':{'arcAudioMode':'legacy'},
            'charts':[{'id':self.chart,'name':'Expert+','rank':9,'mode':'Standard','data':{'version':'3.3.0','colorNotes':[
                {'b':2,'x':0,'y':0,'c':0,'d':1,'a':0,'customData':{'authored':True}}],
                'bpmEvents':[],'basicBeatmapEvents':[{'b':1,'et':0,'i':1,'f':1}]}}]}
        projects.write_json(self.data/'projects'/self.key/'project.json',self.project)
        audio=self.data/'audio'/self.audio;audio.mkdir(parents=True);(audio/'song.ogg').write_bytes(b'OggS-test-fixture')
        projects.write_json(audio/'analysis.json',self.project['audio'])
        self.client=TestClient(server.app)
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def test_analysis_patch_never_overwrites_notes_or_global_preferences(self):
        result=self.client.patch(f'/api/projects/{self.key}/analysis',json={'arcAudioMode':'arcaudio3','learnLighting':True})
        self.assertEqual(result.status_code,200,result.text)
        value=result.json();self.assertEqual(value['charts'],self.project['charts']);self.assertEqual(value['revision'],2)
        self.assertEqual(value['config']['arcAudioMode'],'arcaudio3');self.assertFalse(studio.SETTINGS.exists())
        bad=self.client.patch(f'/api/projects/{self.key}/analysis',json={'charts':[]})
        self.assertEqual(bad.status_code,400)
    def test_draft_preview_exports_unsaved_lights_without_saving_project(self):
        original=(self.data/'projects'/self.key/'project.json').read_bytes()
        data=copy.deepcopy(self.project['charts'][0]['data']);data['basicBeatmapEvents'][0]['i']=6
        result=self.client.post(f'/api/projects/{self.key}/preview',json={'chartId':self.chart,'chart':data})
        self.assertEqual(result.status_code,200,result.text);value=result.json()
        response=self.client.get(value['zipUrl']);self.assertEqual(response.status_code,200)
        with zipfile.ZipFile(__import__('io').BytesIO(response.content)) as z:
            chart=json.loads(z.read('ExpertPlusStandard.dat'));self.assertEqual(chart['basicBeatmapEvents'][0]['i'],6)
            self.assertEqual(z.read('song.ogg'),b'OggS-test-fixture')
        self.assertEqual(original,(self.data/'projects'/self.key/'project.json').read_bytes())
        self.assertFalse(value['savedProjectChanged'])
    def test_permanent_delete_and_purge_only_named_project_folders(self):
        projects.write_json(studio.SETTINGS,{'storage':{'permanentDeletion':True}})
        other=copy.deepcopy(self.project);other['id']='e'*32
        projects.write_json(self.data/'projects'/other['id']/'project.json',other)
        result=self.client.post(f'/api/projects/{self.key}/action',json={'action':'delete'})
        self.assertEqual(result.status_code,200,result.text);self.assertTrue(result.json()['permanent'])
        self.assertFalse((self.data/'projects'/self.key).exists());self.assertTrue((self.data/'audio'/self.audio/'song.ogg').exists())
        trash=self.data/'trash'/('d'*32);trash.mkdir(parents=True);(trash/'draft.txt').write_text('delete me')
        result=self.client.post('/api/trash/purge');self.assertEqual(result.json()['deleted'],1)
        self.assertFalse(trash.exists());self.assertTrue((self.data/'audio'/self.audio).exists())
        result=self.client.post(f"/api/projects/{other['id']}/action",json={'action':'delete'})
        self.assertTrue(result.json()['audioDeleted']);self.assertFalse((self.data/'audio'/self.audio).exists())
    def test_defaults_preserve_old_preferences_and_hide_audio_history(self):
        projects.write_json(studio.SETTINGS,{'theme':'Forest','editor':{'wheelBeats':.5}})
        value=self.client.get('/api/preferences').json()
        self.assertEqual(value['theme'],'Forest');self.assertFalse(value['storage']['rememberImportedAudio'])
        self.assertTrue(value['workflow']['autoOpenGenerated']);self.assertTrue(value['editor']['ghost'])
    def test_shutdown_refuses_active_work(self):
        with patch.object(server,'JOBS',{'x':{'status':'running'}}):
            result=self.client.post('/api/system/stop')
            self.assertEqual(result.status_code,400);self.assertIn('active job',result.json()['detail'])
    def test_recognition_status_is_real_optional_model(self):
        result=self.client.get('/api/recognition/status')
        self.assertEqual(result.status_code,200);self.assertEqual(result.json()['classes'],521)
        self.assertTrue(result.json()['local'])
if __name__=='__main__':unittest.main()
