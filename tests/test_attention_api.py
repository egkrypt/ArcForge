"""API contracts: optional attention, advisory checks and review-only AI edits."""
import copy, json, pathlib, sys, tempfile, unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import bootstrap
from fastapi.testclient import TestClient
import server, attention_api, studio
from test_music_mapping import fixture
from audio import public_analysis
from engine.generate import generate_map

class AttentionApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis=fixture();cls.analysis['id']='a'*32
        cls.client=TestClient(server.app)

    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(prefix='attention-api-',dir=ROOT/'data/test-artifacts')
        self.path=pathlib.Path(self.folder.name)
        self.assertTrue(self.path.resolve().is_relative_to((ROOT/'data/test-artifacts').resolve()))
        self.patches=[patch.object(attention_api,'DATA',self.path),patch.object(studio,'DATA',self.path),
                      patch.object(server,'audio_record',lambda key:copy.deepcopy(self.analysis))]
        for p in self.patches:p.start()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.folder.cleanup()

    def test_attention_cache_obeys_grid_and_tuning(self):
        a=copy.deepcopy(self.analysis)
        first=attention_api.cached_attention(a,'arcaudio',{})
        again=attention_api.cached_attention(a,'arcaudio',{})
        self.assertFalse(first['performance']['cacheHit']);self.assertTrue(again['performance']['cacheHit'])
        a['offset']+=.04;a['beatTimes']=[t+.04 for t in a['beatTimes']]
        shifted=attention_api.cached_attention(a,'arcaudio',{})
        self.assertFalse(shifted['performance']['cacheHit'])
        self.assertEqual(len(list(self.path.glob('audio/*/attention/*.json'))),2)
        response=self.client.post('/api/audio/'+'a'*32+'/attention',json={'mode':'legacy'})
        self.assertEqual(response.status_code,200);self.assertFalse(response.json()['enabled'])
        self.assertEqual(self.client.post('/api/audio/'+'a'*32+'/attention',json={'mode':'invalid'}).status_code,400)

    def test_private_attention_never_leaks_into_project_audio(self):
        value=public_analysis(self.analysis|{'_arcAudioAttention':{'large':'private'},'sourceEvidence':{'private':True}})
        self.assertNotIn('_arcAudioAttention',value);self.assertNotIn('sourceEvidence',value)

    def test_manual_double_blue_warning_never_changes_input(self):
        chart={'bpmEvents':[],'colorNotes':[{'b':4,'x':3,'y':0,'c':1,'d':1,'a':0},
                                         {'b':5,'x':3,'y':0,'c':1,'d':1,'a':0}]}
        original=copy.deepcopy(chart)
        result=self.client.post('/api/playtest/advisory',json={'chart':chart,'audioId':'a'*32,'config':{'bpm':190,'rank':9}})
        self.assertEqual(result.status_code,200)
        report=result.json();parity=[i for i in report['issues'] if i['code']=='parity']
        self.assertTrue(parity);self.assertEqual(parity[0]['severity'],'warning')
        self.assertIn('blue/right',parity[0]['description']);self.assertEqual(report['changedNotes'],0)
        self.assertFalse(report['autoFix']);self.assertEqual(chart,original)
        self.assertEqual(report['errors'],sum(i['severity']=='error' for i in report['issues']))
        self.assertEqual(self.client.post('/api/playtest/advisory',json={'chart':{'colorNotes':[{}]},'audioId':'a'*32}).status_code,400)

    def test_preview_accept_conflict_and_undo_preserve_authored_work(self):
        generated=generate_map(self.analysis,{'bpm':190,'nps':8,'rank':9,'seed':76,'backend':'local'},ROOT/'models/style')
        data=copy.deepcopy(generated['beatmap'])
        data['colorNotes']=[{'b':t*190/60,'c':i%2,'x':0 if i%2==0 else 3,'y':0,'d':1,'a':0,
            'customData':{'authored':'preserve'}} for i,t in enumerate([2.,6.,9.,14.,21.])]
        data['basicBeatmapEvents']=[{'b':1.,'et':0,'i':1,'f':1.}]
        source={'id':'b'*32,'name':'API fixture','bpm':190,'audio':public_analysis(self.analysis),
            'created':1,'revision':1,'config':{'arcAudioMode':'legacy','style':'Flow'},'environment':'DefaultEnvironment',
            'charts':[{'id':'d'*32,'name':'Expert+','mode':'Standard','rank':9,'difficulty':'ExpertPlus',
                'njs':20,'jumpDistance':24,'data':data,'report':{'requestedNps':8}}]}
        store={'project':copy.deepcopy(source),'result':None}
        studio.write_json(self.path/'projects'/source['id']/'project.json',source)
        def read(key):return copy.deepcopy(store['project'])
        original_persist=studio.persist
        def save(previous,project,message):
            result=original_persist(previous,project,message);store['project']=copy.deepcopy(result);return result
        def start(kind,work,resources=None):
            store['result']=work(lambda *args,**kwargs:None);return {'jobId':'c'*32}
        chart_id=source['charts'][0]['id'];payload={'chartId':chart_id,'ranges':[{'start':8,'end':12},{'start':17,'end':20}],'operation':'fill'}
        with patch.object(attention_api,'read_project',read),patch.object(attention_api,'persist',save),patch.object(server,'start_job',start),patch('engine.generate_map',return_value=generated):
            result=self.client.post('/api/projects/'+'b'*32+'/continue',json=payload)
            self.assertEqual(result.status_code,200,result.text);self.assertEqual(store['project'],source)
            preview=store['result'];self.assertTrue(preview['report']['preservedOutside']);self.assertTrue(preview['report']['preservedExistingInside'])
            candidate=preview['previewProject'];old=source['charts'][0]['data'];new=candidate['charts'][0]['data']
            self.assertEqual(new['colorNotes'][:len(old['colorNotes'])],old['colorNotes'])
            self.assertEqual(new['basicBeatmapEvents'],old['basicBeatmapEvents'])
            route='/api/projects/'+'b'*32+'/continue/'+preview['previewId']
            store['project']['revision']+=1
            self.assertEqual(self.client.post(route+'/accept',json={}).status_code,400)
            store['project']=copy.deepcopy(source)
            accepted=self.client.post(route+'/accept',json={})
            self.assertEqual(accepted.status_code,200,accepted.text);self.assertEqual(store['project']['revision'],source['revision']+1)
            self.assertEqual(self.client.post(route+'/accept',json={}).status_code,400)
            undone=self.client.post(route+'/undo',json={})
            self.assertEqual(undone.status_code,200,undone.text)
            self.assertEqual(store['project']['charts'],source['charts'])
            self.assertEqual(store['project']['revision'],source['revision']+2)

if __name__=='__main__':unittest.main()
