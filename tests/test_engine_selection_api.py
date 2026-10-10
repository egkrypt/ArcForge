"""The engine shown in the UI must reach generation without silent fallbacks."""
import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import bootstrap
import attention_api
import projects
import server
import studio
from fastapi.testclient import TestClient


class EngineSelectionApi(unittest.TestCase):
    def setUp(self):
        (ROOT/'data/test-artifacts').mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='engine-api-',dir=ROOT/'data/test-artifacts')
        self.data=pathlib.Path(self.temp.name)
        self.id='a'*32;self.chart='b'*32;self.audio='c'*32
        self.analysis={'id':self.audio,'name':'Controlled API fixture','duration':24,'bpm':120,'offset':0,'beatTimes':[i*.5 for i in range(49)],'frames':[]}
        native={'version':'3.3.0','colorNotes':[{'b':2,'x':0,'y':0,'c':0,'d':1,'a':0},{'b':20,'x':3,'y':0,'c':1,'d':1,'a':0}],
                'bpmEvents':[],'bombNotes':[],'obstacles':[],'sliders':[],'burstSliders':[],'basicBeatmapEvents':[],'rotationEvents':[]}
        self.original={'id':self.id,'name':'Isolated engine contract','created':1,'revision':1,'bpm':120,'environment':'DefaultEnvironment',
            'audio':self.analysis,'config':{'arcAudioMode':'legacy','backend':'hybrid','style':'Flow','arcAudioSettings':{'vocalPriority':.83}},
            'charts':[{'id':self.chart,'name':'Expert+','mode':'Standard','rank':9,'difficulty':'ExpertPlus','njs':20,'jumpDistance':24,'data':native,'report':{'requestedNps':8}}]}
        self.state=copy.deepcopy(self.original);self.saved=[];self.generated=[];self.attention=[]
        self.patches=[patch.object(m,'DATA',self.data) for m in [projects,server,studio,attention_api]]
        self.patches.extend([
            patch.object(attention_api,'read_project',lambda key:copy.deepcopy(self.state)),
            patch.object(studio,'read_project',lambda key:copy.deepcopy(self.state)),
            patch.object(server,'audio_record',lambda key:copy.deepcopy(self.analysis)),
            patch.object(server,'start_job',self.run_job),
            patch.object(attention_api,'cached_attention',self.analyze),
            patch('engine.generate_map',self.generate),
            patch('audio.ensure_music_analysis',lambda value:value),
            patch('gameplay.add_events',lambda value,*a,**kw:value),
            patch('modding.apply_mods',lambda value,*a,**kw:(value,{})),
            patch('gameplay.analyze_playability',lambda *a,**kw:{'passed':True,'issues':[]}),
            patch('gameplay.repair_playability',lambda value,*a,**kw:{'beatmap':value,'report':{'passed':True,'issues':[]},'changes':[]}),
            patch.object(studio,'persist',self.persist),
        ])
        for item in self.patches:item.start()
        self.client=TestClient(server.app,raise_server_exceptions=True)
    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.temp.cleanup()
    def run_job(self,kind,work,resources=None):
        self.result=work(lambda *a,**kw:None)
        return {'jobId':'d'*32}
    def analyze(self,analysis,mode,settings,compute=None,mapping_regions=None,focus_regions=None):
        self.attention.append((mode,copy.deepcopy(settings)))
        return {'mode':mode,'frames':[],'events':[]}
    def generate(self,analysis,config,model):
        self.generated.append(copy.deepcopy(config))
        native=copy.deepcopy(self.original['charts'][0]['data'])
        native['colorNotes']=[{'b':10,'x':1,'y':1,'c':0,'d':0,'a':0}]
        return {'beatmap':native,'report':{'passed':True,'engineSelection':{'audioAnalysis':{'effectiveMode':config['arcAudioMode']},'placement':{'requestedBackend':config['backend']}}}}
    def persist(self,before,after,message):
        after['revision']=before['revision']+1;self.saved.append(copy.deepcopy(after));return after

    def test_mode_only_override_inherits_settings_without_mutating_saved_config(self):
        before=copy.deepcopy(self.original['config'])
        result=attention_api.generation_overrides(before,{'analysis':{'arcAudioMode':'arcaudio3'},'backend':'local'})
        self.assertEqual(before,self.original['config'])
        self.assertEqual(result['arcAudioSettings']['vocalPriority'],.83)
        self.assertEqual(result['arcAudioMode'],'arcaudio3');self.assertEqual(result['backend'],'local')
        result['arcAudioSettings']['vocalPriority']=0
        self.assertEqual(before['arcAudioSettings']['vocalPriority'],.83)

    def test_continuation_uses_unsaved_engine_choice_but_only_changes_preview(self):
        payload={'chartId':self.chart,'ranges':[{'start':4,'end':8}],'operation':'fill',
                 'analysis':{'arcAudioMode':'arcaudio3','arcAudioSettings':{'drumPriority':.21}},'backend':'local'}
        response=self.client.post(f'/api/projects/{self.id}/continue',json=payload)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.generated[0]['arcAudioMode'],'arcaudio3');self.assertEqual(self.generated[0]['backend'],'local')
        self.assertEqual(self.attention[0][0],'arcaudio3');self.assertEqual(self.attention[0][1]['drumPriority'],.21)
        self.assertEqual(self.state,self.original);self.assertEqual(self.saved,[])
        preview=self.result['previewProject']
        self.assertEqual(preview['config']['arcAudioMode'],'arcaudio3')
        self.assertEqual(preview['config']['backend'],'local')
        old=self.original['charts'][0]['data']['colorNotes'];new=preview['charts'][0]['data']['colorNotes']
        for note in old:self.assertIn(note,new)
        self.assertEqual(preview['charts'][0]['report']['continuation']['engineSelection']['audioAnalysis']['effectiveMode'],'arcaudio3')

    def test_section_regeneration_receives_same_explicit_engine_contract(self):
        response=self.client.post(f'/api/projects/{self.id}/section',json={'chartId':self.chart,'start':4,'end':8,'action':'regenerate',
            'analysis':{'arcAudioMode':'hybrid'},'backend':'inferno'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.generated[0]['arcAudioMode'],'hybrid');self.assertEqual(self.generated[0]['backend'],'inferno')
        self.assertEqual(self.attention[0][0],'hybrid');self.assertEqual(self.attention[0][1]['vocalPriority'],.83)
        self.assertEqual(self.state,self.original);self.assertEqual(len(self.saved),1)
        changed=self.saved[0]
        self.assertEqual(changed['config']['arcAudioMode'],'hybrid');self.assertEqual(changed['config']['backend'],'inferno')
        self.assertEqual(changed['charts'][0]['report']['sectionRevision']['engineSelection']['placement']['requestedBackend'],'inferno')
        self.assertEqual([n for n in changed['charts'][0]['data']['colorNotes'] if n['b']<8 or n['b']>=16],self.original['charts'][0]['data']['colorNotes'])

    def test_invalid_override_types_are_rejected_before_work_is_queued(self):
        invalid=[{'analysis':[]},{'analysis':'arcaudio3'},{'analysis':{'chart':{}}},
                 {'analysis':{'arcAudioMode':'missing'}},{'analysis':{'arcAudioSettings':[]}},
                 {'analysis':{'arcAudioSettings':False}},{'analysis':{'arcAudioSettings':0}},
                 {'backend':'missing'},{'backend':[]},{'backend':{}},{'backend':False}]
        with patch.object(server,'start_job') as queued:
            for override in invalid:
                for endpoint,base in [('/continue',{'ranges':[{'start':4,'end':8}]}),('/section',{'start':4,'end':8,'action':'regenerate'})]:
                    response=self.client.post(f'/api/projects/{self.id}{endpoint}',json=base|override)
                    self.assertEqual(response.status_code,400,(endpoint,override,response.text))
            queued.assert_not_called()

    def test_new_generation_rejects_unknown_placement_or_analysis_before_queue(self):
        base={'audioId':self.audio,'difficulties':[{'name':'Expert+','rank':9,'nps':10}],
              'environment':'DefaultEnvironment','mods':[],'events':{},'compatibility':'Vanilla'}
        invalid=[{'backend':'missing'},{'backend':[]},{'backend':{}},{'backend':False},
                 {'arcAudioMode':'missing'},{'arcAudioSettings':False},{'arcAudioSettings':0},
                 {'beatmapVersion':'5'},{'beatmapVersion':None},{'dynamicNjs':'true'}]
        with patch.object(server,'start_job') as queued:
            for overrides in invalid:
                response=self.client.post('/api/generate',json=base|overrides)
                self.assertEqual(response.status_code,400,(overrides,response.text))
            queued.assert_not_called()

    def test_creation_echoed_bpm_preserves_dynamic_grid_and_offset_only_shifts_it(self):
        timeline=[{'start':0.,'end':8.,'bpm':120.},{'start':8.,'end':24.,'bpm':180.}]
        beats=[i*.5 for i in range(16)]+[8+i/3 for i in range(49)]
        self.analysis.update(beatTimes=beats,tempoTimeline=timeline)
        captured=[]
        def compose(analysis,config,model):
            captured.append(copy.deepcopy(analysis))
            return self.generate(analysis,config,model)
        base={'audioId':self.audio,'backend':'local','arcAudioMode':'legacy','beatmapVersion':'3',
              'bpm':120.,'startOffset':0.,'events':{},'mods':[],
              'difficulties':[{'name':'Expert+','rank':9,'nps':8}],'modes':['Standard']}
        with patch.object(server,'generate_map',compose),patch.object(server,'model_status',return_value={'ready':True}),\
             patch.object(server,'export_project',return_value=self.data/'unused.zip'),\
             patch.object(server,'validate_project',return_value={'errors':[],'warnings':[]}):
            for change in ({},{'startOffset':.125},{'bpm':150.}):
                response=self.client.post('/api/generate',json=base|change)
                self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(captured[0]['beatTimes'],beats)
        self.assertEqual(captured[0]['tempoTimeline'],timeline)
        self.assertEqual(captured[1]['tempoTimeline'],timeline)
        self.assertEqual(captured[1]['beatTimes'],[round(t+.125,6) for t in beats if t+.125<=24])
        self.assertEqual(captured[2]['tempoTimeline'][0]['source'],'user_override')
        self.assertEqual(captured[2]['tempoTimeline'][0]['bpm'],150.)
        self.assertEqual(self.analysis['beatTimes'],beats)

    def test_optional_native_jump_events_reach_the_saved_generated_chart(self):
        from tests.test_dynamic_jump import fixture
        native,_,_,report=fixture()
        report['passed']=True
        base={'audioId':self.audio,'backend':'local','arcAudioMode':'legacy','beatmapVersion':'4',
              'dynamicNjs':True,'events':{},'mods':[],'compatibility':'Modded',
              'difficulties':[{'name':'Expert+','rank':9,'nps':8}],'modes':['Standard']}
        with patch.object(server,'generate_map',return_value={'beatmap':native,'report':report}),\
             patch.object(server,'model_status',return_value={'ready':True}),\
             patch.object(server,'export_project',return_value=self.data/'unused.zip'),\
             patch.object(server,'validate_project',return_value={'errors':[],'warnings':[]}):
            response=self.client.post('/api/generate',json=base)
        self.assertEqual(response.status_code,200,response.text)
        chart=self.result['project']['charts'][0]
        self.assertTrue(chart['data']['njsEvents'])
        self.assertTrue(chart['report']['jumpPlan']['applied'])
        self.assertTrue(chart['report']['mapPlan']['jumpStrategy']['dynamic'])


if __name__=='__main__':unittest.main()
