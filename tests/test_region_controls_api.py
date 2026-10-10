"""Regional intent must survive analysis, generation, preview and persistence."""
import copy, pathlib, sys, unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import bootstrap, attention_api, server, services_api
from test_engine_selection_api import EngineSelectionApi


class RegionApi(unittest.TestCase):
    setUp=EngineSelectionApi.setUp
    tearDown=EngineSelectionApi.tearDown
    run_job=EngineSelectionApi.run_job
    generate=EngineSelectionApi.generate
    persist=EngineSelectionApi.persist
    analyze=EngineSelectionApi.analyze

    def regions(self):
        return [{'id':'speech','name':'Spoken bridge','start':4.,'end':8.,'primaryStyle':'Flow','overrides':{},
                 'nps':3.5,'analysisMode':'custom','sources':{'spoken':1.},'sourcePolicy':'only'},
                {'id':'drop','name':'Full mix drop','start':12.,'end':20.,'npsRange':[8.,13.],
                 'primaryStyle':'Speed','analysisMode':'all','transitionSeconds':1.}]

    def test_region_intent_reaches_continuation_and_stays_in_preview(self):
        captured=[]
        def analyze(value,mode,settings,compute=None,mapping_regions=None,focus_regions=None):
            captured.append(copy.deepcopy(mapping_regions));return {'mode':mode,'frames':[],'events':[]}
        with patch.object(attention_api,'cached_attention',side_effect=analyze):
            response=self.client.post(f'/api/projects/{self.id}/continue',json={'ranges':[{'start':4,'end':8}],
                'analysis':{'arcAudioMode':'arcaudio3','mappingRegions':self.regions()},'backend':'local'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(captured[0][0]['sources'],{'spoken':1.})
        self.assertEqual(captured[0][1]['npsRange'],[8.,13.])
        self.assertEqual(self.result['previewProject']['config']['mappingRegions'],captured[0])
        self.assertEqual(self.state,self.original);self.assertEqual(self.saved,[])

    def test_project_saves_regions_compute_without_modifying_chart_or_defaults(self):
        with patch.object(services_api,'read_project',lambda key:copy.deepcopy(self.original)),patch.object(services_api,'persist',self.persist):
            response=self.client.patch(f'/api/projects/{self.id}/analysis',json={'arcAudioMode':'arcaudio3','mappingRegions':self.regions(),
                'analysisCompute':{'quality':'deep'}})
        self.assertEqual(response.status_code,200,response.text)
        saved=response.json();self.assertEqual(saved['charts'],self.original['charts'])
        self.assertEqual(saved['config']['mappingRegions'][1]['npsRange'],[8.,13.])
        self.assertEqual(saved['config']['analysisCompute']['quality'],'deep')
        self.assertNotIn('mappingRegions',self.original['config'])

    def test_invalid_regions_and_unsupported_source_engine_fail_before_queue(self):
        invalid=[self.regions()+[{'start':5,'end':7}],
                 [{'start':0,'end':25}],
                 [{'start':0,'end':4,'npsRange':[13,8]}],
                 [{'start':0,'end':4,'nps':8,'npsRange':[8,13]}],
                 [{'start':0,'end':4,'analysisMode':'custom','sources':{}}],
                 [{'start':0,'end':4,'analysisMode':'custom','sources':{'invented':1}}],
                 [{'start':0,'end':4,'analysisMode':'all'}]]
        with patch.object(server,'start_job') as queued:
            for index,regions in enumerate(invalid):
                mode='legacy' if index==len(invalid)-1 else 'arcaudio3'
                response=self.client.post(f'/api/projects/{self.id}/continue',json={'ranges':[{'start':4,'end':8}],
                    'analysis':{'arcAudioMode':mode,'mappingRegions':regions}})
                self.assertEqual(response.status_code,400,(regions,response.text))
            queued.assert_not_called()

    def test_nps_and_style_only_regions_keep_legacy_available(self):
        config=attention_api.regional_config({'arcAudioMode':'legacy','mappingRegions':[{'start':0,'end':4,'nps':12,'primaryStyle':'Speed'}]},24)
        self.assertEqual(config['mappingRegions'][0]['nps'],12.)

del EngineSelectionApi
if __name__=='__main__':unittest.main()
