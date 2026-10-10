"""Compute controls reach the analyzer, cache and generation without private models."""
import copy, pathlib, sys, tempfile, unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import bootstrap
import attention_api, server, services_api
from fastapi.testclient import TestClient
from test_engine_selection_api import EngineSelectionApi


class ComputeContract(unittest.TestCase):
    def test_missing_compute_retains_balanced_and_explicit_settings_validate(self):
        original={'arcAudioMode':'legacy'}
        result=attention_api.generation_overrides(original,{})
        self.assertEqual(result['analysisCompute'],{'quality':'balanced','device':'auto','cpuThreads':0})
        self.assertEqual(original,{'arcAudioMode':'legacy'})
        for value in [[],False,{'quality':'fake'},{'device':'fake'},{'cpuThreads':-1},{'cpuThreads':1.5}]:
            with self.assertRaises(ValueError):attention_api.generation_overrides(original,{'analysis':{'analysisCompute':value}})

    def test_compute_changes_attention_cache_and_endpoint_forwards_it(self):
        (ROOT/'data/test-artifacts').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'data/test-artifacts',prefix='compute-api-') as folder:
            analysis={'id':'a'*32,'duration':10,'bpm':120,'frames':[]}
            def run(value,settings,mode,compute,**kwargs):
                return {'mode':mode,'performance':{'analysisCompute':compute},'recognition':{'performance':{'device':'cpu','inferenceExecuted':True,'cacheHit':False}},'frames':[]}
            with patch.object(attention_api,'DATA',pathlib.Path(folder)),patch('engine.arcaudio.analyze_attention',side_effect=run) as analyzed,patch.object(server,'audio_record',return_value=analysis):
                client=TestClient(server.app)
                balanced=client.post('/api/audio/'+'a'*32+'/attention',json={'mode':'arcaudio3','compute':{'quality':'balanced'}})
                deep=client.post('/api/audio/'+'a'*32+'/attention',json={'mode':'arcaudio3','compute':{'quality':'deep','device':'cpu','cpuThreads':1}})
                cached=client.post('/api/audio/'+'a'*32+'/attention',json={'mode':'arcaudio3','compute':{'quality':'deep','device':'cpu','cpuThreads':1}})
                self.assertEqual(balanced.status_code,200,balanced.text);self.assertEqual(deep.status_code,200,deep.text)
                self.assertEqual(deep.json()['performance']['analysisCompute']['quality'],'deep')
                self.assertFalse(deep.json()['performance']['cacheHit']);self.assertTrue(cached.json()['performance']['cacheHit'])
                self.assertFalse(cached.json()['recognition']['performance']['inferenceExecuted'])
                self.assertEqual(analyzed.call_count,2)
                self.assertEqual(client.post('/api/audio/'+'a'*32+'/attention',json={'mode':'arcaudio3','compute':{'device':'invalid'}}).status_code,400)

    def test_runtime_reports_real_probe_without_requiring_model_weights(self):
        expected={'cpu':{'logicalCores':12},'gpu':{'available':True,'name':'fixture'}}
        with patch('engine.analysis_runtime.analysis_compute_status',return_value=expected):
            response=TestClient(server.app).get('/api/analysis/runtime')
            self.assertEqual(response.status_code,200);self.assertEqual(response.json(),expected)


class ComputeGenerationContract(EngineSelectionApi):
    def test_deep_choice_reaches_continuation_and_saved_preview_only(self):
        compute={'quality':'deep','device':'cpu','cpuThreads':1}
        captured=[]
        def analyze(analysis,mode,settings,compute=None,mapping_regions=None,focus_regions=None):
            captured.append(copy.deepcopy(compute));return {'mode':mode,'frames':[],'events':[]}
        with patch.object(attention_api,'cached_attention',side_effect=analyze):
            response=self.client.post(f'/api/projects/{self.id}/continue',json={'ranges':[{'start':4,'end':8}],
                'analysis':{'arcAudioMode':'arcaudio3','analysisCompute':compute},'backend':'local'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(captured,[compute]);self.assertEqual(self.generated[0]['analysisCompute'],compute)
        self.assertEqual(self.result['previewProject']['config']['analysisCompute'],compute)
        self.assertEqual(self.state,self.original)

del EngineSelectionApi
if __name__=='__main__':unittest.main()
