"""Native jump events are opt-in and survive the actual indexed V4 codec."""
import copy, math, pathlib, sys, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import bootstrap
from engine.jump import apply_dynamic_njs
from engine.formats import encode_v4,decode_v4,validate_v4
from modding import BeatClock


def fixture():
    chart={'version':'3.3.0','colorNotes':[],'bombNotes':[],'obstacles':[],
        'sliders':[],'burstSliders':[],'basicBeatmapEvents':[],'bpmEvents':[{'b':16,'m':240}]}
    sections=[];critic=[]
    for i,(start,end,nps,bpm) in enumerate([(0,8,2,120),(8,16,12,240),(16,24,3,240)]):
        identifier=f'plan-{i}'
        sections.append({'id':identifier,'start':start,'end':end,'targetNps':nps,
            'localBpm':bpm,'goals':{'tech':50,'movement':50}})
        critic.append({'id':identifier,'actualNps':nps})
        chart['colorNotes'].append({'b':start*2,'c':i%2,'x':i%4,'y':1,'d':1,'a':0,
            'customData':{'arcforgePlanSection':identifier}})
    analysis={'bpm':120,'duration':24}
    config={'bpm':120,'beatmapVersion':'4','dynamicNjs':True,'njsAuto':True,
        'compatibility':'Modded','mode':'Standard'}
    report={'mapPlan':{'sections':sections},'critic':{'sections':critic}}
    return chart,analysis,config,report


class DynamicJump(unittest.TestCase):
    def test_measured_density_drives_native_bounded_tempo_aware_events(self):
        chart,analysis,config,report=fixture();original=copy.deepcopy(chart)
        result,diagnostics=apply_dynamic_njs(chart,analysis,config,report,20,24)
        self.assertEqual(chart,original)
        self.assertTrue(diagnostics['applied'])
        self.assertGreater(diagnostics['sections'][1]['recommendedNjs'],diagnostics['sections'][0]['recommendedNjs'])
        events=result['njsEvents'];clock=BeatClock(120,chart['bpmEvents'])
        self.assertTrue(all(abs(e['d'])<=2 for e in events))
        self.assertTrue(all(a['b']<b['b'] for a,b in zip(events,events[1:])))
        for previous,event in zip(events,events[1:]):
            if event['p']:continue
            seconds=clock.time(event['b'])-clock.time(previous['b'])
            self.assertGreaterEqual(seconds,1.49999)
            self.assertLessEqual(math.pi/2*abs(event['d']-previous['d'])/seconds,1.50001)
            self.assertEqual(event['e'],6)
        self.assertTrue(any(e['p']==1 for e in events))
        packed=encode_v4(result)
        validate_v4(packed['difficulty'],packed['lightshow'])
        restored=decode_v4(packed['difficulty'],packed['lightshow'])
        self.assertEqual(restored['njsEvents'],events)

    def test_v3_manual_ranked_disabled_and_rotating_remain_unchanged(self):
        chart,analysis,config,report=fixture()
        for overrides in ({'beatmapVersion':'3'},{'njs':21},{'njsAuto':False},
            {'compatibility':'Ranked-compatible'},{'dynamicNjs':False},{'mode':'360Degree'}):
            result,diagnostics=apply_dynamic_njs(chart,analysis,config|overrides,report,20,24)
            self.assertEqual(result,chart)
            self.assertFalse(diagnostics['applied'])
            self.assertTrue(diagnostics['sections'])
            self.assertTrue(diagnostics['reason'])

    def test_authored_events_preserved_and_native_reaction_margin_not_reduced(self):
        chart,analysis,config,report=fixture()
        chart['njsEvents']=[{'b':2.,'d':3.,'p':0,'e':0}]
        result,diagnostics=apply_dynamic_njs(chart,analysis,config,report,20,24)
        self.assertEqual(result,chart)
        self.assertFalse(diagnostics['applied'])
        self.assertTrue(all(s['nativeReactionTimeMs']>=diagnostics['baseReactionTimeMs'] for s in diagnostics['sections']))
        self.assertTrue(all(s['nativeJumpDistance']>=24 for s in diagnostics['sections']))


if __name__=='__main__':unittest.main()
