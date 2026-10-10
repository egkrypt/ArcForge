"""Generated visual effects preserve real hit timing and native color banks."""
import copy, pathlib, sys, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import bootstrap
from modding import apply_mods
from test_music_mapping import fixture

class ModChoreography(unittest.TestCase):
    def test_drop_rotations_and_paired_banks(self):
        a=fixture();a['musicAnalysis']['sections']=[{'start':0,'end':24,'label':'Drop','activity':.9,'energy':.9}]
        a['musicAnalysis']['accents']=[{'time':i*.16+1,'strength':.9,'roles':['lowPercussion' if i%2 else 'highPercussion']} for i in range(100)]
        chart={'version':'3.3.0','colorNotes':[{'b':12,'x':0,'y':0,'c':0,'d':1,'a':0},{'b':13,'x':3,'y':2,'c':1,'d':0,'a':0}],
            'bpmEvents':[],'obstacles':[],'sliders':[],'burstSliders':[]}
        out,info=apply_mods(chart,a,{'mods':['Chroma','Noodle Extensions'],'seed':13,'events':{'lighting':True}})
        events=out['basicBeatmapEvents'];values={e['i'] for e in events if e['et']<5}
        self.assertTrue(values & {1,2,3});self.assertTrue(values & {5,6,7})
        rotations=[e for e in events if e['et']==8];self.assertGreater(len(rotations),8)
        self.assertEqual({e['customData'].get('direction') for e in rotations if 'direction' in e.get('customData',{})},{0,1})
        self.assertGreater(info['colorLeft']['r'],info['colorLeft']['b']);self.assertGreater(info['colorRight']['b'],info['colorRight']['r'])
        self.assertEqual([{k:n[k] for k in ['b','x','y','c','d']} for n in out['colorNotes']], [{k:n[k] for k in ['b','x','y','c','d']} for n in chart['colorNotes']])
        for note in out['colorNotes']:
            animation=note['customData']['animation']
            for key in ['offsetPosition','localRotation']:
                self.assertEqual(animation[key][1][:3],[0,0,0]);self.assertLess(animation[key][1][3],.5)
                self.assertEqual(animation[key][-1],[0,0,0,1])
    def test_frameworks_are_dependencies_not_visual_requirements(self):
        a=fixture();out,info=apply_mods({'colorNotes':[]},a,{'mods':['Heck','CustomJSONData'],'events':{'lighting':False}})
        self.assertNotIn('Heck',info['requirements']);self.assertNotIn('CustomJSONData',info['requirements'])
    def test_vanilla_contains_no_custom_gameplay_animation(self):
        chart={'colorNotes':[{'b':4,'x':3,'y':0,'c':1,'d':1,'a':0}]};a=fixture()
        out,info=apply_mods(chart,a,{'mods':[],'events':{'lighting':True}})
        self.assertEqual(out['colorNotes'],chart['colorNotes']);self.assertEqual(info['requirements'],[])
if __name__=='__main__':unittest.main()
