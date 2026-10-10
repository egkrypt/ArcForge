import pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap
from modding import apply_mods,BeatClock
from tests.test_music_mapping import fixture
from engine.lighting_plan import lighting_evidence


class SharedLighting(unittest.TestCase):
    def test_style_materially_changes_articulation(self):
        analysis=fixture()
        analysis['musicAnalysis']['sections']=[{'start':0,'end':24,'label':'Drop','activity':.95,'energy':.9}]
        analysis['musicAnalysis']['accents']=[{'time':1+i*.08,'strength':.95,'roles':['lowPercussion' if i%2 else 'highPercussion']} for i in range(270)]
        outputs={style:apply_mods({'colorNotes':[]},analysis,{'mods':['Chroma'],'lightingStyle':style})[0] for style in ['Minimal','Cinematic','Energetic','Classic','High-BPM']}
        counts={s:len(o['basicBeatmapEvents']) for s,o in outputs.items()}
        self.assertGreater(counts['Energetic'],counts['Cinematic']*2)
        self.assertGreater(counts['High-BPM'],counts['Classic'])
        self.assertGreater(counts['Classic'],counts['Minimal'])
        for out in outputs.values():
            self.assertEqual(out['colorNotes'],[])
            self.assertEqual(out['customData']['arcforge']['lightingPlan']['nativeColorBanks'],['blue','red'])

    def test_attention_foreground_is_shared_not_reanalyzed(self):
        analysis=fixture()
        analysis['_arcAudioAttention']={'enabled':True,'frames':[{'time':0,'primarySource':'vocals'}],
            'events':[{'time':1.17,'source':'vocals','attentionSource':'vocals','selected':True,'importance':.8,'strength':.9},
                      {'time':1.3,'source':'piano','attentionSource':'piano','selected':False,'importance':.8,'strength':.9,'reason':'supporting_layer'}]}
        evidence=lighting_evidence(analysis,{})
        self.assertEqual([t['time'] for t in evidence['triggers']],[1.17])
        out,_=apply_mods({'colorNotes':[]},analysis,{'mods':[]})
        self.assertTrue(any(abs(BeatClock(analysis['bpm']).time(e['b'])-1.17)<.00001 for e in out['basicBeatmapEvents']))


if __name__=='__main__':unittest.main()
