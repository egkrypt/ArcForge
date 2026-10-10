"""Output-level regression tests for audited timing/density/motion failures."""
import copy, pathlib, sys, unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import bootstrap
import numpy as np
from engine.generate import generate_map, _mapping_clock
from engine.scheduling import select_supported
from engine.features import untoken
from engine.music import VERSION
from quality_benchmark import metrics


def rhythmic_fixture(bpm=300, duration=12):
    period=60/bpm;grid=np.arange(0,duration,period).tolist()
    attacks=[{'time':round(float(t),6),'strength':1.,'roles':['lowPercussion' if i%4==0 else 'midTransient']}
        for i,t in enumerate(np.arange(0,duration,period/4))]
    return {'bpm':bpm,'duration':duration,'offset':0.,'beatTimes':grid,
        'frames':[{'time':float(t),'energy':.7,'bass':.6,'onset':.8,'high':.5} for t in np.arange(0,duration,.05)],
        'musicAnalysis':{'version':VERSION,'accents':attacks,'motifs':[],
            'sections':[{'start':0,'end':duration,'label':'Drop','energy':.9,'activity':1.,'densityFactor':1.,'primaryRole':'lowPercussion'}],
            'rhythmFrames':[{'time':float(t),'activity':1.} for t in np.arange(0,duration,.25)]}}


class MappingQuality(unittest.TestCase):
    def test_high_bpm_does_not_halve_the_entire_opportunity_grid(self):
        analysis=rhythmic_fixture();config={'nps':15,'rank':9,'backend':'local','seed':221,'style':'Speed','arcAudioMode':'legacy'}
        generated=generate_map(analysis,config,ROOT/'models/style');clock,_=_mapping_clock(analysis,config)
        times=[clock.time(n['b']) for n in generated['beatmap']['colorNotes']]
        actual=sum(2<=t<10 for t in times)/8
        # The old blanket 1/2-beat grid could not exceed 10 NPS at 300 BPM.
        # Safe subsets of the observed 1/4-beat attacks can exceed that ceiling.
        self.assertGreater(actual,11.5)
        self.assertTrue(generated['report']['passed'])
        self.assertTrue(all(min(abs(t-p['time']) for p in analysis['musicAnalysis']['accents'])<1e-5 for t in times))
        self.assertTrue(generated['report']['mapPlan']['sections'])
        self.assertEqual(generated['report']['critic']['density']['placementRejectedHits'],0)

    def test_scheduled_selection_keeps_two_close_accents_without_three_hand_conflict(self):
        pool=[{'time':t,'score':s} for t,s in [(0,1),(.04,.9),(.08,.8),(.16,.7),(.20,.6),(.24,.5)]]
        selected,report=select_supported(pool,6,.12,2)
        times=[p['time'] for p in selected]
        self.assertIn(0,times);self.assertIn(.04,times)
        self.assertTrue(all(times[i]-times[i-2]>=.1199 for i in range(2,len(times))))
        self.assertEqual(report['limitedBy'],'physicalRecovery')

    def test_salience_greedy_deficit_is_backfilled_from_existing_attacks(self):
        pool=[{'time':0.,'score':1.},{'time':.06,'score':10.},{'time':.12,'score':1.}]
        selected,report=select_supported(pool,2,.12,1)
        self.assertEqual([p['time'] for p in selected],[0.,.12])
        self.assertEqual(report['backfilledEvents'],1)

    def test_sparse_missing_motif_does_not_force_all_notes_to_red(self):
        analysis=rhythmic_fixture(120,12)
        times=[round(float(t),5) for t in np.arange(1,11,.5)]
        frames=[{'time':float(t),'state':'vocal_focused','primarySource':'vocals','rhythmDensity':.5,
            'densityTarget':1.,'confidence':1.} for t in np.arange(0,12,.25)]
        events=[{'time':t,'source':'vocals','importance':.9,'selected':True} for t in times]
        analysis['_arcAudioAttention']={'frames':frames,'events':events,'selectedEvents':events}
        generated=generate_map(analysis,{'nps':4,'rank':7,'backend':'local','style':'Flow','arcAudioMode':'arcaudio','seed':22},ROOT/'models/style')
        colors=[n['c'] for n in generated['beatmap']['colorNotes']]
        self.assertGreater(colors.count(0),4);self.assertGreater(colors.count(1),4)
        self.assertLessEqual(abs(colors.count(0)-colors.count(1)),2)

    def test_clock_preserves_abrupt_tempo_boundary_and_offset(self):
        beats=[.13+i*.5 for i in range(10)]+[.13+5+(i-10)*.25 for i in range(10,60)]
        analysis={'bpm':120,'offset':.13,'duration':20,'beatTimes':beats}
        clock,events=_mapping_clock(analysis,{})
        self.assertTrue(any(abs(e['b']-(10+.26))<1e-6 and abs(e['m']-240)<1e-6 for e in events))
        for i,t in enumerate(beats):self.assertAlmostEqual(clock.time(i+.26),t,places=5)

    def test_unknown_and_unavailable_explicit_engine_do_not_silently_use_local(self):
        with self.assertRaisesRegex(ValueError,'Unknown placement engine'):
            generate_map(rhythmic_fixture(),{'backend':'imaginary'},ROOT/'models/style')
        with self.assertRaisesRegex(ValueError,'InfernoSaber requires'):
            generate_map(rhythmic_fixture(),{'backend':'inferno'},ROOT/'models/style')

    def test_v3_source_trace_is_preserved_and_backend_report_is_truthful(self):
        analysis=rhythmic_fixture();times=np.arange(1,10,.1).tolist()
        frames=[{'time':float(t),'state':'rhythm_focused','primarySource':'drums','rhythmDensity':1.,
            'densityTarget':1.,'confidence':1.} for t in np.arange(0,12,.25)]
        events=[{'time':t,'source':'kick','importance':.9,'selected':True} for t in times]
        attention={'version':'3.0','frames':frames,'events':events,'selectedEvents':events,
            'recognition':{'model':'independent-fixture-recognizer'},'method':'supplied_evidence'}
        # Direct generation can calculate attention without server pre-caching.
        with patch('engine.arcaudio.analyze_attention',return_value=attention):
            generated=generate_map(analysis,{'arcAudioMode':'arcaudio3','backend':'local','nps':10,'rank':9,'seed':24},ROOT/'models/style')
        self.assertTrue(all(n['customData'].get('arcforgeSource') for n in generated['beatmap']['colorNotes']))
        self.assertEqual(generated['report']['engineSelection']['placement']['effectiveBackend'],'local')
        self.assertEqual(generated['report']['engineSelection']['audioAnalysis']['effectiveMode'],'arcaudio3')
        self.assertEqual(generated['report']['engineSelection']['audioAnalysis']['recognition'],'independent-fixture-recognizer')
        self.assertEqual(generated['report']['attention']['sourceAnalysisMethod'],'supplied_evidence')

    def test_style_and_profile_change_measured_motion_while_preserving_validity(self):
        analysis=rhythmic_fixture(120,24)
        base={'nps':4,'rank':9,'backend':'local','seed':324,'arcAudioMode':'legacy'}
        measured={}
        for style in ('Accuracy','Speed','Tech','Dance','Flow','Challenge'):
            config=dict(base,style=style)
            output=generate_map(analysis,config,ROOT/'models/style')
            measured[style]=metrics(output['beatmap'],analysis,config)
            self.assertTrue(measured[style]['playability']['passed'],style)
        self.assertGreater(measured['Dance']['meanGridTravel'],measured['Speed']['meanGridTravel']+.2)
        self.assertGreater(measured['Tech']['nonVerticalFraction'],measured['Accuracy']['nonVerticalFraction']+.04)
        self.assertGreater(measured['Dance']['crossLaneFraction'],.05)
        self.assertEqual(measured['Accuracy']['crossLaneFraction'],0)
        extremes=[]
        for movement in (0,100):
            config=dict(base,style='Dance',philosophy={'movement':movement,'tech':55,'variety':80,'musicality':85,'contrast':65,'stamina':65})
            output=generate_map(analysis,config,ROOT/'models/style')
            extremes.append(metrics(output['beatmap'],analysis,config))
        self.assertGreater(extremes[1]['meanGridTravel'],extremes[0]['meanGridTravel']+.2)

    def test_difficulty_spread_changes_density_and_directional_structure(self):
        analysis=rhythmic_fixture(190,24);rows=[]
        for rank,nps in ((1,1.5),(3,2.5),(5,4),(7,6),(9,12)):
            config={'rank':rank,'nps':nps,'backend':'local','style':'Modern','seed':934}
            result=generate_map(analysis,config,ROOT/'models/style')
            rows.append(metrics(result['beatmap'],analysis,config))
            self.assertTrue(rows[-1]['playability']['passed'])
        self.assertTrue(all(b['notes']>a['notes'] for a,b in zip(rows,rows[1:])))
        self.assertEqual(rows[0]['nonVerticalFraction'],0.)
        self.assertEqual(rows[1]['nonVerticalFraction'],0.)
        self.assertGreater(rows[-1]['nonVerticalFraction'],.1)

    def test_doubles_count_toward_the_requested_note_budget(self):
        analysis=rhythmic_fixture(120,24)
        config={'rank':9,'nps':4,'backend':'local','style':'Modern','seed':324}
        output=generate_map(analysis,config,ROOT/'models/style')
        self.assertLessEqual(len(output['beatmap']['colorNotes']),round(24*4))
        self.assertGreater(output['report']['doubleAccents'],0)
        self.assertEqual(output['report']['critic']['density']['placementRejectedHits'],0)


if __name__=='__main__':unittest.main()
