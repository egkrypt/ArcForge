"""Observed accent doubles use note quotas, attention admission and hand recovery."""
import copy
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import bootstrap
import numpy as np
from modding import BeatClock
from engine.attention_candidates import select_attention_candidates, _hit_times


def fixture(times=None, *, state='rhythm_focused', source='kick', strength=.95):
    times=list(np.arange(1.,11.,.25)) if times is None else list(times)
    frames=[{'time':float(t),'state':state,'primarySource':'drums',
        'rhythmDensity':1.,'densityTarget':1.,'confidence':.9} for t in np.arange(0,12,.25)]
    events=[{'id':f'observed-{i}','time':float(t),'source':source,'strength':strength,'confidence':.8,
        'importance':.9,'selected':True,'reason':'primary_event','timingEvidence':'observed_onset',
        'temporalSalience':1.2,'rhythmicSalience':1.,'coincidentSources':[source],
        'admission':{'importancePassed':True,'target':'primary','observedAttack':True}}
        for i,t in enumerate(times)]
    attention={'version':'3.0','frames':frames,'events':events,'selectedEvents':list(events),
        'settings':{'maxDensity':1.},'phrases':[{'start':t,'end':t+2.} for t in range(0,12,2)]}
    return {'duration':12.,'bpm':120.,'offset':0.,'beatTimes':list(np.arange(0,12,.5)),
        '_arcAudioAttention':attention}


def candidates(analysis, **overrides):
    config={'rank':9,'nps':7.,'mode':'Standard','style':'Powerhouse','arcAudioMode':'arcaudio3',
        'philosophy':{'movement':55,'stamina':75},**overrides}
    model={'doubles':np.full((3,16),.8)}
    contexts=(np.array([0.,12.]),np.ones((2,4)))
    return select_attention_candidates(analysis,config,np.random.default_rng(1),model,
        BeatClock(120),contexts)


class AttentionAccentContracts(unittest.TestCase):
    def test_fractional_phrase_boundary_retains_first_hit_and_plan_ownership(self):
        from engine.planning import build_map_plan
        analysis=fixture([1.23456,1.73456,2.23456,2.73456,3.23456,3.73456])
        analysis['beatTimes']=[.23456+i*.5 for i in range(24)]
        analysis['_arcAudioAttention']['phrases']=[{'start':1.23456,'end':3.23456},
            {'start':3.23456,'end':5.23456}]
        points,_,_=candidates(analysis,nps=4.)
        rows=analysis['_arcAudioMapperReport']['densityCurve']
        self.assertEqual(rows[0]['start'],1.23456)
        for row in rows:
            local=[p for p in points if row['start']<=p['time']<row['end']]
            self.assertEqual(len(local),row['chosenEvents'])
            self.assertEqual(len(_hit_times(local)),row['chosenHits'])
        plan=build_map_plan(analysis,{'arcAudioMode':'arcaudio3','learnWholeLevel':True},points,.12,BeatClock(120))
        self.assertTrue(all(s['compositionPrunedHits']==0 for s in plan['sections']))
        first=plan['sections'][0]
        self.assertEqual(first['selectedEvents'],4)
        self.assertTrue(first['start']<=points[0]['time']<first['end'])

    def test_whole_map_envelope_cannot_spend_active_note_budget_twice(self):
        from engine.generate import _composition_budget
        from engine.planning import build_map_plan
        analysis=fixture()
        points=[{'time':2+i*.5,'musicalBeat':4+i,'chord':i%2==0,
            'score':.9,'importance':.9,'targetNps':10.,'attentionState':'rhythm_focused',
            'section':'rhythm_focused','activity':1.,'source':'kick'} for i in range(4)]
        profile={'ready':True,'densityRatio':.08,'confidence':1.,'doubleFraction':0.,'matches':[]}
        config={'rank':9,'arcAudioMode':'arcaudio3','learnWholeLevel':True,'nps':10.}
        with patch('engine.composition.composition_profile',return_value=profile):
            selected,_,_=_composition_budget((points,.12,16.6),analysis,config)
        self.assertEqual(selected,points)
        self.assertEqual(len(_hit_times(selected)),6)
        summary=analysis['_arcForgeCompositionReport']['phrases'][0]
        self.assertEqual(summary['hitsBefore'],summary['hitsAfter'])
        analysis['_arcAudioMapperReport']={'densityCurve':[{'start':2.,'end':4.,
            'targetNps':10.,'chosenEvents':4,'chosenHits':6}]}
        plan=build_map_plan(analysis,config,selected,.12,BeatClock(120))
        self.assertEqual(plan['sections'][0]['selectedEvents'],4)
        self.assertEqual(plan['sections'][0]['selectedHits'],6)
        self.assertEqual(plan['sections'][0]['compositionPrunedHits'],0)

    def test_learned_quiet_envelope_still_preserves_only_a_foreground_anchor(self):
        from engine.generate import _composition_budget
        analysis=fixture()
        points=[{'time':2+i*.5,'musicalBeat':4+i,'chord':False,
            'score':.9,'importance':.9,'targetNps':1.,'attentionState':'vocal_focused',
            'section':'vocal_focused','activity':.1,'source':'vocals'} for i in range(4)]
        profile={'ready':True,'densityRatio':.08,'confidence':1.,'doubleFraction':0.,'matches':[]}
        with patch('engine.composition.composition_profile',return_value=profile):
            selected,_,_=_composition_budget((points,.12,16.6),analysis,
                {'rank':9,'arcAudioMode':'arcaudio3','philosophy':{'contrast':90}})
        self.assertEqual(len(selected),1)
        self.assertIn(selected[0],points)
        self.assertFalse(selected[0]['chord'])

    def test_strong_observed_accents_add_notes_without_changing_event_times(self):
        analysis=fixture();original=copy.deepcopy(analysis['_arcAudioAttention'])
        points,minimum,_=candidates(analysis)
        doubles=[p for p in points if p['chord']]
        self.assertGreater(len(doubles),0)
        self.assertLess(len(doubles),len(points)*.40)
        self.assertEqual([p['time'] for p in points],[e['time'] for e in original['selectedEvents']])
        self.assertEqual(analysis['_arcAudioAttention'],original)
        for point in doubles:
            self.assertEqual(point['accentDecision']['reason'],'learned_observed_accent')
            self.assertTrue(point['audioEvidence']['admission']['importancePassed'])
            self.assertEqual(point['audioEvidence']['eventTime'],point['time'])
        report=analysis['_arcAudioMapperReport']
        self.assertEqual(report['mapperCandidateNotes'],len(points)+len(doubles))
        for row in report['densityCurve']:
            self.assertLessEqual(row['chosenHits'],row['noteQuota'])
            self.assertAlmostEqual(row['actualCandidateNps'],row['chosenHits']/(row['end']-row['start']),places=3)
        times=sorted(_hit_times(points))
        self.assertTrue(all(t-times[i-2]>=minimum-.0001 for i,t in enumerate(times) if i>=2))

    def test_full_note_quota_never_becomes_a_double_overrun(self):
        analysis=fixture();points,_,_=candidates(analysis,nps=4.)
        self.assertTrue(points)
        rows=analysis['_arcAudioMapperReport']['densityCurve']
        self.assertTrue(all(r['accentDoubles']==0 for r in rows if r['chosenEvents']==r['noteQuota']))
        self.assertTrue(all(r['chosenHits']<=r['noteQuota'] for r in rows))

    def test_one_saber_and_quiet_voice_keep_single_source_space(self):
        for options,analysis in [({'mode':'OneSaber'},fixture()),
                                 ({},fixture(state='low_energy_vocal_space',source='vocals')),
                                 ({},fixture(state='vocal_focused',source='vocals'))]:
            with self.subTest(options=options,state=analysis['_arcAudioAttention']['frames'][0]['state']):
                points,_,_=candidates(analysis,**options)
                self.assertTrue(points)
                self.assertFalse(any(p['chord'] for p in points))

    def test_fast_alternating_attacks_do_not_leave_both_hands_for_doubles(self):
        analysis=fixture(np.arange(1.,11.,.08));points,minimum,_=candidates(analysis,nps=14.)
        self.assertGreater(len(points),100)
        self.assertFalse(any(p['chord'] for p in points))
        self.assertGreater(analysis['_arcAudioMapperReport']['accentRejections'].get('both_hands_need_recovery',0),0)
        times=sorted(_hit_times(points))
        self.assertTrue(all(t-times[i-2]>=minimum-.0001 for i,t in enumerate(times) if i>=2))

    def test_rejected_coincident_background_is_not_accent_evidence(self):
        # Offbeat low-salience attacks would only become doubles if the rejected
        # accompaniment were incorrectly treated as another foreground layer.
        analysis=fixture(np.arange(1.25,11.,.5),strength=.8)
        attention=analysis['_arcAudioAttention']
        for event in attention['selectedEvents']:
            event['coincidentSources']=['kick','piano']
            attention['events'].append({**event,'id':'rejected-'+event['id'],'source':'piano',
                'selected':False,'reason':'low_importance','strength':1.,
                'admission':{'importancePassed':False,'target':'supporting'}})
        points,_,_=candidates(analysis)
        self.assertFalse(any(p['chord'] for p in points))
        self.assertTrue(all(p['audioEvidence']['coincidentSources']==['kick','piano'] for p in points))
        self.assertTrue(all(p['audioEvidence']['eligibleCoincidentGroups']==['drums'] for p in points))

    def test_admitted_coincident_foreground_can_emphasize_an_offbeat(self):
        analysis=fixture(np.arange(1.25,11.,.5),strength=.8)
        attention=analysis['_arcAudioAttention']
        for event in list(attention['events']):
            attention['events'].append({**event,'id':'coincident-'+event['id'],'source':'melody',
                'selected':False,'reason':'redundant_simultaneous'})
        points,_,_=candidates(analysis)
        self.assertTrue(any(p['chord'] for p in points))
        self.assertTrue(all(p['audioEvidence']['eligibleCoincidentGroups']==['drums','melody'] for p in points))

    def test_missing_onset_evidence_cannot_be_invented_from_model_probability(self):
        for change in ({'strength':0.}, {'timingEvidence':'grid_estimate'},
                       {'sustained':True},{'admission':{'importancePassed':False}}):
            analysis=fixture()
            for event in analysis['_arcAudioAttention']['events']:event.update(change)
            points,_,_=candidates(analysis)
            self.assertFalse(any(p['chord'] for p in points),change)

    def test_lower_difficulty_and_stream_objective_reduce_learned_doubles(self):
        high,_,_=candidates(fixture())
        low,_,_=candidates(fixture(),rank=3)
        stream,_,_=candidates(fixture(),style='Stream')
        self.assertGreater(sum(p['chord'] for p in high),sum(p['chord'] for p in low))
        self.assertGreater(sum(p['chord'] for p in high),sum(p['chord'] for p in stream))

    def test_phrase_edge_doubles_do_not_consume_next_attack_recovery(self):
        analysis=fixture([1.,1.5,1.96,2.04,2.5,3.,3.5,4.,4.5,5.,5.5])
        points,minimum,_=candidates(analysis)
        edge=next(p for p in points if p['audioEvidence']['eventTime']==1.96)
        self.assertFalse(edge['chord'])
        times=sorted(_hit_times(points))
        self.assertTrue(all(t-times[i-2]>=minimum-.0001 for i,t in enumerate(times) if i>=2))

    def test_recurring_motifs_keep_identity_across_distinct_phrase_ids(self):
        analysis=fixture();attention=analysis['_arcAudioAttention']
        for index,phrase in enumerate(attention['phrases']):
            phrase.update(id=f'phrase-{index}',motifId=index%2,repeatConfidence=.96)
        for event in attention['events']:
            event['phraseId']=f"phrase-{int(event['time']//2)}"
        points,_,_=candidates(analysis)
        a=next(p for p in points if p['time']==2.)
        repeated=next(p for p in points if p['time']==6.)
        other=next(p for p in points if p['time']==4.)
        self.assertEqual(a['audioMotif'],repeated['audioMotif'])
        self.assertNotEqual(a['audioEvidence']['phraseId'],repeated['audioEvidence']['phraseId'])
        self.assertNotEqual(a['audioMotif'],other['audioMotif'])
        self.assertEqual(repeated['audioEvidence']['repeatConfidence'],.96)

    def test_hybrid_grid_support_cannot_smuggle_a_legacy_double(self):
        analysis=fixture(strength=.1)
        attention=analysis['_arcAudioAttention']
        attention['phrases'][0].update(id='first',motifId=7,repeatConfidence=.95)
        for event in attention['events']:event['phraseId']='first'
        legacy=[{'time':e['time']+.02,'beat':2*(e['time']+.02),
            'musicalBeat':2*(e['time']+.02),'context':np.ones(4),'chord':True,
            'role':'lowPercussion','score':100.,'audioMotif':'unrelated-legacy-motif'}
            for e in attention['selectedEvents']]
        contexts=(np.array([0.,12.]),np.ones((2,4)))
        config={'rank':9,'nps':7.,'style':'Powerhouse','arcAudioMode':'hybrid'}
        points,_,_=select_attention_candidates(analysis,config,np.random.default_rng(1),
            {'doubles':np.ones((3,16))},BeatClock(120),contexts,legacy)
        self.assertTrue(points)
        self.assertFalse(any(p['chord'] for p in points))
        self.assertEqual({p['audioMotif'] for p in points},{'attention-motif-7:drums'})
        self.assertTrue(all(p['audioEvidence']['eventId'].startswith('observed-') for p in points))

    def test_real_placement_and_native_codec_preserve_double_provenance(self):
        # A tiny isolated network checks the actual placement/format contract;
        # it is neither a trained checkpoint nor a claim of mapping quality.
        from engine.generate import generate_map
        from engine.formats import encode_v4, decode_v4
        analysis=fixture()
        config={'rank':9,'nps':7.,'mode':'Standard','style':'Powerhouse','arcAudioMode':'arcaudio',
            'backend':'local','seed':71,'philosophy':{'movement':55,'stamina':75}}
        model={'mean':np.zeros(44),'std':np.ones(44),'w1':np.zeros((44,4)),'b1':np.zeros(4),
            'w2':np.zeros((4,108)),'b2':np.zeros(108),'rhythm':np.ones((3,16,2)),
            'doubles':np.full((3,16),.8)}
        profile={'prior':np.ones(108)/108,'goal':[.8,.5,.6,.65],'matches':[],'method':'Isolated fixture'}
        artifacts=ROOT/'data/test-artifacts';artifacts.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as folder,patch('engine.styles.style_profile',return_value=profile):
            np.savez(pathlib.Path(folder)/'model.npz',**model)
            generated=generate_map(analysis,config,folder)
        notes=generated['beatmap']['colorNotes']
        double_notes=[n for n in notes if n['customData']['arcforgeAccent']['double']]
        self.assertGreater(len(double_notes),0)
        self.assertEqual(len(notes),generated['report']['attention']['mapperCandidateNotes'])
        self.assertTrue(all(n['customData']['arcforgeEventId'].startswith('observed-') for n in notes))
        packed=encode_v4(generated['beatmap']);restored=decode_v4(packed['difficulty'],packed['lightshow'])
        self.assertEqual([n['customData'] for n in restored['colorNotes']],[n['customData'] for n in notes])
        for beat in {n['b'] for n in double_notes}:
            self.assertEqual({n['c'] for n in double_notes if n['b']==beat},{0,1})


if __name__=='__main__':unittest.main()
