"""Opt-in attention/guidance integrates without changing the legacy maps."""
import copy
import hashlib
import json
import pathlib
import sys
import unittest
from unittest.mock import patch
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
import bootstrap
import numpy as np
from test_music_mapping import fixture
from engine.generate import generate_map, _candidate_times, _mapping_clock


def attended_fixture():
    analysis = fixture()
    times = [2.113, 2.691, 3.207, 8.456, 8.912, 9.384, 9.931]
    frames = [{'time': t, 'state': 'VOCAL_FOCUSED', 'primarySource': 'vocals',
               'rhythmDensity': .6, 'densityTarget': 1., 'confidence': 1.}
              for t in np.arange(0, analysis['duration'], .25)]
    events = [{'time': t, 'source': 'vocals', 'importance': .91, 'selected': True,
               'phraseId': 'vocal-response-phrase-A'} for t in times]
    analysis['_arcAudioAttention'] = {'frames': frames, 'events': events, 'selectedEvents': events,
                                    'summary': {'independentEvidence': True}, 'limitations': []}
    return analysis, times


class GeneratorAttentionIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = fixture()
        with np.load(ROOT / 'models/style/model.npz') as saved:
            cls.model = {key: saved[key].copy() for key in saved.files}

    def test_default_off_and_legacy_are_identical_with_current_trained_model(self):
        # User training can replace model.npz independently of implementation.
        # Preserve the historical golden hashes, and compare all disabled routes
        # against default generation using exactly the same current checkpoint.
        golden = json.loads((ROOT / 'data/legacy-behavior-v11.json').read_text('utf-8'))
        for row in golden:
            baseline=generate_map(self.analysis,dict(row['config']),ROOT/'models/style')['beatmap']
            baseline_digest=hashlib.sha256(json.dumps(baseline,sort_keys=True).encode()).hexdigest()
            for mode in (None, 'off', 'legacy'):
                config = dict(row['config'])
                if mode is not None: config['arcAudioMode'] = mode
                result = generate_map(self.analysis, config, ROOT / 'models/style')
                digest = hashlib.sha256(json.dumps(result['beatmap'], sort_keys=True).encode()).hexdigest()
                self.assertEqual(digest, baseline_digest, (config['mode'], mode))
                # The historical artifact remains a before-refactor reference.
                # Recovery-aware selection can retain supported opportunities
                # that the old blanket subdivision reduction discarded.
                self.assertTrue(result['report']['passed'])
                self.assertGreater(len(result['beatmap']['colorNotes']), 0)
                self.assertNotIn('attention', result['report'])
                self.assertNotIn('regionReport', result['report'])

    def test_off_does_not_run_source_attention(self):
        with patch('engine.attention_candidates.select_attention_candidates', side_effect=AssertionError('Attention ran while disabled')):
            generated = generate_map(self.analysis, {'nps': 10, 'rank': 9, 'backend': 'local',
                'seed': 87, 'style': 'Flow', 'arcAudioMode': 'off'}, ROOT / 'models/style')
        self.assertTrue(generated['report']['passed'])

    def test_each_attention_engine_uses_one_active_phrase_budget(self):
        analysis=copy.deepcopy(self.analysis)
        times=[round(float(t),5) for t in np.arange(8,16,.08)]
        frames=[{'time':float(t),'state':'rhythm_focused','primarySource':'drums','rhythmDensity':1.,
            'densityTarget':.35,'confidence':.4} for t in np.arange(0,analysis['duration'],.25)]
        events=[{'time':t,'source':'kick','importance':.9,'selected':i%3==0,
            'reason':'primary_event' if i%3==0 else 'attention_budget','phraseId':'drop'} for i,t in enumerate(times)]
        analysis['_arcAudioAttention']={'version':'3.0','frames':frames,'events':events,
            'selectedEvents':[e for e in events if e['selected']],'settings':{'maxDensity':1.}}
        from engine.attention_candidates import select_attention_candidates
        from engine.generate import _contexts
        for mode in ('arcaudio','hybrid','arcaudio3'):
            with self.subTest(mode=mode):
                config={'bpm':190,'nps':10,'rank':9,'seed':83,'backend':'local','style':'Modern','arcAudioMode':mode}
                clock,_=_mapping_clock(analysis,config)
                candidates,_,_=select_attention_candidates(analysis,config,np.random.default_rng(83),self.model,clock,_contexts(analysis),legacy=None)
                self.assertGreater(sum(8.5<=p['time']<15.5 for p in candidates)/7,9.)
                self.assertTrue(all(min(abs(p['time']-t) for t in times)<.019 for p in candidates))
                self.assertGreater(analysis['_arcAudioMapperReport']['eligibleForegroundEvents'],analysis['_arcAudioMapperReport']['selectedByAttention'])

    def test_budget_reconsideration_preserves_quiet_voice_and_rejections(self):
        from engine.attention_candidates import select_attention_candidates
        from engine.generate import _contexts
        analysis,times=attended_fixture()
        rejected=[{'time':float(t),'source':'piano','importance':.99,'selected':False,
            'reason':['supporting_layer','low_importance','sustained_background','redundant_simultaneous'][i%4]}
            for i,t in enumerate(np.arange(2,10,.05))]
        analysis['_arcAudioAttention']['events'].extend(rejected)
        analysis['_arcAudioAttention']['settings']={'maxDensity':.6}
        for mode in ('arcaudio','hybrid','arcaudio3'):
            config={'nps':12,'rank':9,'arcAudioMode':mode,'style':'Modern'}
            clock,_=_mapping_clock(analysis,config)
            candidates,_,_=select_attention_candidates(analysis,config,np.random.default_rng(1),self.model,clock,_contexts(analysis))
            self.assertEqual([p['time'] for p in candidates],times)
            self.assertTrue(all(p['source']=='vocals' for p in candidates))
            self.assertTrue(all(r['targetNps']<=7.2 for r in analysis['_arcAudioMapperReport']['densityCurve']))

    def test_attention_uses_observed_vocal_times_not_an_invented_grid(self):
        analysis, expected = attended_fixture()
        config = {'nps': 12, 'rank': 9, 'backend': 'local', 'seed': 83, 'style': 'Flow', 'arcAudioMode': 'arcaudio'}
        clock, _ = _mapping_clock(analysis, config)
        candidates, _, _ = _candidate_times(analysis, config, np.random.default_rng(83), self.model)
        self.assertEqual([p['time'] for p in candidates], expected)
        self.assertTrue(all(p['audioMotif'] == 'vocal-response-phrase-A' for p in candidates))
        result = generate_map(analysis, config, ROOT / 'models/style')
        actual = [clock.time(n['b']) for n in result['beatmap']['colorNotes']]
        self.assertEqual(len(actual), len(expected))
        self.assertTrue(all(abs(a - b) < .00001 for a, b in zip(actual, expected)))
        self.assertEqual(result['report']['attention']['mode'], 'arcaudio')
        self.assertTrue(all(n['customData']['arcforgeSource'] == 'vocals' for n in result['beatmap']['colorNotes']))

    def test_v3_density_windows_follow_measured_tempo_and_guided_boundaries(self):
        from engine.attention_candidates import select_attention_candidates
        from engine.generate import _contexts
        analysis=copy.deepcopy(self.analysis)
        # Slow four-beat phrases, then twice the tempo; no window should span
        # across the explicit 10.3-second creative-region boundary.
        analysis['offset']=0.
        analysis['beatTimes']=list(np.arange(0,8,.5))+list(np.arange(8,24,.25))
        times=[round(float(t),5) for t in np.arange(.6,23.6,.08)]
        frames=[{'time':float(t),'state':'rhythm_focused','primarySource':'drums',
            'rhythmDensity':1.,'densityTarget':1.,'confidence':1.} for t in np.arange(0,24,.25)]
        events=[{'time':t,'source':'kick','importance':.9,'selected':True} for t in times]
        analysis['_arcAudioAttention']={'version':'3.0','frames':frames,'events':events,'selectedEvents':events}
        config={'nps':6,'rank':9,'arcAudioMode':'arcaudio3','dynamicDensity':False,
            'mappingRegions':[{'start':10.3,'end':11.7,'primaryStyle':'Flow','nps':4}]}
        clock,_=_mapping_clock(analysis,config)
        candidates,_,_=select_attention_candidates(analysis,config,np.random.default_rng(1),self.model,clock,_contexts(analysis))
        rows=analysis['_arcAudioMapperReport']['densityCurve']
        self.assertIn((2.,4.),[(r['start'],r['end']) for r in rows])
        self.assertIn((8.,9.),[(r['start'],r['end']) for r in rows])
        self.assertTrue(any(r['end']==10.3 for r in rows))
        self.assertTrue(any(r['start']==11.7 for r in rows))
        self.assertFalse(any(r['start']<10.3<r['end'] for r in rows))
        self.assertTrue(all(abs(p['time']-min(times,key=lambda t:abs(t-p['time'])))<=.01801 for p in candidates))

    def test_hybrid_does_not_union_all_legacy_background_hits(self):
        analysis, selected = attended_fixture()
        config = {'nps': 12, 'rank': 9, 'backend': 'local', 'seed': 84, 'style': 'Flow', 'arcAudioMode': 'hybrid'}
        candidates, _, _ = _candidate_times(analysis, config, np.random.default_rng(84), self.model)
        self.assertTrue(candidates)
        self.assertTrue(all(min(abs(p['time'] - t) for t in selected) <= .035 for p in candidates))
        self.assertLess(len(candidates), 12)

    def test_legacy_fusion_cannot_bypass_importance_or_source_compatibility(self):
        from engine.attention_candidates import select_attention_candidates
        from engine.generate import _contexts
        analysis,times=attended_fixture()
        for frame in analysis['_arcAudioAttention']['frames']:
            frame.update(primarySource='bass',confidence=1.)
        for event in analysis['_arcAudioAttention']['events']:event.update(source='bass')
        legacy=[{'time':t+.02,'beat':t*190/60,'musicalBeat':t*190/60,'context':np.zeros(7),
            'chord':False,'role':'highPercussion','score':100.,'timingError':.01} for t in times]
        for mode in ('hybrid','arcaudio3'):
            config={'nps':12,'rank':9,'arcAudioMode':mode,'style':'Modern'}
            clock,_=_mapping_clock(analysis,config)
            candidates,_,_=select_attention_candidates(analysis,config,np.random.default_rng(1),self.model,clock,_contexts(analysis),legacy)
            self.assertEqual([p['time'] for p in candidates],times)
            for event in analysis['_arcAudioAttention']['events']:event.update(selected=False,reason='low_importance')
            analysis['_arcAudioAttention']['selectedEvents']=[]
            candidates,_,_=select_attention_candidates(analysis,config,np.random.default_rng(1),self.model,clock,_contexts(analysis),legacy)
            self.assertEqual(candidates,[])
            for event in analysis['_arcAudioAttention']['events']:event.update(selected=True,reason='primary_event')
            analysis['_arcAudioAttention']['selectedEvents']=analysis['_arcAudioAttention']['events']

    def test_importance_setting_changes_real_analysis_to_map_output(self):
        # V3 requires original waveform recognition; the isolated fixture has
        # spectral analysis only. Real V3 audio is covered by engine comparison.
        for mode in ('arcaudio','hybrid'):
            config={'nps':10,'rank':9,'arcAudioMode':mode,'style':'Modern','backend':'local','seed':84}
            low=generate_map(self.analysis,config|{'arcAudioSettings':{'minImportance':.1}},ROOT/'models/style')
            high=generate_map(self.analysis,config|{'arcAudioSettings':{'minImportance':.95}},ROOT/'models/style')
            self.assertGreater(len(low['beatmap']['colorNotes']),len(high['beatmap']['colorNotes']),mode)

    def test_regions_change_supported_density_without_creating_new_times(self):
        analysis = copy.deepcopy(self.analysis)
        config = {'nps': 12, 'rank': 9, 'backend': 'local', 'seed': 91, 'style': 'Flow', 'mappingRegions': [
            {'id': 'light', 'name': 'Readable riff', 'start': 8, 'end': 12, 'primaryStyle': 'Light / Easy',
             'nps': 2, 'overrides': {'movement': 0, 'tech': 0}},
            {'id': 'challenge', 'name': 'Response', 'start': 12, 'end': 16, 'primaryStyle': 'Challenge',
             'secondaryStyles': ['Powerhouse'], 'nps': 12, 'overrides': {'movement': 85, 'tech': 90}}]}
        result = generate_map(analysis, config, ROOT / 'models/style')
        clock, _ = _mapping_clock(analysis, config)
        times = np.array([clock.time(n['b']) for n in result['beatmap']['colorNotes']])
        light = int(((times >= 8) & (times < 12)).sum())
        hard = int(((times >= 12) & (times < 16)).sum())
        self.assertGreater(hard, light * 3)
        grid = np.interp(np.arange((len(analysis['beatTimes']) - 1) * 4) / 4,
                         np.arange(len(analysis['beatTimes'])), analysis['beatTimes'])
        attacks=np.array([a['time'] for a in analysis['musicAnalysis']['accents']])
        self.assertTrue(all(min(float(np.min(abs(grid-t))),float(np.min(abs(attacks-t))))<.00001 for t in times))
        rows = result['report']['regionReport']['regions']
        self.assertEqual(rows[0]['goals']['movement'], 0)
        self.assertEqual(rows[1]['goals']['movement'], 85)
        self.assertEqual(rows[1]['secondaryStyles'], ['Powerhouse'])
        self.assertEqual(result['report']['regionReport']['styleArchetypes'], 3)
        self.assertTrue(result['report']['passed'])

    def test_short_region_quota_uses_exact_boundary_not_bar_midpoint(self):
        config = {'nps': 12, 'rank': 9, 'backend': 'local', 'seed': 92, 'style': 'Flow', 'mappingRegions': [
            {'start': 10.1, 'end': 10.5, 'primaryStyle': 'Light/easy', 'nps': .25}]}
        result = generate_map(self.analysis, config, ROOT / 'models/style')
        clock, _ = _mapping_clock(self.analysis, config)
        times = [clock.time(n['b']) for n in result['beatmap']['colorNotes']]
        self.assertFalse(any(10.1 <= t < 10.5 for t in times))
        self.assertTrue(any(9.8 <= t < 10.1 for t in times))
        self.assertTrue(any(10.5 <= t < 10.8 for t in times))

    def test_invalid_overlap_is_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            generate_map(self.analysis, {'nps': 10, 'mappingRegions': [
                {'start': 2, 'end': 5}, {'start': 4, 'end': 6}]}, ROOT / 'models/style')


if __name__ == '__main__': unittest.main()
