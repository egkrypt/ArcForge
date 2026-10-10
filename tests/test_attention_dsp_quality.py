"""Known-signal source evidence tests, without claiming instrument accuracy.

Pitched attacks and broadband transients are independently constructed here.
These fixtures verify harmonic/percussive routing, not singer/piano identity.
"""
import collections, copy, pathlib, sys, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap
import numpy as np
from scipy.io import wavfile
from engine.music import analyze_music
from engine.arcaudio.sources import extract_sources, _spectral_features, SOURCES
from engine.arcaudio.hybrid import fuse_legacy


def fixture(kind):
    sr=22050;duration=12.;times=np.arange(int(sr*duration))/sr
    samples=np.zeros(len(times),np.float32);rng=np.random.default_rng(41)
    for index,second in enumerate(np.arange(.5,11.5,.5)):
        start=int(second*sr);local=times[:int(sr*.42)]
        frequency=220*2**([0,3,7,10][index%4]/12)
        pitched=sum(np.sin(2*np.pi*frequency*h*local)/h for h in [1,2,3,4])*np.exp(-local*7)
        noise=rng.normal(size=len(local))*np.exp(-local*85)
        value=pitched if kind=='pitched' else noise if kind=='noise' else pitched+.8*noise if kind=='mixed' else np.zeros(len(local))
        samples[start:start+len(local)]+=np.asarray(value*.25,np.float32)
    grid=np.arange(0,duration,.5).tolist()
    music=analyze_music(samples,sr,120,0,grid)
    with tempfile.TemporaryDirectory(dir=ROOT/'data') as directory:
        audio=pathlib.Path(directory)/'controlled.wav';wavfile.write(audio,sr,samples)
        return extract_sources({'duration':duration,'bpm':120,'offset':0,'audioPath':str(audio),
            'frames':[],'beatTimes':grid,'musicAnalysis':music})


class SourceEvidenceQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.results={kind:fixture(kind) for kind in ['pitched','noise','mixed','silence']}

    def test_pitched_transients_do_not_become_three_drum_streams(self):
        evidence=self.results['pitched'];events=evidence['events']
        drums=sum(e['source'] in ['kick','snare','percussion'] for e in events)
        pitched=sum(e['source'] in ['bass','melody','piano'] for e in events)
        self.assertGreater(pitched,50);self.assertLess(drums,pitched*.08)
        rerouted=[e for e in events if e.get('originalRoles')]
        self.assertTrue(rerouted)
        self.assertTrue(all(e['componentEvidence']['harmonic']>.56 for e in rerouted))
        # Names remain explicitly uncertain proxies, never verified stems.
        self.assertTrue(evidence['limitations'])
        self.assertLessEqual(max(f['sourceConfidence']['piano'] for f in evidence['frames']),.27)

    def test_broadband_transients_retain_rhythmic_evidence(self):
        evidence=self.results['noise'];frames=evidence['frames'][3:-3]
        self.assertGreater(sum(e['source'] in ['kick','snare','percussion'] for e in evidence['events']),50)
        # Quiet residual tails have little musical weight. Weight by measured
        # energy rather than letting near-silent windows dominate this check.
        self.assertGreater(np.average([f['componentEvidence']['percussiveFraction'] for f in frames],
            weights=[f['energy'] for f in frames]),.95)
        # Source strength must reflect duty cycle; two brief hits per second
        # must not force continuously high drum importance between those hits.
        self.assertGreater(np.average([f['sources']['drums'] for f in frames],weights=[f['energy'] for f in frames]),
            np.average([f['sources']['melody'] for f in frames],weights=[f['energy'] for f in frames]))
        self.assertLess(np.mean([f['sources']['melody'] for f in frames]),.02)

    def test_overlapping_components_are_preserved_before_attention(self):
        events=self.results['mixed']['events'];overlapping=0
        for second in np.arange(.5,11.5,.5):
            nearby=[e for e in events if abs(e['time']-second)<.045]
            overlapping+=int(any(e['source'] in ['kick','snare','percussion'] for e in nearby) and
                any(e['source'] in ['bass','melody','piano'] for e in nearby))
        self.assertGreaterEqual(overlapping,16)

    def test_absolute_silence_has_no_source_activity_or_events(self):
        evidence=self.results['silence']
        self.assertTrue(all(all(value==0 for value in f['sources'].values()) for f in evidence['frames']))
        self.assertFalse(evidence['events'])

    def test_hybrid_adds_only_observed_supported_legacy_attacks(self):
        sources={s:0. for s in SOURCES};sources.update(melody=.85)
        frame={'time':0.,'sources':sources,'sourceConfidence':{s:.8 for s in SOURCES},'energy':.4,
            'componentEvidence':{'harmonicFraction':.9}}
        evidence={'frames':[frame,{**frame,'time':2.}],
            'events':[{'time':.5,'source':'melody','strength':.8}],'method':'controlled'}
        original=copy.deepcopy(evidence)
        analysis={'onsets':[.5,.7,1.2],'beatTimes':[0.,.5,1.,1.5,2.],
            'frames':[{'time':0.,'onset':.8},{'time':2.,'onset':.8}]}
        result,added=fuse_legacy(analysis,evidence)
        self.assertEqual(evidence,original);self.assertEqual(added,2)
        additions=[e for e in result['events'] if e.get('evidence')=='legacy_attack_with_spectral_foreground']
        self.assertEqual([e['time'] for e in additions],[.7,1.2])
        self.assertTrue(all(e['source']=='melody' and e['timingEvidence']=='observed_onset' for e in additions))
        # A quiet/unsupported onset must not become a fabricated beat candidate.
        silent=copy.deepcopy(evidence)
        for f in silent['frames']:f['energy']=0.
        self.assertEqual(fuse_legacy(analysis,silent)[1],0)

    def test_fast_transient_activity_does_not_depend_on_context_sample_phase(self):
        sr=22050;duration=8.;centers=np.arange(0,duration,.25);means=[]
        for phase in [0.,.0625]:
            samples=np.zeros(int(sr*duration),np.float32);rng=np.random.default_rng(417)
            for second in np.arange(.5+phase,7.5,.125):
                start=int(second*sr);local=np.arange(int(sr*.06))/sr
                samples[start:start+len(local)]+=np.asarray(.3*rng.normal(size=len(local))*np.exp(-local*140),np.float32)
            measured,_=_spectral_features(samples,sr,centers)
            means.append({name:float(np.mean(measured[name][4:-4])) for name in ['energy','drums']})
        self.assertGreater(min(m['energy'] for m in means),.1)
        self.assertGreater(min(m['drums'] for m in means),.1)
        for name in ['energy','drums']:
            self.assertLess(abs(means[0][name]-means[1][name])/max(means[0][name],means[1][name]),.15)


if __name__=='__main__':unittest.main(verbosity=2)
