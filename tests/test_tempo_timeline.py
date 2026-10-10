import pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap
import numpy as np
from engine.tempo import analyze_tempo


def fixture(tempos,durations):
    times=np.arange(0,sum(durations),.01);envelope=np.zeros(len(times));onsets=[];start=0.
    for bpm,duration in zip(tempos,durations):
        for t in np.arange(start,start+duration,60/bpm):
            envelope+=np.exp(-.5*((times-t)/.016)**2);onsets.append(t)
        start+=duration
    return times,envelope,onsets


class TempoTimeline(unittest.TestCase):
    def test_real_acceleration_and_deceleration(self):
        for tempos in [[120,180],[180,120],[120,180,90]]:
            with self.subTest(tempos=tempos):
                times,env,onsets=fixture(tempos,[24]*len(tempos))
                result=analyze_tempo(times,env,tempos[0],0,times[-1],onsets)
                self.assertTrue(result['changed'])
                for second,wanted in zip(np.arange(len(tempos))*24+12,tempos):
                    segment=next(s for s in result['timeline'] if s['start']<=second<s['end'])
                    self.assertLess(abs(segment['bpm']-wanted),wanted*.025)
                self.assertTrue(all(b>a for a,b in zip(result['beatTimes'],result['beatTimes'][1:])))
                distance=[min(abs(t-o) for o in onsets) for t in result['beatTimes']]
                self.assertLess(np.percentile(distance,90),.025)

    def test_stable_clock_and_silence_preserve_existing_beats(self):
        times,env,onsets=fixture([190],[48])
        for candidate in [env,np.zeros(len(env))]:
            result=analyze_tempo(times,candidate,190,0,times[-1],onsets)
            self.assertFalse(result['changed']);self.assertEqual(result['beatTimes'],onsets)

    def test_isolated_fill_does_not_change_meter(self):
        times,env,onsets=fixture([120,180,120],[23,2,23])
        result=analyze_tempo(times,env,120,0,times[-1],onsets)
        self.assertFalse(result['changed'])

    def test_percussion_subdivisions_do_not_rewrite_an_established_meter(self):
        for bpm in [120,190,235]:
            with self.subTest(bpm=bpm):
                times,env,onsets=fixture([bpm,bpm*2,bpm/2],[24,24,24])
                existing=np.arange(0,times[-1],60/bpm).tolist()
                result=analyze_tempo(times,env,bpm,0,times[-1],existing)
                self.assertFalse(result['changed'])
                self.assertEqual(result['beatTimes'],existing)
                # Attack-rate hypotheses remain observable for intensity; their
                # existence does not establish a change in musical meter.
                if bpm*2<=400:
                    self.assertTrue(any(abs(e['rawBpm']/bpm-2)<.04 for e in result['estimates']))

    def test_gradual_tempo_uses_measured_attacks_without_large_phase_drift(self):
        duration=96.;times=np.arange(0,duration,.01);env=np.zeros(len(times));onsets=[];second=0.
        while second<duration:
            onsets.append(second);env+=np.exp(-.5*((times-second)/.016)**2)
            second+=60/(120+60*second/duration)
        result=analyze_tempo(times,env,120,0,duration,onsets)
        self.assertTrue(result['changed'])
        self.assertGreater(result['timeline'][-1]['bpm'],170)
        distances=[min(abs(t-o) for o in onsets) for t in result['beatTimes']]
        self.assertLess(np.percentile(distances,95),.02)
        self.assertTrue(all(b>a for a,b in zip(result['beatTimes'],result['beatTimes'][1:])))

    def test_nonfinite_or_unordered_evidence_never_changes_saved_grid(self):
        for times,env in [(np.arange(40)[::-1],np.ones(40)),(np.arange(40),np.full(40,float('nan')))]:
            result=analyze_tempo(times,env,120,0,40,[0,.5,1.])
            self.assertFalse(result['changed']);self.assertEqual(result['beatTimes'],[0,.5,1.])

    def test_brief_periodic_bursts_during_silence_do_not_establish_new_meter(self):
        times=np.arange(0,72,.01);env=np.zeros(len(times));existing=np.arange(0,72,.5).tolist()
        for start,end,bpm in [(0,24,120),(30,32,180),(44,46,180),(58,60,180)]:
            for second in np.arange(start,end,60/bpm):env+=np.exp(-.5*((times-second)/.016)**2)
        result=analyze_tempo(times,env,120,0,72,existing)
        self.assertFalse(result['changed']);self.assertEqual(result['beatTimes'],existing)


if __name__=='__main__':unittest.main()
