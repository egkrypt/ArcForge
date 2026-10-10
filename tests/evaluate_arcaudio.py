"""Repeatable attention evaluation with controlled source evidence and real audio.

Run from ArcForge: python tests/evaluate_arcaudio.py [--actual AUDIO_ID ...]
Output data/arcaudio-evaluation.json. Ground-truth source recognition remains
explicitly unmeasured without stems/annotations; attention is evaluated separately.
"""
import argparse, json, pathlib, sys, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
import bootstrap
import numpy as np
from test_arcaudio import controlled, scenario_result, SCENARIOS
from engine.arcaudio import analyze_attention, analyze_evidence
from engine.arcaudio.phrases import build_phrases


def brief(result):
    return {'candidateEvents': result['summary']['candidateEvents'], 'selectedEvents': result['summary']['selectedEvents'],
            'eventReduction': result['summary']['eventReduction'], 'sourceSwitches': result['summary']['sourceSwitches'],
            'meanConfidence': result['summary']['meanConfidence'], 'meanDensityTarget': result['summary']['meanDensityTarget'],
            'rejectionReasons': result['summary']['rejectionReasons'], 'selectedSourceCounts': result['summary']['selectedSourceCounts'],
            'analysisSeconds': result['performance']['analysisSeconds']}


def evaluate(actual_ids=None):
    expected = {'quiet_vocal_piano': 'vocals', 'vocal_full_drums': 'drums', 'instrumental_drop': 'drums',
                'dense_chorus': 'drums', 'spoken_section': 'spoken', 'piano_only': 'piano', 'drum_only': 'drums',
                'high_energy_instrumental': 'melody', 'competing_instruments': 'drums', 'sparse_arrangement': 'melody'}
    cases = []
    for name in SCENARIOS:
        result = scenario_result(name)
        cases.append({'case': name, 'evidence': 'Known, explicitly supplied source strengths/onsets; tests attention, not source classification',
                      **brief(result), 'expectedPrimary': expected[name],
                      'primarySelectionAgreement': round(sum(f['primarySource'] == expected[name] for f in result['frames']) / len(result['frames']), 4)})
    first, a, _, beats = controlled({'vocals': .95, 'piano': .25}, .2, .2, event_rates={'vocals': 2.})
    second, b, _, _ = controlled({'drums': .95, 'bass': .8}, .9, .9, event_rates={'drums': 8.})
    frames = first[:16] + [{**f, 'time': f['time'] + 4} for f in second[:16]]
    events = [e for e in a if e['time'] < 4] + [{**e, 'time': e['time'] + 4} for e in b if e['time'] < 4]
    transition = analyze_evidence(frames, events, beat_times=beats)
    first_drum = next((f['time'] for f in transition['frames'] if f['time'] >= 4 and f['primarySource'] == 'drums'), None)
    cases.append({'case': 'sudden_transition', **brief(transition), 'transitionTime': 4.,
                  'targetChangeLatencySeconds': round(first_drum - 4, 5) if first_drum is not None else None,
                  'transitions': transition['transitions']})
    repeated_frames, _, _, grid = controlled({'piano': .9}, .5, .35, duration=9.6)
    repeated_events = [{'time': round(bar + offset, 5), 'source': 'piano', 'strength': .9, 'confidence': .95}
                       for bar in np.arange(0, 9.6, 1.2) for offset in [.1, .4, .7, 1.]]
    phrases = build_phrases({'duration': 9.6, 'bpm': 200, 'beatTimes': grid}, repeated_frames, repeated_events)
    repeated = analyze_evidence(repeated_frames, repeated_events, phrases)
    cases.append({'case': 'repeated_phrases', **brief(repeated), 'phraseCount': len(phrases),
                  'distinctMotifs': len(set(p['motifId'] for p in phrases)), 'repeatedPhrases': repeated['summary']['repeatedPhrases']})
    real = []
    for identity in actual_ids or []:
        source = ROOT / 'data/audio' / identity / 'analysis.json'
        if not source.is_file(): raise ValueError('Unknown local audio ID: ' + identity)
        analysis = json.loads(source.read_text(encoding='utf-8'))
        result = analyze_attention(analysis)
        warm = analyze_attention(analysis, {'vocalPriority': .75})
        selected = result['selectedEvents']; legacy_onsets = np.asarray(analysis.get('onsets', []))
        distances = [float(np.min(np.abs(legacy_onsets - e['time']))) for e in selected] if len(legacy_onsets) else []
        states = result['summary']['states']
        real.append({'audioId': identity, 'name': analysis['name'], 'duration': analysis['duration'],
                     **brief(result), 'states': states, 'sourceDetectionAccuracy': None,
                     'sourceDetectionAccuracyReason': 'No annotated stems/source labels for this recording',
                     'medianDistanceToLegacyObservedOnsetMs': round(float(np.median(distances)) * 1000, 2) if distances else None,
                     'warmSourceAnalysisSeconds': warm['performance']['sourceAnalysisSeconds'],
                     'limitations': result['limitations']})
    return {'version': '2.0', 'controlledAttentionCases': cases, 'actualRecordings': real,
            'methodology': {'sourceSelection': 'Agreement against controlled source-evidence fixtures, independent of mixed-audio source classification',
                            'timing': 'Selected events retain observed timestamps; actual-song distances to legacy onsets are consistency evidence, not human sync ratings',
                            'sourceDetection': 'Unmeasured for real instruments without labeled recordings/stems',
                            'performance': 'Measured wall clock on this local host; cached retuning measured separately'},
            'testCommand': 'python tests/test_arcaudio.py', 'caseCount': len(cases)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--actual', nargs='*', default=[]); args = parser.parse_args()
    result = evaluate(args.actual)
    destination = ROOT / 'data/arcaudio-evaluation.json'
    destination.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'caseCount': result['caseCount'], 'controlledSelection': {c['case']: c.get('primarySelectionAgreement', c.get('targetChangeLatencySeconds', c.get('distinctMotifs'))) for c in result['controlledAttentionCases']},
                      'actual': [{k: r[k] for k in ['name', 'candidateEvents', 'selectedEvents', 'eventReduction', 'analysisSeconds', 'warmSourceAnalysisSeconds', 'selectedSourceCounts']} for r in result['actualRecordings']], 'report': str(destination)}, indent=2))
