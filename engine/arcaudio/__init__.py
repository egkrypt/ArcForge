"""ArcAudio 2.0: opt-in attention between audio detection and map candidates.

This package never mutates a map or the legacy analysis. A downstream mapper
consumes selectedEvents and frame density goals using its existing hand guards.
"""
from __future__ import annotations
import collections, copy, time
from statistics import median
from .settings import VERSION, DEFAULTS, normalize_settings, canonical_mode
from .sources import extract_sources
from .phrases import build_phrases, detect_transitions
from .attention import calculate_attention
from .events import select_events
from .supervision import export_human_supervision
from .focus import normalize_focus_regions, focused_frames, focus_summary


def analyze_evidence(frames, events, phrases=None, settings=None, beat_times=None,
                     mapping_regions=None, focus_regions=None, duration=None):
    """Pure decision layer for independently annotated/separated-source fixtures."""
    started = time.perf_counter(); settings = normalize_settings(settings)
    if duration is None:
        duration = max([float(f.get('time', 0)) for f in frames] +
                       [float(r.get('end', 0)) for r in (mapping_regions or []) + (focus_regions or [])] + [.001]) + .001
    regions = normalize_focus_regions(mapping_regions, focus_regions, duration)
    decided = calculate_attention(frames, events, phrases, settings)
    decided = focused_frames(decided, events, regions)
    selected = select_events(decided, events, phrases, settings, beat_times, regions)
    transitions = detect_transitions(decided, phrases or [])
    result = _result(decided, selected, phrases or [], transitions, settings,
                     'supplied_temporal_evidence', [], time.perf_counter() - started)
    if regions:
        result.update(mappingRegions=regions, mappingFocus=focus_summary(regions, selected))
    return result


def analyze_attention(analysis, settings=None, mode='arcaudio', compute=None,
                      mapping_regions=None, focus_regions=None):
    """Analyze a private legacy audio record with optional original audioPath.

    Source features are cached independently of settings; caller can cache JSON.
    OFF/LEGACY return disabled metadata. No source files or existing analysis data
    are written, and exact observed event times remain available off the grid.
    """
    from ..analysis_runtime import normalize_compute
    mode = canonical_mode(mode); settings = normalize_settings(settings)
    compute = normalize_compute(compute)
    regions = normalize_focus_regions(mapping_regions, focus_regions, analysis['duration'])
    if mode in ['off', 'legacy']:
        if regions:
            raise ValueError('Manual source attention needs ArcAudio 2.0, Hybrid or ArcAudio 3.0. Select an attention engine or use Auto regions.')
        return {'version': VERSION, 'enabled': False, 'mode': mode, 'settings': settings,
                'frames': [], 'events': [], 'selectedEvents': [], 'phrases': [], 'transitions': [],
                'summary': {'pipeline': 'Existing ArcForge analysis unchanged'}, 'performance': {'analysisSeconds': 0.}, 'limitations': []}
    started = time.perf_counter()
    # Reuse the current spectral roles and beat clock. Only opt-in calls upgrade
    # old in-memory analysis when it lacks measured role/phrase evidence.
    from ..music import ensure_music_analysis
    analysis = ensure_music_analysis(analysis)
    recognition=None
    if mode=='arcaudio3':
        from ..recognition import recognize
        analysis=copy.deepcopy(analysis)
        recognition=recognize(analysis,settings=compute)
        analysis['sourceEvidence']=recognition
    source_started = time.perf_counter(); evidence = extract_sources(analysis)
    hybrid_added=0
    if mode=='hybrid':
        from .hybrid import fuse_legacy
        evidence,hybrid_added=fuse_legacy(analysis,evidence)
    source_seconds = time.perf_counter() - source_started
    phrases = build_phrases(analysis, evidence['frames'], evidence['events'])
    decided = calculate_attention(evidence['frames'], evidence['events'], phrases, settings)
    decided = focused_frames(decided, evidence['events'], regions)
    events = select_events(decided, evidence['events'], phrases, settings, analysis.get('beatTimes', []), regions)
    transitions = detect_transitions(decided, phrases)
    result = _result(decided, events, phrases, transitions, settings, evidence['method'],
                     evidence['limitations'], time.perf_counter() - started)
    result['mode'] = mode; result['performance']['sourceAnalysisSeconds'] = round(source_seconds, 5)
    if regions:
        result.update(mappingRegions=regions, mappingFocus=focus_summary(regions, events))
    result['performance']['analysisCompute'] = compute
    result['performance']['computeAppliesToRecognition'] = mode == 'arcaudio3'
    if recognition:
        result['version']='3.0'
        result['recognition']={k:recognition.get(k) for k in ['model','method','version','performance','limitations']}
        result['limitations']=recognition.get('limitations',[])+result['limitations']
    result['beatGrid'] = {'bpm': analysis.get('bpm'), 'offset': analysis.get('offset', 0),
                          'confidence': analysis.get('confidence'), 'method': 'Existing ArcForge local transient-supported grid',
                          'timing': 'Selected event times are observed; this package does not snap them'}
    if mode == 'hybrid':
        result['summary']['hybridPolicy'] = 'Observed legacy attacks fused with spectral foreground and beat-phase support before the shared finite attention budget.'
        result['summary']['legacySupplementCount']=hybrid_added
    return result


def apply_focus_to_result(result, analysis, regions):
    """Reuse detected evidence when a caller supplied attention for other regions."""
    if result.get('mappingRegions', []) == regions:
        return result
    frames = []
    for frame in result.get('frames', []):
        original = frame.get('_automaticDecision')
        value = {**frame, **original} if original else dict(frame)
        value.pop('_automaticDecision', None)
        value.pop('mappingFocus', None)
        frames.append(value)
    frames = focused_frames(frames, result.get('events', []), regions)
    events = select_events(frames, result.get('events', []), result.get('phrases', []),
                           result.get('settings', {}), analysis.get('beatTimes', []), regions)
    changed = _result(frames, events, result.get('phrases', []), result.get('transitions', []),
                      result.get('settings', {}), result.get('method', 'cached_evidence'), [], 0.)
    value = {**result, **{key: changed[key] for key in ('frames', 'events', 'selectedEvents', 'summary')}}
    value['mappingRegions'] = copy.deepcopy(regions)
    value['mappingFocus'] = focus_summary(regions, events)
    return value


def _result(frames, events, phrases, transitions, settings, method, limitations, elapsed):
    chosen = [e for e in events if e['selected']]
    states = collections.Counter(f['state'] for f in frames)
    source_counts = collections.Counter(e['source'] for e in chosen)
    reasons = collections.Counter(e['reason'] for e in events if not e['selected'])
    switches = sum(a['primarySource'] != b['primarySource'] for a, b in zip(frames, frames[1:]))
    duration = frames[-1]['time'] - frames[0]['time'] if len(frames) > 1 else 0.
    return {'version': VERSION, 'enabled': True, 'mode': 'arcaudio', 'method': method,
            'settings': settings, 'frames': frames, 'events': events, 'selectedEvents': chosen,
            'phrases': phrases, 'transitions': transitions,
            'summary': {'candidateEvents': len(events), 'selectedEvents': len(chosen),
                        'eventReduction': round(1 - len(chosen) / max(len(events), 1), 4),
                        'rejectionReasons': dict(reasons), 'selectedSourceCounts': dict(source_counts),
                        'states': dict(states), 'sourceSwitches': switches,
                        'switchesPerMinute': round(switches * 60 / max(duration, 1.), 3),
                        'meanConfidence': round(sum(f['confidence'] for f in frames) / max(len(frames), 1), 4),
                        'meanDensityTarget': round(sum(f['densityTarget'] for f in frames) / max(len(frames), 1), 4),
                        'repeatedPhrases': sum(bool(p.get('repeated')) for p in phrases),
                        'energyIsImportance': False, 'attentionIsDetection': False},
            'performance': {'analysisSeconds': round(elapsed, 5), 'frameStepSeconds': round(median([b['time'] - a['time'] for a, b in zip(frames, frames[1:])]), 5) if len(frames) > 1 else None,
                            'frameCount': len(frames), 'candidateCount': len(events),
                            'sourceFeatureCache': 'In-process, up to three original-audio feature sets; settings independent'},
            'limitations': limitations + ['Context and attention confidence are evidence-strength estimates, not calibrated accuracy percentages.',
                                          'Verse/chorus/story semantics are not inferred. Phrase labels describe measured four-beat recurrence and activity transitions.']}


__all__ = ['analyze_attention', 'analyze_evidence', 'export_human_supervision', 'normalize_settings', 'DEFAULTS', 'VERSION']
