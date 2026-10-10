"""Measured rhythmic structure, not source-separated instrument guesses.

The channels below are spectral proxies (low/mid/high attacks and tonal change).
They describe audible events useful for rhythm mapping; they do not identify an
actual singer, guitar, drums, or stems in a mixed recording.
"""
from __future__ import annotations
import collections, copy, math, pathlib
import numpy as np
from scipy import signal, ndimage

VERSION = 4
ROLE_LABELS = {'lowPercussion': 'Low-frequency attacks', 'midTransient': 'Mid-frequency attacks',
               'highPercussion': 'High-frequency attacks', 'tonalChange': 'Tonal-change proxy'}


def _scale(values, percentile=97):
    return np.clip(values / max(float(np.percentile(values, percentile)), 1e-8), 0, 2)


def _peaks(envelope, times, hop_seconds, minimum_gap=.055):
    if not len(envelope) or float(envelope.max()) < 1e-7: return []
    value = _scale(envelope)
    # A local floor rejects vibrato/sustained-noise modulation while preserving
    # isolated quiet attacks. A global threshold avoids dense negligible wiggles.
    floor = ndimage.median_filter(value, size=max(3, int(.75 / hop_seconds) | 1), mode='nearest')
    peaks, properties = signal.find_peaks(value, distance=max(1, int(minimum_gap / hop_seconds)), prominence=.12)
    result = []
    for index, prominence in zip(peaks, properties['prominences']):
        if value[index] < max(.18, float(floor[index]) * 1.35 + .06): continue
        result.append({'time': round(max(0., float(times[index]) - hop_seconds / 2), 5),
                       'strength': round(min(1.5, float(prominence)), 4)})
    return result


def _merge_attacks(roles):
    points = sorted([(p['time'], p['strength'], role) for role, peaks in roles.items() for p in peaks])
    groups = []
    for second, strength, role in points:
        if groups and second - groups[-1]['time'] <= .032:
            group = groups[-1]
            if strength > group['strength']: group['time'], group['strength'] = second, strength
            if role not in group['roles']: group['roles'].append(role)
        else: groups.append({'time': second, 'strength': strength, 'roles': [role]})
    return groups


def _rates(points, centers, window=2.):
    if not points: return np.zeros(len(centers), np.float32)
    times = np.array([p['time'] for p in points])
    strengths = np.array([min(1., max(.3, p['strength'])) for p in points])
    total = np.r_[0., strengths.cumsum()]
    left = np.searchsorted(times, centers - window / 2)
    right = np.searchsorted(times, centers + window / 2, side='right')
    return (total[right] - total[left]) / window


def _time_at_beat(beat, bpm, offset, beat_times):
    if len(beat_times) > 1:
        if beat <= len(beat_times) - 1: return float(np.interp(beat, np.arange(len(beat_times)), beat_times))
        return float(beat_times[-1] + (beat - len(beat_times) + 1) * np.median(np.diff(beat_times[-9:])))
    return float(offset + beat * 60 / bpm)


def analyze_music(samples, sample_rate, bpm, offset=0., beat_times=None, spectrum=None, times=None, frequencies=None,stereo=None):
    """Return serializable sections, spectral roles, rhythms and repeat evidence."""
    hop = 256; hop_seconds = hop / sample_rate
    if spectrum is None:
        frequencies, times, spectrum = signal.stft(samples, fs=sample_rate, nperseg=1024, noverlap=1024-hop, boundary='zeros')
    magnitude = np.abs(spectrum).astype(np.float32)
    # Median-filter harmonic/percussive decomposition suppresses vibrato and
    # sustained formants in the rhythmic detector. It is a spectral estimate,
    # not instrument/stem separation. Vocals can still contain percussive sounds.
    harmonic = ndimage.median_filter(magnitude, size=(1, 31), mode='nearest')
    percussive = ndimage.median_filter(magnitude, size=(17, 1), mode='nearest')
    percussive_mask = percussive ** 2 / np.maximum(harmonic ** 2 + percussive ** 2, 1e-12)
    percussive_spectrum = magnitude * percussive_mask
    harmonic_spectrum = magnitude * (1 - percussive_mask)
    logged = np.log1p(percussive_spectrum * 100)
    positive_flux = np.maximum(np.diff(logged, axis=1, prepend=logged[:, :1]), 0)
    power = np.sqrt((magnitude ** 2).sum(0))
    energy = _scale(power, 98)
    band_ranges = {'lowPercussion': (35, 200), 'midTransient': (200, 2000), 'highPercussion': (2000, 10000)}
    roles = {}
    for role, (low, high) in band_ranges.items():
        band = positive_flux[(frequencies >= low) & (frequencies < high)]
        roles[role] = _peaks(band.mean(0), times, hop_seconds, .075 if role == 'lowPercussion' else .055)
    tonal_band = (frequencies >= 250) & (frequencies < 4000)
    flatness = np.exp(np.log(np.maximum(magnitude[tonal_band], 1e-8)).mean(0)) / np.maximum(magnitude[tonal_band].mean(0), 1e-8)
    harmonic_log = np.log1p(harmonic_spectrum * 100)
    harmonic_flux = np.maximum(np.diff(harmonic_log, axis=1, prepend=harmonic_log[:, :1]), 0)
    tonal_flux = harmonic_flux[tonal_band].mean(0) * np.clip(1 - flatness, .1, 1)
    roles['tonalChange'] = _peaks(tonal_flux, times, hop_seconds, .10)
    attacks = _merge_attacks(roles)
    duration = len(samples) / sample_rate
    centers = np.arange(0., duration + .001, .25)
    density_attacks = [p | {'strength': p['strength'] * (.4 if p['roles'] == ['tonalChange'] else 1.)} for p in attacks]
    rate = _rates(density_attacks, centers)
    rate_scale = max(float(np.percentile(rate, 90)), 4.)
    activity = np.clip(rate / rate_scale, 0, 1.5)
    role_rates = {role: _rates(peaks, centers) for role, peaks in roles.items()}
    rhythm_frames = []
    for index, second in enumerate(centers):
        dominant = max(role_rates, key=lambda key: role_rates[key][index])
        rhythm_frames.append({'time': round(float(second), 5), 'activity': round(float(activity[index]), 4),
                              'attacksPerSecond': round(float(rate[index]), 3),
                              'energy': round(float(np.interp(second, times, energy)), 4),
                              'primaryRole': dominant, 'roleRates': {role: round(float(v[index]), 3) for role, v in role_rates.items()}})
    # Four-beat measurements retain brief breakdowns and abrupt drop entries.
    # Sections are merged only while both measured activity and energy agree.
    beat_times = beat_times or []
    nbeats = len(beat_times) - 1 if len(beat_times) > 1 else int((duration - offset) * bpm / 60)
    bars = []
    for beat in range(0, max(1, nbeats), 4):
        start = max(0., _time_at_beat(beat, bpm, offset, beat_times)); end = min(duration, _time_at_beat(beat + 4, bpm, offset, beat_times))
        if end <= start: continue
        ids = (centers >= start) & (centers < end)
        temporal = (times >= start) & (times < end)
        a = float(activity[ids].mean()) if ids.any() else float(np.interp(start, centers, activity))
        e = float(energy[temporal].mean()) if temporal.any() else 0.
        rr = {role: float(v[ids].mean()) if ids.any() else 0. for role, v in role_rates.items()}
        primary = max(rr, key=rr.get)
        bars.append({'start': start, 'end': end, 'beat': beat, 'activity': a, 'energy': e, 'primaryRole': primary,
                     'attacksPerSecond': float(rate[ids].mean()) if ids.any() else 0.})
    sections, raw_labels = [], []
    for index, bar in enumerate(bars):
        a, e = bar['activity'], bar['energy']
        previous = np.mean([b['activity'] for b in bars[max(0, index - 4):index]]) if index else a
        following = np.mean([b['activity'] for b in bars[index + 1:index + 4]]) if index + 1 < len(bars) else a
        if a < .16: label = 'Intro' if index < 4 else 'Outro' if index > len(bars) - 5 else 'Break'
        elif a < .40: label = 'Build' if following > a * 1.35 else 'Break'
        elif a >= .68 and (e > .35 or bar['primaryRole'] == 'highPercussion'): label = 'Drop'
        elif following > a * 1.3 and a > previous * .95: label = 'Build'
        else: label = 'Drive'
        raw_labels.append(label)
        factor = {'Intro': .16, 'Outro': .18, 'Break': .22, 'Build': .40, 'Drive': .68, 'Drop': 1.}[label]
        # Rhythm, rather than amplitude, is the main term. A loud sustained voice
        # with few transients remains sparse at every requested difficulty.
        density = min(1.15, max(.08, .08 + .90 * a))
        if label in ['Intro', 'Outro', 'Break']: density = min(density, .36)
        bar.update(label=label, densityFactor=density)
        similar = sections and sections[-1]['label'] == label and abs(sections[-1]['activity'] - a) < .23
        if similar:
            section = sections[-1]; count = section.pop('_bars')
            for key in ['activity', 'energy', 'densityFactor', 'attacksPerSecond']:
                section[key] = (section[key] * count + bar[key]) / (count + 1)
            section['_bars'] = count + 1; section['end'] = bar['end']; section['endBeat'] = beat + 4
        else: sections.append(bar | {'startBeat': beat, 'endBeat': beat + 4, '_bars': 1})
    for index, section in enumerate(sections):
        section.pop('_bars', None); section['id'] = f'section-{index + 1}'
        section['mappingRole'] = ROLE_LABELS[section['primaryRole']]
        for key, value in list(section.items()):
            if isinstance(value, float): section[key] = round(value, 5)
    # Motif IDs come from measured sixteen-bin attack distributions, not a library
    # of prewritten note layouts. Different phrases are compared by cosine distance.
    motifs, signatures, motif_ids = [], [], []
    attack_times = np.array([a['time'] for a in attacks])
    for beat in range(0, max(1, nbeats - 3), 4):
        start = _time_at_beat(beat, bpm, offset, beat_times); end = _time_at_beat(beat + 4, bpm, offset, beat_times)
        indices = (attack_times >= start) & (attack_times < end)
        positions = (attack_times[indices] - start) / max(end - start, .001)
        signature = np.histogram(positions, bins=16, range=(0, 1))[0].astype(np.float32)
        signature /= max(float(np.linalg.norm(signature)), 1.)
        match = None
        for j, old in enumerate(signatures):
            if float(signature @ old) > .96 and signature.any(): match = j; break
        if match is None: match = len(signatures); signatures.append(signature)
        motif_ids.append(match)
        motifs.append({'start': round(start, 5), 'end': round(end, 5), 'beat': beat, 'motifId': match,
                       'attackBins': [int(v) for v in np.histogram(positions, bins=16, range=(0, 1))[0]]})
    tempo_evidence = tempo_segments(bpm, offset, beat_times)
    from .meter_stereo import estimate_downbeats,stereo_features
    downbeats=estimate_downbeats(beat_times,bpm,offset,roles,duration)
    stereo_analysis=stereo_features(stereo,sample_rate)
    return {'version': VERSION, 'method': 'Multiband spectral attacks, tonal-change proxy, phrase activity and onset motifs',
            'instrumentLabels': 'Spectral proxies with harmonic/percussive filtering; no stems or verified instrument separation',
            'roleLabels': ROLE_LABELS, 'roleOnsets': roles, 'accents': attacks, 'rhythmFrames': rhythm_frames,
            'sections': sections, 'motifs': motifs, 'distinctAudioMotifs': len(signatures),
            'tempoEvidence': tempo_evidence, 'tempoChangeWarning': any(s['relativeChange'] > .02 for s in tempo_evidence[1:]),
            'downbeats':downbeats,'stereo':stereo_analysis}


def tempo_segments(bpm, offset, beat_times, block=32):
    segments = []
    if len(beat_times) < 4: return [{'startBeat': 0, 'time': float(offset), 'bpm': float(bpm), 'relativeChange': 0., 'evidence': 'Global detector estimate'}]
    periods = np.diff(beat_times)
    for index in range(0, len(periods), block):
        part = periods[index:index + block]
        value = 60 / max(float(part.mean()), .01)
        previous = segments[-1]['bpm'] if segments else bpm
        segments.append({'startBeat': index, 'endBeat': index + len(part), 'time': float(beat_times[index]),
                         'bpm': round(value, 5), 'relativeChange': round(abs(value - previous) / max(previous, 1), 5),
                         'evidence': 'Local transient-supported beat-grid fit; review on the waveform'})
    return segments


def ensure_music_analysis(analysis):
    """Lazily upgrade previously uploaded recordings without changing the files."""
    if analysis.get('musicAnalysis', {}).get('version') == VERSION: return analysis
    result = copy.deepcopy(analysis)
    path = pathlib.Path(result.get('audioPath', ''))
    if path.is_file():
        from .features import decode_audio
        samples = decode_audio(path)
        stereo=decode_audio(path,channels=2)
        music = analyze_music(samples, 22050, result['bpm'], result.get('offset', 0), result.get('beatTimes', []),stereo=stereo)
    else:
        music = from_frames(result)
    result['musicAnalysis'] = music; result['sections'] = music['sections']
    return result


def from_frames(analysis):
    """Lightweight compatibility path for tests/imports without original audio."""
    frames = analysis.get('frames', [])
    if not frames:
        return {'version': VERSION, 'sections': analysis.get('sections', []), 'rhythmFrames': [], 'roleOnsets': {}, 'accents': [], 'motifs': [], 'tempoEvidence': []}
    times = np.array([f['time'] for f in frames]); flux = np.array([f.get('onset', 0) for f in frames])
    peaks, _ = signal.find_peaks(flux, prominence=max(.08, float(np.percentile(flux, 80)) * .15), distance=2)
    if not len(peaks) and analysis.get('onsets'): attack_times = list(analysis['onsets'])
    else: attack_times = times[peaks].tolist()
    attacks = [{'time': float(t), 'strength': 1., 'roles': ['midTransient']} for t in attack_times]
    centers = np.arange(0., analysis['duration'] + .01, .25); rates = _rates(attacks, centers)
    scale = max(float(np.percentile(rates, 90)), 4.)
    rhythm = [{'time': float(t), 'activity': float(r / scale), 'attacksPerSecond': float(r), 'primaryRole': 'midTransient',
               'roleRates': {'midTransient': float(r)}} for t, r in zip(centers, rates)]
    sections = []
    for start in np.arange(0, analysis['duration'], 4 * 60 / analysis['bpm']):
        end = min(analysis['duration'], start + 4 * 60 / analysis['bpm']); ids = (centers >= start) & (centers < end)
        activity = float(rates[ids].mean() / scale) if ids.any() else 0.
        label = 'Break' if activity < .3 else 'Drop' if activity > .65 else 'Drive'
        sections.append({'start': float(start), 'end': float(end), 'label': label, 'activity': activity, 'energy': activity,
                         'densityFactor': max(.08, min(1.15, .08 + .9 * activity)), 'primaryRole': 'midTransient'})
    return {'version': VERSION, 'sections': sections, 'rhythmFrames': rhythm, 'roleOnsets': {'midTransient': attacks},
            'accents': attacks, 'motifs': [], 'tempoEvidence': tempo_segments(analysis['bpm'], analysis.get('offset', 0), analysis.get('beatTimes', [])),
            'instrumentLabels': 'Spectral proxies inferred from existing onset frames'}
