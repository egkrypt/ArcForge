"""Time-aligned source evidence from existing roles and lightweight mixed-audio DSP.

The mixed-audio path is deliberately not advertised as instrument separation.
Speech, a synth, and piano can share a spectrum; their evidence is uncertain.
Externally separated or annotated sourceEvidence has an explicit provenance path.
"""
from __future__ import annotations
import pathlib, functools
import numpy as np
from scipy import signal, ndimage
from .settings import SOURCES

STEP = .25
PROXY_CONFIDENCE = {'kick': .56, 'snare': .36, 'percussion': .62, 'bass': .56, 'drums': .62,
                    'piano': .27, 'chords': .48, 'melody': .48, 'lead': .40, 'vocals': .30,
                    'spoken': .23, 'background': .52, 'other': .54}
ROLE_SOURCES = {'lowPercussion': 'kick', 'midTransient': 'snare', 'highPercussion': 'percussion', 'tonalChange': 'melody'}


def _norm(values, percentile=98):
    values = np.maximum(np.asarray(values, dtype=float), 0)
    return np.clip(values / max(float(np.percentile(values, percentile)) if len(values) else 0., 1e-8), 0, 1)


def _interp(centers, times, values):
    return np.interp(centers, times, values, left=0., right=0.) if len(times) else np.zeros(len(centers))


def _rate(event_times, centers, window=1.):
    values = np.asarray(sorted(event_times), dtype=float)
    return (np.searchsorted(values, centers + window / 2, side='right') - np.searchsorted(values, centers - window / 2)) / window


def _role_events(analysis):
    music = analysis.get('musicAnalysis', {})
    events = []
    for role, peaks in music.get('roleOnsets', {}).items():
        source = ROLE_SOURCES.get(role, 'other')
        for peak in peaks:
            events.append({'time': float(peak['time']), 'source': source, 'roles': [role],
                           'strength': min(1., max(0., float(peak.get('strength', 1.)))),
                           'confidence': PROXY_CONFIDENCE[source], 'evidence': 'spectral_proxy',
                           'timingEvidence': 'observed_onset'})
    if not events:
        for second in analysis.get('onsets', []):
            events.append({'time': float(second), 'source': 'other', 'roles': ['midTransient'],
                           'strength': .65, 'confidence': .35, 'evidence': 'legacy_mixed_onset',
                           'timingEvidence': 'observed_onset'})
    return events


def _provided_evidence(analysis):
    evidence = analysis.get('sourceEvidence')
    if not isinstance(evidence, dict) or not isinstance(evidence.get('frames'), list): return None
    frames = evidence['frames']
    if not frames: return None
    provenance = str(evidence.get('method', 'supplied_source_evidence'))
    # Supplied evidence is not silently called ground truth. Default uncertain;
    # an annotated/separation adapter must provide its calibrated confidence.
    output = []
    for frame in sorted(frames, key=lambda f: float(f.get('time', 0))):
        sources = {s: min(1., max(0., float(frame.get('sources', {}).get(s, 0)))) for s in SOURCES}
        sources['drums'] = max(sources['drums'], sources['kick'], sources['snare'], sources['percussion'])
        output.append({**frame, 'time': float(frame['time']), 'sources': sources,
                       'sourceConfidence': {s: min(1., max(0., float(frame.get('sourceConfidence', {}).get(s, .45)))) for s in SOURCES},
                       'evidence': provenance})
    events = []
    for event in evidence.get('events', []):
        if event.get('source') not in SOURCES: continue
        events.append({'time': float(event['time']), 'source': event['source'], 'roles': event.get('roles', []),
                       'strength': min(1., max(0., float(event.get('strength', .7)))),
                       'confidence': min(1., max(0., float(event.get('confidence', .45)))),
                       'evidence': event.get('evidence', provenance), 'timingEvidence': 'observed_onset',
                       **{k: event[k] for k in ['sustained', 'phraseId', 'componentEvidence', 'sourceEvidence',
                            'classificationUncertain', 'originalRoles', 'gridSupport'] if k in event}})
    return output, sorted(events, key=lambda e: e['time']), provenance


def extract_sources(analysis):
    supplied = _provided_evidence(analysis)
    if supplied:
        frames, events, provenance = supplied
        return {'frames': frames, 'events': events, 'method': provenance,
                'limitations': ['Supplied source evidence confidence is supplied by its adapter; it is not independently verified here.']}
    duration = max(0., float(analysis.get('duration', 0)))
    centers = np.arange(0., duration + .0001, STEP)
    legacy = analysis.get('frames', [])
    old_times = np.asarray([float(f['time']) for f in legacy])
    old = {key: _interp(centers, old_times, [float(f.get(key, 0)) for f in legacy]) for key in ['energy', 'bass', 'onset', 'high']}
    events = _role_events(analysis)
    measured = {}
    path = pathlib.Path(analysis.get('audioPath', ''))
    if path.is_file():
        measured, extra = _cached_spectral_file(str(path.resolve()), path.stat().st_mtime_ns, round(duration, 5))
        # A frequency-band attack is not an instrument label. A piano's low
        # harmonics and upper partials used to become three independent drum
        # streams, whose event counts then overrode the harmonic separation.
        # Keep overlapping attack evidence but route pitched attacks by their
        # measured harmonic/percussive balance, with explicit proxy provenance.
        attack_components = measured.get('_attackComponents', {})
        attack_times = attack_components.get('times', centers)
        supported_events=[]
        for event in events:
            role=next((r for r in event.get('roles',[]) if r in attack_components.get('roleHarmonicFraction',{})),None)
            harmonic_curve=attack_components.get('roleHarmonicFraction',{}).get(role,attack_components.get('harmonicFraction',measured['harmonicFraction']))
            harmonic=float(np.interp(event['time'],attack_times,harmonic_curve))
            bass=float(np.interp(event['time'],attack_times,attack_components.get('bassFraction',measured['bassFraction'])))
            band_support=float(np.interp(event['time'],attack_times,attack_components.get('roleEnergyFraction',{}).get(role,np.ones(len(attack_times)))))
            # Per-band normalization can turn inaudible spectral leakage into a
            # large onset score. A band with less than 3% of the attack's actual
            # amplitude is not evidence of an independent audible component.
            if role and band_support<.03:continue
            event['componentEvidence']={'harmonic':round(harmonic,4),'percussive':round(1-harmonic,4),
                                       'lowFrequency':round(bass,4),'bandSupport':round(band_support,4),
                                       'resolutionSeconds':round(256/22050,6)}
            if event['source'] in ['kick','snare','percussion']:
                percussive=1-harmonic
                if harmonic>.56:
                    event['source']='bass' if event['source']=='kick' and bass>.20 else 'melody'
                    event['originalRoles']=event['roles'];event['roles']=['tonalChange']
                    event['confidence']=PROXY_CONFIDENCE[event['source']]
                else:
                    event['strength']*=max(.15,min(1.,percussive*1.8))
                event['evidence']='harmonic_percussive_attack_proxy'
            supported_events.append(event)
        events=supported_events
        events.extend(extra)
    rates = {source: _rate([e['time'] for e in events if e['source'] == source], centers) for source in SOURCES}
    energy = measured.get('energy', np.clip(old['energy'], 0, 1))
    # Coincident band detections are one rhythmic instant. Counting them as
    # independent hits made quiet pitched passages look maximally dense.
    attacks=[]
    for event in sorted(events,key=lambda e:e['time']):
        if event.get('strength',0)<.12:continue
        if not attacks or event['time']-attacks[-1]>.032:attacks.append(event['time'])
    activity = np.clip(_rate(attacks, centers, 2.) / 10., 0, 1)
    if measured:
        activity*=np.clip(.20+measured['spectralActivity']*1.5,0,1)
    kick = np.maximum(np.clip(rates['kick'] / 4, 0, 1), old['bass'] * .25)
    snare = np.clip(rates['snare'] / 4, 0, 1)
    percussion = np.clip(rates['percussion'] / 8, 0, 1)
    frames = []
    for i, second in enumerate(centers):
        sources = {s: float(measured.get(s, np.zeros(len(centers)))[i]) for s in SOURCES}
        if not measured:
            sources['kick'] = float(kick[i])
            sources['snare'] = float(snare[i])
            sources['percussion'] = float(percussion[i])
            sources['drums'] = min(1., .60 * sources['kick'] + .55 * sources['snare'] + .42 * sources['percussion'])
            sources['bass'] = float(np.clip(old['bass'][i] * .35, 0, 1))
            sources['other'] = float(np.clip(old['onset'][i], 0, 1))
            sources['melody'] = float(np.clip(rates['melody'][i] / 4, 0, 1))
        # Actual near-silence wins over track-relative band normalization.
        if energy[i] < .003: sources = {s: 0. for s in SOURCES}
        frames.append({'time': round(float(second), 5), 'sources': {s: round(min(1., max(0., v)), 4) for s, v in sources.items()},
                       'sourceConfidence': {s: round(float(measured.get(s + 'Confidence', np.full(len(centers), PROXY_CONFIDENCE[s]))[i]), 4) for s in SOURCES},
                       'energy': round(float(energy[i]), 4), 'spectralActivity': round(float(measured.get('spectralActivity', activity)[i]), 4),
                       'rhythmDensity': round(float(activity[i]), 4),
                       'intensity': round(float(np.clip(.35 * energy[i] + .65 * activity[i], 0, 1)), 4),
                       'transientPresence':{'drums':round(float(measured['drumPresence'][i]),4)} if measured else {},
                       'componentEvidence':{k:round(float(measured[k][i]),4) for k in ['harmonicFraction','percussiveFraction','bassFraction','tonalConcentration'] if k in measured},
                       'evidence': 'mixed_audio_DSP_proxies' if measured else 'legacy_spectral_role_proxies'})
    return {'frames': frames, 'events': sorted(events, key=lambda e: e['time']),
            'method': 'Harmonic/percussive spectral masks, multiband attacks, harmonic concentration and modulation proxies',
            'limitations': ['This DSP path does not use a trained source separator or instrument classifier. Kick/snare/bass/voice/piano labels are uncertain spectral hypotheses, not verified stems.',
                           'Mixed speech, piano, guitar and synth timbres can be ambiguous. Source confidence is capped, and uncertain selection becomes conservative.',
                           'No word transcription or syllable labeling is performed. Voice-compatible envelope attacks can provide timing candidates.']}


@functools.lru_cache(maxsize=3)
def _cached_spectral_file(path, modified, duration):
    """Feature cache is independent of attention settings; tuning avoids decoding."""
    from ..features import decode_audio
    samples = decode_audio(path, sample_rate=22050)
    centers = np.arange(0., duration + .0001, STEP)
    return _spectral_features(samples, 22050, centers) if len(samples) else ({}, [])


def _spectral_features(samples, sample_rate, centers):
    """Extract reusable features once; voice-compatible features do not prove a voice."""
    frequencies, times, transform = signal.stft(samples, fs=sample_rate, nperseg=1024, noverlap=768, boundary='zeros')
    magnitude = np.abs(transform).astype(np.float32)
    harmonic_med = ndimage.median_filter(magnitude, size=(1, 17), mode='nearest')
    percussive_med = ndimage.median_filter(magnitude, size=(13, 1), mode='nearest')
    harmonic_mask = harmonic_med ** 2 / np.maximum(harmonic_med ** 2 + percussive_med ** 2, 1e-12)
    harmonic = magnitude * harmonic_mask; percussive = magnitude * (1 - harmonic_mask)
    harmonic_energy = np.sqrt((harmonic ** 2).sum(0)); percussive_energy = np.sqrt((percussive ** 2).sum(0))
    power = np.sqrt((magnitude ** 2).sum(0)); energy = _norm(power)
    ratio_h = harmonic_energy / np.maximum(harmonic_energy + percussive_energy, 1e-8)
    bands = {}
    for name, low, high in [('bass', 35, 220), ('low', 35, 180), ('mid', 180, 2400), ('voice', 300, 3500), ('high', 2400, 10000)]:
        ids = (frequencies >= low) & (frequencies < high)
        bands[name] = np.sqrt((magnitude[ids] ** 2).sum(0))
    bass_fraction = bands['bass'] / np.maximum(power, 1e-8)
    voice_fraction = bands['voice'] / np.maximum(power, 1e-8)
    log_harmonic = np.log1p(harmonic * 100)
    harmonic_flux = np.maximum(np.diff(log_harmonic, axis=1, prepend=log_harmonic[:, :1]), 0).mean(0)
    log_percussion = np.log1p(percussive * 100)
    flux = np.maximum(np.diff(log_percussion, axis=1, prepend=log_percussion[:, :1]), 0).mean(0)
    tonal_ids = (frequencies >= 200) & (frequencies < 4000)
    tonal = harmonic[tonal_ids]
    tonal_concentration = np.clip(tonal.max(0) / np.maximum(tonal.sum(0), 1e-8) * 6, 0, 1)
    # A track-relative normalization of an almost absent harmonic residual can
    # otherwise turn broadband noise tails into a strong melodic component.
    harmonic_floor=max(float(np.percentile(power,98))*.10,1e-8)
    harmonic_presence = _norm(harmonic_energy) * ratio_h * np.clip(harmonic_energy/harmonic_floor,0,1)
    melody = harmonic_presence * (.35 + .65 * tonal_concentration)
    # Modulation in the speech-rate range is evidence only. AM instruments and
    # vibrato also meet it, so voice confidence stays <= .30 in this DSP path.
    voice_envelope = _norm(bands['voice'])
    rate = sample_rate / 256
    smooth = ndimage.uniform_filter1d(voice_envelope, max(3, int(rate * .15)))
    slow = ndimage.uniform_filter1d(voice_envelope, max(3, int(rate * .7)))
    modulation = _norm(np.abs(smooth - slow))
    voice_candidate = voice_envelope * ratio_h * voice_fraction * modulation
    flatness = np.exp(np.log(np.maximum(magnitude[tonal_ids], 1e-8)).mean(0)) / np.maximum(magnitude[tonal_ids].mean(0), 1e-8)
    speech_candidate = voice_envelope * voice_fraction * modulation * np.clip(flatness * 3, 0, 1)
    attack = _norm(harmonic_flux)
    piano = harmonic_presence * attack * (1 - tonal_concentration * .40)
    voice_gate = voice_envelope * np.clip(voice_fraction * 1.8, 0, 1)
    # Relative foreground evidence is distinct from absolute RMS. Normalize the
    # modulation cue, then gate it with actual voice-band presence. A quiet
    # foreground phrase can therefore compete with a louder accompaniment.
    # Confidence remains capped: this still cannot tell a singer from an AM synth.
    vocal_strength = _norm(voice_candidate) * voice_gate * ratio_h
    spoken_strength = _norm(speech_candidate) * voice_gate * np.clip(flatness * 3, 0, 1)
    values = {'energy': energy, 'spectralActivity': _norm(flux + harmonic_flux),
              'harmonicFraction':ratio_h,'percussiveFraction':1-ratio_h,
              'bassFraction':bass_fraction,'tonalConcentration':tonal_concentration,
              'bass': _norm(bands['bass']) * bass_fraction * ratio_h,
              'drums': _norm(percussive_energy) * (1 - ratio_h),
              'piano': _norm(piano) * harmonic_presence, 'chords': harmonic_presence * (1 - tonal_concentration),
              'melody': melody, 'lead': melody * tonal_concentration,
              'vocals': vocal_strength, 'spoken': spoken_strength,
              'background': harmonic_presence * (1 - attack * .65), 'other': _norm(flux) * .50}
    # Counts describe rhythm density, not source strength. Continuous drum
    # evidence comes from measured separated energy, so a busy supporting beat
    # cannot win attention merely by appearing in several frequency bands.
    for source,low,high in [('kick',35,180),('snare',180,2400),('percussion',2400,10000)]:
        ids=(frequencies>=low)&(frequencies<high)
        band_p=np.sqrt((percussive[ids]**2).sum(0))
        values[source]=_norm(band_p)*(1-ratio_h)
    # Context uses the information in the whole frame, not one arbitrary STFT
    # sample every quarter second. Point sampling aliases fast attacks against
    # the 4 Hz context clock and can make an active phrase look silent. Event
    # timing and event-local components retain the original STFT resolution.
    pool_width = max(1, int(round(STEP * sample_rate / 256)))
    pooled = {name: np.sqrt(np.maximum(0.,ndimage.uniform_filter1d(value ** 2, pool_width, mode='constant'))) for name, value in values.items()}
    # Drums are transient: their presence within the context window matters,
    # even if their short attack occupies little of its average energy. This
    # does not add timing candidates or equate that peak to musical importance.
    pooled['drumPresence']=ndimage.maximum_filter1d(values['drums'],pool_width,mode='constant')
    pooled_h = ndimage.uniform_filter1d(harmonic_energy, pool_width, mode='constant')
    pooled_p = ndimage.uniform_filter1d(percussive_energy, pool_width, mode='constant')
    pooled['harmonicFraction'] = pooled_h / np.maximum(pooled_h + pooled_p, 1e-8)
    pooled['percussiveFraction'] = 1 - pooled['harmonicFraction']
    result = {name: np.clip(_interp(centers, times, value), 0, 1) for name, value in pooled.items()}
    # HPSS needs a few frames to distinguish a pitched attack from a noise
    # attack. Use a short, forward-looking window at the observed onset, within
    # each onset's frequency band. A simultaneous high drum and low melody can
    # therefore remain different components instead of one global label. The
    # window describes timbre; it never shifts the onset sent to the mapper.
    attack_width=5
    role_fractions={};role_energy={}
    attack_power=ndimage.uniform_filter1d((magnitude**2).sum(0),attack_width,origin=-1,mode='constant')
    attack_h=np.sqrt(np.maximum(0.,ndimage.uniform_filter1d(harmonic_energy**2,attack_width,origin=-1,mode='constant')))
    attack_p=np.sqrt(np.maximum(0.,ndimage.uniform_filter1d(percussive_energy**2,attack_width,origin=-1,mode='constant')))
    attack_harmonic=attack_h/np.maximum(attack_h+attack_p,1e-8)
    for role,low,high in [('lowPercussion',35,180),('midTransient',180,2400),('highPercussion',2400,10000),('tonalChange',200,4000)]:
        selected=(frequencies>=low)&(frequencies<high)
        h=ndimage.uniform_filter1d((harmonic[selected]**2).sum(0),attack_width,origin=-1,mode='constant')
        p=ndimage.uniform_filter1d((percussive[selected]**2).sum(0),attack_width,origin=-1,mode='constant')
        h=np.sqrt(np.maximum(0.,h));p=np.sqrt(np.maximum(0.,p))
        role_fractions[role]=h/np.maximum(h+p,1e-8)
        # The low band contains very few FFT bins. Window leakage from a
        # nearby pitched fundamental looks transient there, so use the short
        # whole-spectrum harmonic context for this ambiguous low-band attack.
        if role=='lowPercussion':role_fractions[role]=attack_harmonic
        band_power=ndimage.uniform_filter1d((magnitude[selected]**2).sum(0),attack_width,origin=-1,mode='constant')
        role_energy[role]=np.sqrt(np.maximum(0.,band_power)/np.maximum(attack_power,1e-12))
    result['_attackComponents'] = {'times': times, 'harmonicFraction': ratio_h, 'bassFraction': bass_fraction,
                                  'roleHarmonicFraction':role_fractions,'roleEnergyFraction':role_energy}
    for name in SOURCES:
        # A category's evidence changes over the song. Percussive spectral
        # regions strengthen rhythmic hypotheses; harmonic regions weaken them.
        support = 1 - ratio_h if name in ['kick', 'snare', 'percussion', 'drums'] else ratio_h if name in ['piano', 'chords', 'melody', 'lead', 'vocals'] else np.ones(len(times))
        supported = ndimage.uniform_filter1d(PROXY_CONFIDENCE[name] * (.40 + .60 * support), pool_width, mode='nearest')
        result[name + 'Confidence'] = _interp(centers, times, supported)
    extra = []
    hop_seconds = 256 / sample_rate
    # Voice-compatible rhythmic envelope events survive off-grid, but uncertain
    # voice candidates never get elevated to known syllables.
    for source, envelope in [('vocals', vocal_strength), ('spoken', spoken_strength), ('piano', values['piano']), ('bass', np.maximum(np.diff(values['bass'], prepend=0), 0))]:
        scaled = _norm(envelope)
        peak_ids, properties = signal.find_peaks(scaled, distance=max(1, int(.075 / hop_seconds)), prominence=.22)
        floor = ndimage.median_filter(scaled, size=max(3, int(.65 / hop_seconds) | 1))
        for index, prominence in zip(peak_ids, properties['prominences']):
            if envelope[index] < .035 or scaled[index] < floor[index] * 1.4 + .08: continue
            extra.append({'time': round(max(0., float(times[index]) - hop_seconds / 2), 5), 'source': source,
                          'roles': ['tonalChange' if source in ['piano', 'vocals'] else 'midTransient'],
                          'strength': round(min(1., float(prominence)), 4), 'confidence': PROXY_CONFIDENCE[source],
                          'evidence': 'spectral_proxy', 'timingEvidence': 'observed_onset'})
    return result, extra
