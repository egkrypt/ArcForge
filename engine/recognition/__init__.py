"""ArcAudio 3.0: trained local source classification with observed-event timing.

No audio leaves the machine. Model installation fetches Google's official
YAMNet HDF5 weights only; no repository archive or source is downloaded.
"""
from __future__ import annotations
import base64, copy, hashlib, importlib.util, json, os, pathlib, threading, time, urllib.request, uuid
import numpy as np
from .yamnet import SAMPLE_RATE, WINDOW_SECONDS, HOP_SECONDS, log_mel_patches, build_model

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'models' / 'recognition'
WEIGHTS = MODEL_DIR / 'yamnet.h5'
MANIFEST = MODEL_DIR / 'yamnet-install.json'
VERSION = '3.0'
MODEL_URL = 'https://storage.googleapis.com/audioset/yamnet.h5'
MODEL_SOURCE = 'https://github.com/tensorflow/models/tree/master/research/audioset/yamnet'
APPROX_BYTES = 15296092
EXPECTED_SHA256 = '13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744'
_LOCK = threading.RLock()
_MODEL = None
_MODEL_KEY = None
_DEVICE = None

SOURCES = ('kick', 'snare', 'percussion', 'bass', 'drums', 'piano', 'chords', 'melody', 'lead', 'vocals', 'spoken', 'background', 'other')
GROUPS = {'kick': [163], 'snare': [160, 161], 'percussion': [156, 162, 165, 166, 167, 168, 169, 170, 171, 172],
          'drums': [156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172],
          'piano': [148, 149], 'bass': [137, 189],
          'melody': [134, 135, 136, 138, 139, 142, 143, 144, 146, 147, 148, 149, 150, 151, 152, 153, 155, 174, 175, 176, 177, 178, 180, 182, 183, 184, 186, 188, 190, 191, 192, 193, 194, 203, 204, 208],
          'lead': [136, 148, 149, 153, 182, 186, 191, 192, 193, 208],
          'vocals': [24, 25, 26, 27, 28, 29, 30, 31, 32, 249, 250],
          'spoken': [0, 1, 2, 3, 4, 5, 12], 'background': [262], 'other': [57, 58, 453, 454, 455, 457]}
# Timbre evidence is separate from mapping roles. A guitar can supply a lead,
# chords, or accompaniment; its class posterior must not assert which one.
INSTRUMENT_GROUPS = {'piano': [148, 149], 'keyboard': [147],
                     'guitar': [135, 136, 138, 139, 140, 141],
                     'synthesizer': [153, 154], 'plucked_strings': [134, 142, 143, 144, 145, 146, 187, 194],
                     'bowed_strings': [184, 185, 186, 188], 'brass': [180, 181, 182, 183],
                     'woodwind': [190, 191, 192, 193], 'bass': [137, 189]}
COMPONENT_FAMILIES = {'rhythm': ('drums', 'kick', 'snare', 'percussion'),
                      'pitched': ('piano', 'melody', 'lead'), 'bass': ('bass',),
                      'voice': ('vocals', 'spoken')}
SOURCE_FAMILY = {source: family for family, sources in COMPONENT_FAMILIES.items() for source in sources}
SOURCE_FAMILY['chords'] = 'pitched'
CLASS_NAMES = {0: 'Speech', 1: 'Child speech', 2: 'Conversation', 3: 'Narration', 12: 'Whispering',
               24: 'Singing', 25: 'Choir', 27: 'Chant', 30: 'Synthetic singing', 31: 'Rapping', 32: 'Humming',
               132: 'Music', 133: 'Musical instrument', 134: 'Plucked string instrument', 135: 'Guitar', 136: 'Electric guitar',
               137: 'Bass guitar', 138: 'Acoustic guitar', 147: 'Keyboard', 148: 'Piano', 149: 'Electric piano',
               150: 'Organ', 151: 'Electronic organ', 152: 'Hammond organ', 153: 'Synthesizer', 155: 'Harpsichord',
               156: 'Percussion', 157: 'Drum kit', 158: 'Drum machine', 159: 'Drum', 160: 'Snare drum', 161: 'Rimshot',
               162: 'Drum roll', 163: 'Bass drum', 164: 'Timpani', 165: 'Tabla', 166: 'Cymbal', 167: 'Hi-hat',
               168: 'Wood block', 169: 'Tambourine', 170: 'Rattle', 171: 'Maraca', 172: 'Gong', 174: 'Mallet percussion',
               175: 'Marimba / xylophone', 176: 'Glockenspiel', 177: 'Vibraphone', 178: 'Steelpan', 179: 'Orchestra',
               180: 'Brass instrument', 182: 'Trumpet', 183: 'Trombone', 184: 'Bowed string instrument', 185: 'String section',
               186: 'Violin', 187: 'Pizzicato', 188: 'Cello', 189: 'Double bass', 190: 'Wind instrument', 191: 'Flute', 192: 'Saxophone',
               193: 'Clarinet', 194: 'Harp', 203: 'Harmonica', 204: 'Accordion', 208: 'Theremin',
               214: 'Rock music', 215: 'Heavy metal', 232: 'Classical music', 234: 'Electronic music',
               235: 'House music', 236: 'Techno', 237: 'Dubstep', 238: 'Drum and bass', 240: 'Electronic dance music',
               249: 'Vocal music', 250: 'A capella', 262: 'Background music', 494: 'Silence'}
LIMITATIONS = ['YAMNet predicts overlapping AudioSet sound classes in 0.96-second windows; it does not separate stems or transcribe words.',
               'Class scores are model posteriors, not calibrated instrument-detection accuracy. Dense mixes and unfamiliar synthetic timbres can be ambiguous.',
               'Chords and musical foreground are contextual/DSP evidence, not dedicated YAMNet classes. Bass synths may be missed by the bass-guitar/double-bass vocabulary.',
               'Instrument evidence and source posteriors remain raw model scores. Pitched attacks do not prove piano, and guitar detection does not prove a lead role.',
               'Note candidates retain observed legacy onset times. Classifier window boundaries never become note timings.']


def _check(callback):
    from training_runtime import check_cancel
    check_cancel(callback)


def status():
    """Cheap readiness probe; do not import the GPU runtime on every UI refresh."""
    runtime = importlib.util.find_spec('torch') is not None or (ROOT / '.gpu-deps' / 'torch').is_dir()
    installed = WEIGHTS.is_file() and MANIFEST.is_file()
    record = {}
    if installed:
        try: record = json.loads(MANIFEST.read_text('utf-8'))
        except (ValueError, OSError): installed = False
    verified = bool(record.get('inferenceVerified')) and record.get('sha256') == EXPECTED_SHA256 and record.get('bytes') == (WEIGHTS.stat().st_size if WEIGHTS.exists() else 0)
    return {'ready': bool(installed and runtime and verified), 'installed': installed, 'runtimeAvailable': runtime,
            'model': 'YAMNet', 'version': VERSION, 'classes': 521, 'local': True, 'device': _DEVICE or record.get('device') or 'Auto · CPU/CUDA',
            'download': {'url': MODEL_URL, 'approxBytes': APPROX_BYTES, 'license': 'Apache-2.0', 'source': MODEL_SOURCE},
            'verification': record if installed else {}, 'limitations': LIMITATIONS}


def _torch():
    # Match the app's existing optional GPU runtime, without requiring TensorFlow.
    import sys
    gpu = ROOT / '.gpu-deps'
    if gpu.is_dir() and str(gpu) not in sys.path: sys.path.insert(0, str(gpu))
    import torch
    return torch


def _load_model(weight_path=WEIGHTS, device='auto'):
    global _MODEL, _MODEL_KEY, _DEVICE
    torch = _torch()
    if device not in ['auto', 'cpu', 'cuda']: raise ValueError('Recognition device must be auto, cpu or cuda')
    use_cuda = torch.cuda.is_available()
    if device == 'cuda' and not use_cuda: raise ValueError('CUDA recognition was requested but the runtime has no available CUDA device')
    chosen = 'cuda' if (device == 'cuda' or device == 'auto' and use_cuda) else 'cpu'
    weight_path = pathlib.Path(weight_path)
    fingerprint = (str(weight_path.resolve()), weight_path.stat().st_size, weight_path.stat().st_mtime_ns, chosen)
    with _LOCK:
        if _MODEL is None or _MODEL_KEY != fingerprint:
            _MODEL = build_model(weight_path, chosen); _MODEL_KEY = fingerprint; _DEVICE = chosen
        return _MODEL, torch, chosen


def install(progress=None, cancel_callback=None):
    """Install a bounded official weight file atomically and verify real inference."""
    _check(cancel_callback); MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if status()['ready']: return status()
    temporary = MODEL_DIR / (uuid.uuid4().hex + '.pending.h5')
    sha = hashlib.sha256(); md5 = hashlib.md5(); total = 0
    try:
        request = urllib.request.Request(MODEL_URL, headers={'User-Agent': 'ArcForge/1.3 local-model-installer'})
        with urllib.request.urlopen(request, timeout=45) as response, temporary.open('wb') as target:
            expected = int(response.headers.get('Content-Length', 0)); checksum = response.headers.get('x-goog-hash', '')
            if expected > 64 * 1024 * 1024: raise ValueError('Official model exceeds the 64 MB installation limit')
            for block in iter(lambda: response.read(256 * 1024), b''):
                _check(cancel_callback); total += len(block)
                if total > 64 * 1024 * 1024: raise ValueError('Model download exceeds the 64 MB limit')
                target.write(block); sha.update(block); md5.update(block)
                if progress: progress(min(90., 90. * total / max(expected, APPROX_BYTES)), f'Downloading official YAMNet weights · {total / 1048576:.1f} MB')
        if expected and total != expected: raise ValueError('Model download was incomplete')
        if sha.hexdigest() != EXPECTED_SHA256: raise ValueError('Official YAMNet weights do not match the pinned ArcAudio checksum')
        expected_md5 = next((item.split('=', 1)[1] for item in checksum.split(',') if item.strip().startswith('md5=')), None)
        if expected_md5 and base64.b64encode(md5.digest()).decode() != expected_md5: raise ValueError('Official storage checksum does not match')
        with temporary.open('rb') as source:
            if source.read(8) != b'\x89HDF\r\n\x1a\n': raise ValueError('Downloaded weights are not HDF5')
        if progress: progress(92., 'Verifying trained network with real local inference')
        _check(cancel_callback)
        model, torch, device = _load_model(temporary)
        with torch.inference_mode(): scores, _ = model(torch.from_numpy(log_mel_patches(np.zeros(16000, np.float32))).to(device))
        values = scores.detach().cpu().numpy()
        if values.shape != (2, 521) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError('Installed weights failed the 521-class inference check')
        _check(cancel_callback); temporary.replace(WEIGHTS)
        record = {'url': MODEL_URL, 'source': MODEL_SOURCE, 'license': 'Apache-2.0', 'bytes': total,
                  'sha256': sha.hexdigest(), 'storageMD5Verified': bool(expected_md5), 'installedAt': time.time(),
                  'inferenceVerified': True, 'device': device, 'silenceTopClass': int(values.mean(0).argmax()), 'pinnedSHA256Verified': True}
        MANIFEST.write_text(json.dumps(record, indent=2), 'utf-8')
        return status()
    finally:
        temporary.unlink(missing_ok=True)


def classify_samples(samples, device='auto', progress=None, cancel_callback=None, settings=None):
    """Run trained 521-class inference with Balanced or denser Deep windows.

    The original positional API and Balanced features are preserved. These are
    classifier context timestamps, never independent note timing candidates.
    """
    from ..analysis_runtime import normalize_compute, compute_plan, initial_batch_size, analysis_thread_scope
    if not status()['ready']: raise ValueError('ArcAudio 3.0 recognition weights are not installed. Install the optional local YAMNet model first.')
    config = normalize_compute(settings)
    if settings is None or ('device' not in settings and 'recognitionDevice' not in settings):
        config['device'] = device
        config = normalize_compute(config)
    started = time.perf_counter()
    scores = []
    # Serialize a shared inference model; changes to the selected device cannot
    # race another recognition job. Temporary CPU pool changes are restored.
    with _LOCK:
        _check(cancel_callback)
        if progress: progress(1., 'Loading the local source-recognition model')
        model_started = time.perf_counter()
        model, torch, chosen = _load_model(device=config['device'])
        model_seconds = time.perf_counter() - model_started
        plan = compute_plan(config, torch)
        with analysis_thread_scope(torch, plan), torch.inference_mode():
            _check(cancel_callback)
            if progress: progress(3., f'Computing {config["quality"]} audio features on {plan["threads"]} CPU threads')
            feature_started = time.perf_counter()
            patches = log_mel_patches(samples, hop_frames=plan['hopFrames'], workers=plan['fftWorkers'])
            feature_seconds = time.perf_counter() - feature_started
            batch, gpu_memory = initial_batch_size(torch, chosen, plan)
            initial_batch = batch
            if chosen == 'cuda':
                if gpu_memory is None:
                    free, total = torch.cuda.mem_get_info()
                    gpu_memory = {'freeBytesAtStart': int(free), 'totalBytes': int(total),
                                  'headroomPolicy': 'Balanced fixed 32-window batches.'}
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            network_started = time.perf_counter(); offset = 0; retries = 0
            while offset < len(patches):
                _check(cancel_callback)
                count = min(batch, len(patches) - offset)
                try:
                    prediction, _ = model(torch.from_numpy(patches[offset:offset + count]).to(chosen))
                    scores.append(prediction.detach().cpu().numpy())
                    del prediction
                except torch.cuda.OutOfMemoryError:
                    if chosen != 'cuda' or batch <= 1:
                        raise ValueError('The selected GPU has insufficient free memory for source recognition. Close other GPU tasks or choose CPU.') from None
                    # Retry the same unconsumed windows, never silently omit a
                    # region or fall back to a different quality/device.
                    batch = max(1, batch // 2); retries += 1
                    torch.cuda.empty_cache()
                    continue
                offset += count
                if progress:
                    progress(min(95., 10. + 85. * offset / max(len(patches), 1)),
                             f'Recognizing {config["quality"]} source context · {chosen.upper()} · {offset}/{len(patches)} windows')
            network_seconds = time.perf_counter() - network_started
            if chosen == 'cuda':
                gpu_memory.update(peakAllocatedBytes=int(torch.cuda.max_memory_allocated()),
                                  peakReservedBytes=int(torch.cuda.max_memory_reserved()))
    values = np.concatenate(scores)
    return {'scores': values, 'times': np.arange(len(values)) * plan['hopSeconds'] + .4875,
            'device': chosen, 'inferenceSeconds': time.perf_counter() - started, 'patchCount': len(values),
            'quality': config['quality'], 'requestedCompute': config,
            'hopSeconds': plan['hopSeconds'], 'windowSeconds': WINDOW_SECONDS,
            'cpuThreads': plan['threads'], 'fftWorkers': plan['fftWorkers'],
            'batchSize': batch, 'initialBatchSize': initial_batch, 'oomRetries': retries,
            'modelLoadSeconds': round(model_seconds, 5), 'featureSeconds': round(feature_seconds, 5),
            'networkSeconds': round(network_seconds, 5), 'gpuMemory': gpu_memory,
            'timingPolicy': 'Denser classifier contexts only; observed attack timestamps are unchanged.'}


def evidence_from_scores(analysis, scores, times, baseline=None):
    """Attach class evidence to measured attacks without inventing attack times."""
    from ..arcaudio.sources import extract_sources
    scores = np.asarray(scores, np.float32); times = np.asarray(times, float)
    if scores.ndim != 2 or scores.shape[1] != 521 or len(scores) != len(times) or not len(times):
        raise ValueError('Recognition requires aligned 521-class score frames')
    if baseline is None:
        baseline = extract_sources({k: v for k, v in analysis.items() if k != 'sourceEvidence'})
    grouped = {source: np.max(scores[:, indices], axis=1) for source, indices in GROUPS.items()}
    instruments = {name: np.max(scores[:, indices], axis=1) for name, indices in INSTRUMENT_GROUPS.items()}
    family_scores = {family: np.max(np.stack([grouped[source] for source in sources]), axis=0)
                     for family, sources in COMPONENT_FAMILIES.items()}
    # AudioSet often predicts broad Music confidently while leaving individual
    # instrument classes uncertain in commercial mixes. That is unidentified
    # musical information, not silence and not evidence of drums. Preserve
    # observed transient activity as an explicit unknown musical layer.
    observed = np.asarray(sorted(float(event['time']) for event in baseline['events'] if not event.get('sustained')), float)
    music_probability = scores[:, 132]
    instrument_probability = np.max(np.stack([grouped[source] for source in ['drums', 'piano', 'bass', 'melody', 'vocals', 'spoken']]), axis=0)
    frames = []
    for old in baseline['frames']:
        second = float(old['time'])
        sources = {source: float(np.interp(second, times, grouped[source])) if source in grouped else float(old['sources'].get(source, 0)) * .25 for source in SOURCES}
        event_rate = (np.searchsorted(observed, second + .5, side='right') - np.searchsorted(observed, second - .5, side='left'))
        music = float(np.interp(second, times, music_probability))
        identified = float(np.interp(second, times, instrument_probability))
        unknown_activity = min(1., event_rate / 8.) * music * (1. - min(1., identified / .35))
        # Late fusion: a weak AudioSet instrument posterior is not evidence that
        # measured pitched/percussive/bass activity disappeared. Retain those
        # functional components as DSP hypotheses, separately from class scores.
        posteriors=dict(sources)
        structural=old.get('evidence')=='mixed_audio_DSP_proxies'
        reliability=min(1.,identified/.35)
        family_reliability = {family: min(1., float(np.interp(second, times, values)) / .35)
                              for family, values in family_scores.items()}
        if structural:
            for source in ['bass','drums','kick','snare','percussion','melody','chords','lead','vocals','spoken']:
                proxy=float(old['sources'].get(source,0))
                # A confident voice does not disprove simultaneous drums or a
                # guitar phrase. AudioSet classes overlap; defer to measured
                # components independently for each family, instead of letting
                # one named class erase every other part of the arrangement.
                family = SOURCE_FAMILY[source]
                weight=(1-family_reliability[family])*(.75 if source in ['bass','drums','kick','snare','percussion','melody','chords'] else .50)
                sources[source]=max(sources[source],proxy*weight)
        resolved_activity=max(sources.get(s,0) for s in ['drums','bass','melody','vocals','spoken'])
        # "Music" is a parent class, not a separate audible source. Its large
        # posterior must not manufacture a dominant OTHER stream from event
        # counts. Unknown instrument identity remains in diagnostics; only
        # measured unresolved transient activity competes for mapping attention.
        unknown_strength=float(old['sources'].get('other',0))*(1-reliability) if structural else .80*unknown_activity
        sources['other'] = max(sources['other'], unknown_strength*(1-min(.90,resolved_activity)))
        # Generic Music / EDM classifications are not evidence of drums. Never
        # fall back from a strong bass or piano attack to an invented drum class.
        sources['drums'] = max(sources['drums'], sources['kick'], sources['snare'], sources['percussion'])
        confidences = {source: round(min(.85, .20 + .65 * posteriors[source]), 4) if source in grouped else .20 for source in SOURCES}
        if structural:
            for source in SOURCES:
                if sources[source]>posteriors[source]:confidences[source]=round(min(.62,float(old['sourceConfidence'].get(source,.30))),4)
        # We can be confident that attacks exist without claiming to know which
        # instrument generated them. This confidence describes the unknown layer.
        if unknown_activity > .03: confidences['other'] = .42
        if float(old.get('energy', 0)) < .003: sources = {source: 0. for source in SOURCES}
        index = int(np.argmin(np.abs(times - second)))
        top = np.argsort(scores[index])[-5:][::-1]
        frames.append({**old, 'sources': {source: round(value, 5) for source, value in sources.items()},
                       'sourceConfidence': confidences, 'evidence': 'YAMNet_pretrained_AudioSet_521',
                       'sourcePosteriors':{s:round(v,5) for s,v in posteriors.items()},
                       'instrumentEvidence':{name:round(float(np.interp(second,times,values)),5) for name,values in instruments.items()},
                       'componentEvidence':old.get('componentEvidence',{}),
                       'componentFamilySupport':{name:round(value,5) for name,value in family_reliability.items()},
                       'fusionPolicy':'Overlapping class evidence plus family-specific measured components; instrument identity is not a mapping role',
                       'unknownMusicalActivity': round(unknown_activity, 5), 'musicClassScore': round(music, 5),
                       'topClasses': [{'index': int(i), 'name': CLASS_NAMES.get(int(i), f'AudioSet class {int(i)}'), 'score': round(float(scores[index, i]), 5)} for i in top]})
    events = []
    compatibility = {'lowPercussion': {'kick': 1., 'bass': 1., 'piano': .4},
                     'midTransient': {'snare': 1., 'percussion': .7, 'piano': .9, 'vocals': .8, 'spoken': .8, 'melody': .65},
                     'highPercussion': {'percussion': 1., 'snare': .6, 'piano': .4, 'melody': .45},
                     'tonalChange': {'piano': 1., 'melody': .9, 'lead': .8, 'bass': .8, 'vocals': .9, 'spoken': .8}}
    candidates = {}
    # Use the legacy detector's observed attacks. The DSP voice/piano hypotheses
    # only provide supplemental observed envelopes, never unobserved grid points.
    for old in baseline['events']:
        second = float(old['time']); key = round(second, 5)
        if second < 0 or second > float(analysis.get('duration', 0)): continue
        strengths = {source: float(np.interp(second, times, grouped[source])) for source in grouped}
        weights = {}
        for role in old.get('roles', []):
            for source, multiplier in compatibility.get(role, {}).items(): weights[source] = max(weights.get(source, 0.), multiplier)
        if not weights: weights = {source: .8 for source in ['piano', 'bass', 'drums', 'vocals', 'spoken', 'melody']}
        frame=frames[min(len(frames)-1,max(0,int(round(second/.25))))]
        fusion=frame['sources']
        source = max(weights, key=lambda value: max(strengths.get(value,0.),fusion.get(value,0.)) * weights[value])
        posterior = strengths.get(source, 0.)
        unidentified = posterior < .06
        proxy=frame.get('componentEvidence') and fusion.get(source,0.)>.08
        if unidentified and not proxy: source = 'other'
        music = float(np.interp(second, times, music_probability))
        event = {**old, 'source': source, 'confidence': .42 if unidentified and music > .15 else round(min(.85, .20 + .65 * posterior), 4),
                 'evidence': 'spectral_component_at_observed_onset' if unidentified and proxy else 'observed_unidentified_musical_transient' if unidentified else 'trained_classification_at_observed_onset',
                 'componentEvidence':frame.get('componentEvidence',{}),'sourceEvidence':{s:round(fusion.get(s,0.)*w,5) for s,w in weights.items()},
                 'classificationScore': round(posterior, 5), 'musicClassScore': round(music, 5),
                 'classificationUncertain': unidentified,
                 'strength': round(float(old.get('strength', .5)), 5), 'timingEvidence': 'observed_onset'}
        if key not in candidates or event['classificationScore'] > candidates[key]['classificationScore']: candidates[key] = event
    events = sorted(candidates.values(), key=lambda event: event['time'])
    return {'method': 'YAMNet_pretrained_AudioSet_521', 'frames': frames, 'events': events,
            'model': 'YAMNet', 'version': VERSION, 'windowSeconds': WINDOW_SECONDS, 'hopSeconds': HOP_SECONDS,
            'classes': 521, 'local': True, 'limitations': LIMITATIONS}


def _source_baseline(analysis, cancel_callback=None):
    """One independent CPU feature job; never create or mutate classifier data."""
    from ..arcaudio.sources import extract_sources
    _check(cancel_callback); started = time.perf_counter()
    result = extract_sources({key: value for key, value in analysis.items() if key != 'sourceEvidence'})
    _check(cancel_callback)
    return result, time.perf_counter() - started


def _uncached_recognition(analysis, samples, settings, progress, cancel_callback):
    """Deep CUDA overlaps useful CPU HPSS with GPU inference, at most one worker.

    A context manager always joins the CPU worker on errors or cancellation.
    CPU-only inference already uses its CPU budget and avoids oversubscription.
    """
    parallel = (settings['quality'] == 'deep' and settings['device'] != 'cpu'
                and _torch().cuda.is_available())
    source_seconds = None
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='ArcAudio-source-features') as pool:
            future = pool.submit(_source_baseline, analysis, cancel_callback)
            try:
                inference = classify_samples(samples, progress=progress, cancel_callback=cancel_callback, settings=settings)
                baseline, source_seconds = future.result()
                _check(cancel_callback)
            finally:
                future.cancel()
        fusion_started = time.perf_counter()
        evidence = evidence_from_scores(analysis, inference['scores'], inference['times'], baseline)
    else:
        inference = classify_samples(samples, progress=progress, cancel_callback=cancel_callback, settings=settings)
        fusion_started = time.perf_counter()
        evidence = evidence_from_scores(analysis, inference['scores'], inference['times'])
    return evidence, inference, {'parallelSourceAnalysis': parallel,
        'sourceFeatureWorkers': 1 if parallel else 0,
        'sourceFeatureSeconds': round(source_seconds, 5) if source_seconds is not None else None,
        'fusionSeconds': round(time.perf_counter() - fusion_started, 5)}


def recognize(analysis, settings=None, progress=None, cancel_callback=None, force=False):
    """Reusable time-aligned source evidence; cache is independent of attention knobs."""
    started = time.perf_counter()
    from ..analysis_runtime import normalize_compute, COMPUTE_VERSION
    settings = normalize_compute(settings)
    path = pathlib.Path(analysis.get('audioPath', ''))
    if not path.is_file(): raise ValueError('ArcAudio 3.0 requires the original local audio file')
    if not status()['ready']: raise ValueError('Install the optional local YAMNet model to enable ArcAudio 3.0')
    if settings['device'] == 'cuda' and not _torch().cuda.is_available():
        raise ValueError('CUDA recognition was requested but the runtime has no available CUDA device')
    implementation_files = [pathlib.Path(__file__), pathlib.Path(__file__).with_name('yamnet.py'),
                            pathlib.Path(__file__).parents[1] / 'analysis_runtime.py']
    fingerprint = {'audio': str(path.resolve()), 'bytes': path.stat().st_size, 'modified': path.stat().st_mtime_ns,
                   'weights': json.loads(MANIFEST.read_text('utf-8'))['sha256'], 'version': VERSION,
                   'duration': analysis.get('duration'), 'roles': analysis.get('musicAnalysis', {}).get('version'),
                   'compute': settings, 'computeVersion': COMPUTE_VERSION,
                   'implementation': hashlib.sha256(b''.join(source.read_bytes() for source in implementation_files)).hexdigest()}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    cache = ROOT / 'data' / 'recognition-cache'; cache.mkdir(parents=True, exist_ok=True)
    target = cache / (key + '.npz')
    _check(cancel_callback)
    if target.is_file() and not force:
        try:
            with np.load(target, allow_pickle=False) as saved:
                scores, times = saved['scores'], saved['times']; performance = json.loads(str(saved['metadata']))
            evidence = evidence_from_scores(analysis, scores, times)
            evidence['hopSeconds'] = performance.get('hopSeconds', HOP_SECONDS)
            evidence['performance'] = performance | {'cacheHit': True, 'requestedCompute': settings,
                'cacheOriginDevice': performance.get('device'), 'inferenceExecuted': False,
                'processingSeconds': round(time.perf_counter() - started, 5)}; return evidence
        except (ValueError, OSError, KeyError): pass
    from ..features import decode_audio
    decode_started = time.perf_counter(); samples = decode_audio(path, sample_rate=SAMPLE_RATE)
    decode_seconds = time.perf_counter() - decode_started
    evidence, inference, stages = _uncached_recognition(analysis, samples, settings, progress, cancel_callback)
    evidence['hopSeconds'] = inference['hopSeconds']
    performance = {key: value for key, value in inference.items() if key not in {'scores', 'times'}}
    performance.update(stages)
    performance['decodeSeconds'] = round(decode_seconds, 5)
    temporary = cache / (uuid.uuid4().hex + '.pending.npz')
    try:
        _check(cancel_callback)
        np.savez_compressed(temporary, scores=inference['scores'], times=inference['times'], metadata=json.dumps(performance))
        temporary.replace(target)
    finally: temporary.unlink(missing_ok=True)
    evidence['performance'] = performance | {'cacheHit': False, 'inferenceExecuted': True,
        'processingSeconds': round(time.perf_counter() - started, 5)}; return evidence


__all__ = ['status', 'install', 'recognize', 'classify_samples', 'evidence_from_scores', 'VERSION', 'LIMITATIONS']
