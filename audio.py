"""Local audio conversion, transient analysis and waveform pyramid creation."""
import bootstrap
import hashlib, json, math, pathlib, re, subprocess, functools
import numpy as np
from scipy import signal
import imageio_ffmpeg
from engine.music import ensure_music_analysis

SR = 22050
HOP = 256
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

def decode(path, max_seconds=1800):
    result = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-vn', '-t', str(max_seconds + 1),
        '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, timeout=240)
    if result.returncode:
        raise ValueError('This file has no decodable audio. ' + result.stderr.decode(errors='replace')[-350:])
    samples = np.frombuffer(result.stdout, dtype='<f4').copy()
    if len(samples) < SR * 2:
        raise ValueError('Choose a song at least two seconds long.')
    if len(samples) > SR * max_seconds:
        raise ValueError('Songs longer than 30 minutes are not supported in this local session.')
    if not np.isfinite(samples).all():
        raise ValueError('The decoded audio contains invalid samples.')
    return samples

def convert(source, target):
    """Keep compatible original Vorbis OGG; convert all other FFmpeg audio inputs."""
    probe = subprocess.run([FFMPEG, '-hide_banner', '-i', str(source)], capture_output=True, timeout=30)
    description = probe.stderr.decode(errors='replace')
    compatible = source.suffix.lower() in ['.ogg', '.egg'] and 'Audio: vorbis' in description
    if compatible:
        target.write_bytes(source.read_bytes())
    else:
        result = subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', str(source), '-vn',
            '-c:a', 'libvorbis', '-q:a', '7', str(target)], capture_output=True, timeout=240)
        if result.returncode:
            raise ValueError('Audio conversion failed. ' + result.stderr.decode(errors='replace')[-350:])
    return 'original Vorbis OGG' if compatible else 'converted to Vorbis OGG at original speed'

def _normalized(values):
    scale = max(float(np.percentile(values, 98)), 1e-8)
    return np.clip(values / scale, 0, 1.5)

def waveform(samples, points=2400):
    boundaries = np.linspace(0, len(samples), min(points, len(samples)) + 1, dtype=int)
    return [{'min': round(float(samples[a:b].min()), 5), 'max': round(float(samples[a:b].max()), 5)}
        for a, b in zip(boundaries, boundaries[1:]) if b > a]

def track_grid(times,envelope,bpm,offset,duration):
    """Fit continuous local beat periods; audio is never stretched or resampled."""
    spb=60/bpm;correction=HOP/SR/2
    strong=float(np.percentile(envelope,85))
    phases=np.arange(0,spb,.0015)
    ids=np.arange(1,min(48,int(duration/spb)))
    early=np.interp(ids[:,None]*spb+phases+correction,times,envelope).mean(0)
    if early.max()>strong*.5:offset=float(phases[int(early.argmax())])
    anchors=[(0,offset)];block=32;beats=int(math.ceil((duration-offset)/spb))
    for index in range(0,beats,block):
        size=min(block,beats-index)
        start=anchors[-1][1];local_ids=np.arange(1,size+1)
        shifts=np.arange(-.055,.0551,.002)
        periods=spb*np.linspace(.985,1.015,41)
        grid=start+shifts[:,None,None]+periods[None,:,None]*local_ids[None,None,:]+correction
        scores=np.interp(grid,times,envelope).mean(2)
        penalty=scores.max()*(.14*(shifts[:,None]/.055)**2+.05*((periods[None,:]-spb)/(.015*spb))**2)
        if scores.max()>strong*.5:
            a,b=np.unravel_index((scores-penalty).argmax(),scores.shape)
            end=start+float(shifts[a])+float(periods[b])*size
        else:end=start+spb*size
        anchors.append((index+size,end))
    ab,at=np.array(anchors).T
    fitted=np.interp(np.arange(beats+1),ab,at)
    return round(offset,5),[round(float(t),5) for t in fitted if t<=duration]

@functools.lru_cache(maxsize=3)
def waveform_samples(path,modified):return decode(path)

def waveform_window(path,start,end,points=2048):
    source=pathlib.Path(path);samples=waveform_samples(str(source),source.stat().st_mtime_ns)
    duration=len(samples)/SR;start=max(0,min(float(start),duration));end=max(start,min(float(end),duration))
    if end<=start:raise ValueError('Select a waveform interval inside the song.')
    if not 128<=points<=12000:raise ValueError('Waveform resolution must be between 128 and 12000 points.')
    return {'start':start,'end':end,'sampleRate':SR,'waveform':waveform(samples[int(start*SR):int(end*SR)],points)}

def detect_grid(times, bass_flux, full_flux, duration):
    # Whole-song grid coherence, followed by sub-BPM and sub-frame refinement.
    envelope = np.minimum(bass_flux, np.percentile(bass_flux, 97))
    if envelope.max() < 1e-7:
        return 120., 0., 0., [], 'No reliable percussion detected; set the grid manually.'
    weighted = envelope + .10 * np.minimum(full_flux, np.percentile(full_flux, 97))
    candidates = []
    start = min(8., duration * .08)
    end = max(start + 1, duration - start)
    for bpm in np.arange(65, 241, .5):
        spb = 60 / bpm
        offsets = np.linspace(0, spb, 18, endpoint=False)
        beats = np.arange(math.ceil(start / spb), math.floor(end / spb)) * spb
        if len(beats) < 4:
            continue
        scores = np.interp(beats[:, None] + offsets, times, weighted).mean(axis=0)
        i = int(scores.argmax())
        candidates.append((float(scores[i]), float(bpm), float(offsets[i])))
    # A coarse phase can miss a narrow kick by several milliseconds. Refining
    # only the coarse winner selected slow polyrhythmic aliases on hardtek.
    # Compare multiple distinct tempo candidates before choosing the winner.
    finalists=[]
    for candidate in sorted(candidates,reverse=True):
        if all(abs(candidate[1]-p[1])>2 for p in finalists):finalists.append(candidate)
        if len(finalists)>=18:break
    refined = []
    for best in finalists:
        for bpm in np.arange(best[1] - .55, best[1] + .551, .025):
            spb = 60 / bpm
            offsets = np.arange(0, spb, .002)
            beats = np.arange(math.ceil(start / spb), math.floor(end / spb)) * spb
            scores = np.interp(beats[:, None] + offsets, times, weighted).mean(axis=0)
            i = int(scores.argmax())
            refined.append((float(scores[i]), float(bpm), float(offsets[i])))
    score, bpm, offset = max(refined)
    if bpm < 125:
        # Resolve half-time ambiguity only when the double-tempo grid has almost
        # the same measured attack support. Sparse slow music stays slow.
        doubled=[c for c in refined if abs(c[1]-2*bpm)<.08 and c[0]>=score*.88]
        if doubled:score,bpm,offset=max(doubled)
    # Most sharp attacks occur before the STFT peak by approximately half a hop.
    offset = max(0., offset - HOP / SR / 2)
    spb = 60 / bpm
    baseline = max(float(weighted.mean()), 1e-6)
    confidence = min(.99, max(.05, (score / baseline - 1) / 2.5))
    offset,tracked=track_grid(times,weighted,bpm,offset,duration)
    beats = np.array(tracked)
    residuals = []
    for beat in beats:
        ids = np.flatnonzero((times >= beat - .05) & (times <= beat + .05))
        if len(ids) and bass_flux[ids].max() > np.percentile(bass_flux, 80):
            i = ids[int(bass_flux[ids].argmax())]
            residuals.append(abs(float(times[i] - HOP / SR / 2 - beat)))
    warning = 'Inspect the waveform before exporting; the beat grid is a detector estimate.'
    if residuals and float(np.median(residuals)) > .028:
        warning = 'The recording may drift or have a changing tempo. Add BPM markers in Studio.'
    if len(beats)>8 and np.max(np.abs(np.diff(beats)-spb))>.003:
        warning='Tempo drift detected. The local beat grid follows the audio; verify the phrasing in Studio.'
    return round(bpm, 3), round(offset, 5), round(confidence, 3), [round(float(b), 5) for b in beats], warning

def analyze(source, output_dir, identity=None, name=None):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = decode(source)
    duration = len(samples) / SR
    target = output_dir / 'song.ogg'
    conversion = convert(pathlib.Path(source), target)
    exported = decode(target)
    # A codec export must preserve the timeline. Compare aligned waveform slices.
    shifts = []
    for sec in sorted(set([1., duration * .25, duration * .5, duration * .85])):
        i = min(int(sec * SR), len(samples) - SR - 513)
        if i < 513:
            continue
        correlation = signal.correlate(exported[i-512:i+SR+512], samples[i:i+SR], mode='valid', method='fft')
        shifts.append(int(correlation.argmax()) - 512)
    if abs(len(samples) - len(exported)) > 2 or any(abs(s) > 1 for s in shifts):
        raise ValueError('The audio export changed timing; conversion was stopped.')
    frequencies, times, spectrum = signal.stft(samples, fs=SR, nperseg=1024,
        noverlap=1024-HOP, boundary='zeros')
    magnitude = np.abs(spectrum)
    data = {}
    for key, low, high in [('bass', 35, 200), ('high', 2000, 10000), ('all', 35, 10000)]:
        band = magnitude[(frequencies >= low) & (frequencies < high)]
        data[key] = np.sqrt((band ** 2).sum(axis=0))
        data[key + '_flux'] = np.maximum(np.diff(np.log1p(band * 100), axis=1, prepend=0), 0).sum(axis=0)
    bpm, offset, confidence, beat_times, warning = detect_grid(times, data['bass_flux'], data['all_flux'], duration)
    from engine.tempo import analyze_tempo
    tempo=analyze_tempo(times,data['bass_flux']+.10*data['all_flux'],bpm,offset,duration,beat_times)
    if tempo['changed']:
        beat_times=tempo['beatTimes']
        warning='Sustained local tempo changes detected. Review the tempo timeline and phrase alignment before exporting.'
    onset_env = _normalized(data['all_flux'])
    peaks, _ = signal.find_peaks(onset_env, distance=max(1, int(.035 * SR / HOP)), prominence=.10)
    onsets = [round(max(0., float(times[i]) - HOP / SR / 2), 5) for i in peaks]
    energies = _normalized(data['all'])
    bass = _normalized(data['bass_flux'])
    high = _normalized(data['high_flux'])
    # 46 frames/sec retain note-level rhythmic information; display data is smaller.
    frames = [{'time': round(float(times[i]), 5), 'bass': round(float(bass[i]), 4),
        'onset': round(float(onset_env[i]), 4), 'energy': round(float(energies[i]), 4),
        'high': round(float(high[i]), 4)} for i in range(0, len(times), 2)]
    from engine.music import analyze_music
    from engine.features import decode_audio
    music = analyze_music(samples, SR, bpm, offset, beat_times, spectrum=spectrum, times=times, frequencies=frequencies,stereo=decode_audio(target,channels=2))
    sections = music['sections']
    result = {'id': identity or output_dir.name, 'name': name or pathlib.Path(source).stem,
        'duration': round(duration, 6), 'bpm': bpm, 'offset': offset, 'confidence': confidence,
        'waveform': waveform(samples), 'onsets': onsets, 'sections': sections, 'frames': frames, 'musicAnalysis': music,
        'beatTimes': beat_times, 'beatMarkers':beat_times, 'sampleRate': SR, 'audioPath': str(target),
        'tempoTimeline':tempo['timeline'],'tempoAnalysis':{k:v for k,v in tempo.items() if k!='beatTimes'},
        'audioUrl': f'/api/audio/{identity or output_dir.name}/song.ogg',
        'conversion': conversion, 'timingWarning': warning,
        'exportValidation': {'sourceSamples': len(samples), 'exportSamples': len(exported), 'shiftSamples': shifts},
        'sourceSha256': hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()}
    (output_dir / 'analysis.json').write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
    return result

def public_analysis(result):
    return {k: v for k, v in result.items() if not k.startswith('_') and k not in ['frames', 'audioPath', 'sourceSha256', 'beatTimes', 'sourceEvidence']}
