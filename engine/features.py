"""Shared audio contexts and native BPM-event time conversion."""
from __future__ import annotations
import pathlib, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARENT_TOOLS = ROOT.parent / '.tools'
if str(PARENT_TOOLS) not in sys.path:
    sys.path.append(str(PARENT_TOOLS))


def decode_audio(path, sample_rate=22050, channels=1):
    import imageio_ffmpeg
    result = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(path),
                             '-f', 'f32le', '-ac', str(channels), '-ar', str(sample_rate), '-'], capture_output=True, check=True,timeout=240)
    samples=np.frombuffer(result.stdout, dtype='<f4').copy()
    return samples if channels==1 else samples.reshape(-1,channels)


def audio_context(path):
    from scipy.signal import stft
    y = decode_audio(path)
    frequencies, times, values = stft(y, 22050, nperseg=1024, noverlap=768, boundary='zeros')
    power = np.abs(values).astype(np.float32)
    bands = [power.mean(axis=0), power[frequencies < 180].mean(axis=0),
             np.maximum(np.diff(power, axis=1, prepend=power[:, :1]), 0).mean(axis=0),
             power[frequencies > 3000].mean(axis=0)]
    contexts = np.stack(bands, axis=1)
    contexts /= np.maximum(np.quantile(contexts, .95, axis=0), 1e-7)
    return times.astype(np.float32), np.clip(contexts, 0, 3).astype(np.float32)


def note_seconds(beat, bpm, offset=0, events=None):
    last_beat, seconds, tempo = 0., float(offset), float(bpm)
    for event in sorted(events or [], key=lambda e: e.get('b', 0)):
        b, new_tempo = float(event.get('b', 0)), float(event.get('m', bpm))
        if b > beat: break
        if b >= last_beat and 20 <= new_tempo <= 1000:
            seconds += (b - last_beat) * 60 / tempo
            last_beat, tempo = b, new_tempo
    return seconds + (beat - last_beat) * 60 / tempo


def sample_context(times, values, seconds):
    if not len(times): return np.zeros(4, np.float32)
    return np.array([np.interp(seconds, times, values[:, i]) for i in range(4)], np.float32)


FEATURE_SIZE = 44
TOKEN_COUNT = 108


def token(note):
    return (int(note['x']) * 3 + int(note['y'])) * 9 + int(note['d'])


def untoken(value):
    pos, direction = divmod(int(value), 9)
    x, y = divmod(pos, 3)
    return {'x': x, 'y': y, 'd': direction}


def placement_features(history, other_history, hand, beat, gap_seconds, bpm, nps, context):
    """44 inputs: audio, musical position, difficulty, and two swings of both hands."""
    f = np.zeros(FEATURE_SIZE, np.float32)
    f[:4] = context
    f[4:8] = [hand, min(gap_seconds, 2) / 2, bpm / 300, nps / 14]
    f[8:12] = [np.sin(beat * np.pi / 2), np.cos(beat * np.pi / 2),
               np.sin(beat * np.pi / 8), np.cos(beat * np.pi / 8)]
    index = 12
    for notes in [history[-2:], other_history[-2:]]:
        padded = [None] * (2 - len(notes)) + notes
        for note in padded:
            if note:
                f[index:index + 3] = [note['x'] / 3, note['y'] / 2, 1]
                # Low-dimensional circular cut encoding avoids treating directions as ordinal.
                angles = [90, 270, 180, 0, 135, 45, 225, 315, 0]
                angle = angles[note['d']] * np.pi / 180
                f[index + 3:index + 8] = [np.sin(angle), np.cos(angle), note['d'] == 8,
                                        note['d'] in [0, 4, 5], note['d'] in [1, 6, 7]]
            index += 8
    return f
