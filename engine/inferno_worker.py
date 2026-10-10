"""Real InfernoSaber model runner.

Uses upstream spectrogram preprocessing, batch shifter and class decoder directly
from the pinned source. ffmpeg replaces only aubio's audio decoder, which lacks a
Python 3.12 Windows wheel. The neural encoder and mapping network are unmodified.
"""
from __future__ import annotations
import ast, hashlib, json, os, pathlib, pickle, sys, types, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# engine.__init__ imports the local NumPy mapper before the runtime helper.
sys.path.insert(0, str(ROOT / '.deps'))
sys.path.insert(1, str(ROOT.parent / '.tools'))
from engine.inferno_runtime import initialize_runtime, normalize_probabilities, RUNTIME_VERSION
initialize_runtime(ROOT)
os.environ['TF_USE_LEGACY_KERAS'] = '1'; os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
try:
    import distutils  # Provided by setuptools on Python 3.12, not a copied stdlib.
except ModuleNotFoundError as error:
    raise RuntimeError('InfernoSaber needs setuptools for TensorFlow on Python 3.12. '
                       'Run Setup ArcForge.ps1 -Inferno in the ArcForge folder.') from error
import numpy as np
import tensorflow as tf
import tf_keras
from PIL import Image
from scipy import signal
from engine.features import decode_audio
from engine.inferno_adapter import SOURCE, WEIGHTS, verify_weights


def upstream_functions(file, names, namespace):
    tree = ast.parse((SOURCE / file).read_text('utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names): raise RuntimeError('Pinned upstream function is missing')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE / file), 'exec'), namespace)
    return namespace


def run(request):
    started = time.time(); verify_weights()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    # These pickles are only loaded from hash-verified upstream weights, never user uploads.
    keys = pickle.load((WEIGHTS / 'notes_class_dict.pkl').open('rb'))
    encoder = pickle.load((WEIGHTS / 'onehot_encoder_beats.pkl').open('rb'))
    categories = [int(v) for v in encoder.categories_[0]]
    config = types.SimpleNamespace(samplerate_music=14800, window=2., specgram_res=24)
    namespace = {'np': np, 'signal': signal, 'Image': Image, 'config': config}
    upstream_functions('preprocessing/music_processing.py', {'log_specgram', 'process_song'}, namespace)
    upstream_functions('lighting_prediction/train_lighting.py', {'lstm_shift_events_half'}, namespace)
    upstream_functions('map_creation/map_creator.py', {'decode_class_keys'}, namespace)
    audio = decode_audio(request['audioPath'], 14800)
    seconds = np.array(request['seconds'], np.float32)
    # Include 16 pre-roll contexts so the upstream next-batch architecture predicts
    # the beginning of the requested score instead of dropping its first 16 events.
    padded_times = np.r_[np.maximum(0, seconds[0] - np.arange(16, 0, -1) * .16), seconds]
    pad_count = (-len(padded_times)) % 16
    if pad_count: padded_times = np.r_[padded_times, np.repeat(padded_times[-1], pad_count)]
    windows = []
    for second in padded_times:
        sample = int(round(float(second) * 14800)); start = sample - 14800; end = sample + 14800
        section = audio[max(0, start):min(len(audio), end)]
        windows.append(np.pad(section, (max(0, -start), max(0, end - len(audio))))[:29600])
    images = namespace['process_song'](np.array(windows))
    images = (images - images.min()) / max(float(images.max() - images.min()), 1e-6)
    images = images.reshape((-1, 24, 20, 1)).astype(np.float32)
    enc_model = tf_keras.models.load_model(next(WEIGHTS.glob('tf_model_enc*.h5')), compile=False)
    encoded = np.asarray(enc_model(images, training=False))
    mapper = tf_keras.models.load_model(next(WEIGHTS.glob('tf_model_mapper*.h5')), compile=False)
    differences = np.r_[1., np.diff(padded_times)].astype(np.float32)
    inputs, _ = namespace['lstm_shift_events_half'](encoded, differences, None, 16)
    marginalizer = np.zeros((len(categories), 2, 108), np.float32)
    for class_index, key in enumerate(categories):
        decoded = namespace['decode_class_keys'](keys[key])
        for i in range(0, len(decoded) - 3, 4):
            x, y, hand, direction = decoded[i:i + 4]
            if hand in [0, 1] and 0 <= x < 4 and 0 <= y < 3 and 0 <= direction < 9:
                marginalizer[class_index, hand, (x * 3 + y) * 9 + direction] += 1
    history = np.zeros((1, 16, len(categories)), np.float32)
    outputs = []
    for index in range(len(inputs[0])):
        prediction = np.asarray(mapper([inputs[0][index:index + 1], inputs[1][index:index + 1], history], training=False))[0].copy()
        # The upstream output uses sigmoid. Normalize its nonnegative evidence per time step.
        prediction = normalize_probabilities(prediction)
        outputs.append(np.einsum('tc,chp->thp', prediction, marginalizer))
        winners = prediction.argmax(1)
        history.fill(0); history[0, np.arange(16), winners] = 1
    result = np.concatenate(outputs, axis=0)[:len(seconds)]
    result = normalize_probabilities(result)
    metadata = {'backend': 'InfernoSaber v1.7.1 / fav_15 pretrained neural inference', 'sourceVersion': 'v1.7.1',
                'runtimeVersion': RUNTIME_VERSION, 'pythonVersion': list(sys.version_info[:2]),
                'tensorflowVersion': tf.__version__, 'kerasVersion': tf_keras.__version__,
                'emptyHandDistributions': int(np.count_nonzero(result.sum(-1) == 0)),
                'events': len(result), 'seconds': round(time.time() - started, 2),
                'inputShape': list(mapper.input_shape), 'outputClasses': len(categories),
                'upstreamFunctions': ['process_song', 'lstm_shift_events_half', 'decode_class_keys'],
                'audioDecoderAdaptation': 'ffmpeg mono 14800 Hz replaces aubio source',
                'neuralWeightsModified': False}
    pathlib.Path(request['output']).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(request['output'], probabilities=result, metadata=json.dumps(metadata))
    (ROOT / 'models/inferno-runtime-status.json').write_text(json.dumps({'success': True, 'metadata': metadata}, indent=2), 'utf-8')
    print(json.dumps(metadata))


if __name__ == '__main__':
    run(json.loads(pathlib.Path(sys.argv[1]).read_text('utf-8')))
