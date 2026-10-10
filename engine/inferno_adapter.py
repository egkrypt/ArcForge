"""Isolate InfernoSaber's real pretrained TensorFlow inference from the web server."""
from __future__ import annotations
import hashlib, json, os, pathlib, subprocess, sys, tempfile, importlib.util
import numpy as np
from .inferno_runtime import RUNTIME_VERSION

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'vendor/inferno/InfernoSaber---BeatSaber-Automapper-1.7.1'
WEIGHTS = ROOT / 'models/inferno-fav_15'
EXPECTED = {
    'tf_model_enc_16bneck_5_5__22_3.h5': '4270e16af9206cd00f28215c5a04c76384e1e3fdf0736c86b87d3e8c12cc6d0f',
    'tf_model_mapper_2-12_5_5__22_6.h5': 'afb5512fb2a63c36909c23a1eb5f3b769c0a9d41b6f2692a52079f4e19fda103',
    'notes_class_dict.pkl': '02f83ebecd3fa018e2950daa8d6dc986eb1e590b3b9e826699e37fed5ec527de',
    'onehot_encoder_beats.pkl': '45bce9a234dc50e4acaa13c8c98a2f022271644de05a8ba1e49109bddb626432',
}


def status():
    test = ROOT / 'models/inferno-runtime-status.json'
    required = ['tensorflow', 'tf_keras', 'setuptools', 'sklearn', 'PIL', 'scipy', 'imageio_ffmpeg']
    dependency_dirs = (ROOT / '.deps', ROOT / '.server-deps', ROOT.parent / '.tools')
    missing = [name for name in required if not any((directory / name).is_dir() for directory in dependency_dirs)
               and importlib.util.find_spec(name) is None]
    runtime = not missing
    try:
        record = json.loads(test.read_text('utf-8')) if test.exists() else None
    except (ValueError, OSError):
        record = None
    metadata = (record or {}).get('metadata', {})
    verified = bool((record or {}).get('success') and runtime
                    and metadata.get('runtimeVersion') == RUNTIME_VERSION
                    and metadata.get('pythonVersion') == list(sys.version_info[:2]))
    ready = all((WEIGHTS / n).exists() for n in EXPECTED) and runtime and SOURCE.is_dir()
    return {'name': 'InfernoSaber v1.7.1 / fav_15', 'source': str(SOURCE),
            'weights': 'BierHerr/InfernoSaber · fav_15', 'weightsAvailable': ready,'runtimeAvailable':runtime,'sourceAvailable':SOURCE.is_dir(),
            'inferenceVerified': verified, 'missingRuntimePackages': missing,
            'training': 'Frozen upstream network; local ArcForge placement network trains on uploaded maps',
            'runtime': 'TensorFlow 2.16 CPU / legacy Keras adapter',
            'lastTest': record}


def verify_weights():
    for filename, expected in EXPECTED.items():
        path = WEIGHTS / filename
        if not path.is_file(): raise FileNotFoundError(f'Missing InfernoSaber model file: {filename}')
        hasher = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b''): hasher.update(block)
        if hasher.hexdigest() != expected: raise ValueError(f'InfernoSaber weight checksum mismatch: {filename}')


def predict(audio_path, seconds, timeout=600):
    """Return real upstream marginal probabilities: [event, hand, 108 positions/cuts]."""
    readiness = status()
    if not readiness['weightsAvailable']:
        missing = ', '.join(readiness['missingRuntimePackages'])
        raise RuntimeError('InfernoSaber pretrained runtime is not installed'
                           + (f' (missing: {missing}). Run Setup ArcForge.ps1 -Inferno.' if missing else ''))
    times = np.asarray(seconds, np.float32)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError('InfernoSaber event times must be a nonempty, finite, sorted list of seconds')
    cache = ROOT / 'models/inference-cache'; cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(pathlib.Path(audio_path).read_bytes()).hexdigest()[:24]
    # Worker/runtime changes must not silently reuse an older inference record.
    identity = RUNTIME_VERSION.encode() + bytes(sys.version_info[:2]) + times.tobytes()
    digest += '-' + hashlib.sha256(identity).hexdigest()[:16]
    result_path = cache / (digest + '.npz')
    if not result_path.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            request = pathlib.Path(temp) / 'request.json'
            request.write_text(json.dumps({'audioPath': str(audio_path), 'seconds': list(map(float, seconds)), 'output': str(result_path)}), 'utf-8')
            env = os.environ.copy(); env['TF_USE_LEGACY_KERAS'] = '1'; env['TF_CPP_MIN_LOG_LEVEL'] = '2'
            env['OMP_NUM_THREADS'] = '4'
            env['CUDA_VISIBLE_DEVICES'] = '-1'  # Separate frozen TF worker; GPU is available to the local trainer.
            process = subprocess.run([sys.executable, str(ROOT / 'engine/inferno_worker.py'), str(request)],
                                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
            if process.returncode:
                (ROOT / 'models/inferno-runtime-status.json').write_text(json.dumps({
                    'success': False, 'runtimeVersion': RUNTIME_VERSION,
                    'error': process.stderr[-2500:]}, indent=2), 'utf-8')
                raise RuntimeError('InfernoSaber inference failed: ' + process.stderr[-2500:])
    with np.load(result_path, allow_pickle=False) as loaded:
        probabilities = loaded['probabilities']
        metadata = json.loads(str(loaded['metadata']))
    if probabilities.shape != (len(times), 2, 108) or not np.isfinite(probabilities).all():
        raise RuntimeError('InfernoSaber returned invalid placement probabilities')
    return probabilities, metadata
