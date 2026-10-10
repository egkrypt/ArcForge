"""Compute policy for real local audio analysis, independent of map difficulty.

Deep spends extra inference on overlapping trained-model windows. It does not
invent instrument labels, alter note timings, or equate utilization with quality.
"""
from __future__ import annotations
import contextlib, math, os

COMPUTE_VERSION = '1.0'
COMPUTE_DEFAULTS = {'quality': 'balanced', 'device': 'auto', 'cpuThreads': 0}


def normalize_compute(settings=None):
    if settings is None:
        return COMPUTE_DEFAULTS.copy()
    if not isinstance(settings, dict):
        raise ValueError('Audio analysis compute settings must be an object.')
    quality = str(settings.get('quality', 'balanced')).lower()
    device = str(settings.get('device', settings.get('recognitionDevice', 'auto'))).lower()
    if quality not in {'balanced', 'deep'}:
        raise ValueError('Audio analysis quality must be balanced or deep.')
    if device not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('Audio analysis device must be auto, cpu or cuda.')
    raw_threads = settings.get('cpuThreads', 0)
    try:
        threads = int(raw_threads)
        if isinstance(raw_threads, bool) or float(raw_threads) != threads:
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise ValueError('Audio analysis CPU threads must be a whole number.') from None
    if not 0 <= threads <= max(1, os.cpu_count() or 1):
        raise ValueError(f'Audio analysis CPU threads must be 0 (automatic) or 1–{os.cpu_count() or 1}.')
    return {'quality': quality, 'device': device, 'cpuThreads': threads}


def analysis_compute_status():
    """Reuse the app's cached actual CUDA probe; do not infer GPU availability."""
    from training_runtime import runtime_status
    runtime = runtime_status()
    gpu = runtime['gpu']
    return {'version': COMPUTE_VERSION, 'defaults': COMPUTE_DEFAULTS.copy(),
            'qualities': ['balanced', 'deep'],
            'devices': ['auto', 'cpu'] + (['cuda'] if gpu['available'] else []),
            'cpu': {'logicalCores': os.cpu_count() or 1},
            'gpu': {key: gpu.get(key) for key in ['available', 'name', 'memoryBytes', 'runtime', 'reason']},
            'deep': {'windowSeconds': .96, 'hopSeconds': .12, 'balancedHopSeconds': .48,
                     'temporalViews': 4, 'automaticCpuThreads': os.cpu_count() or 1},
            'policy': 'CPU FFT and spectral features plus local CPU/CUDA inference. Batches reserve VRAM headroom; inference never changes observed onset times.',
            'limitations': ['Denser context windows improve temporal coverage, not guaranteed instrument accuracy.',
                            'The installed model still classifies mixed audio; compute mode does not add source separation.']}


def compute_plan(settings, torch):
    config = normalize_compute(settings)
    deep = config['quality'] == 'deep'
    requested = config['cpuThreads']
    # Balanced with zero retains the process's existing thread policy. Deep
    # intentionally makes all available CPU threads available for useful work.
    threads = requested or ((os.cpu_count() or 1) if deep else torch.get_num_threads())
    return config | {'version': COMPUTE_VERSION, 'threads': max(1, threads),
                     'hopFrames': 12 if deep else 48,
                     'hopSeconds': .12 if deep else .48,
                     'fftWorkers': max(1, threads) if deep else 1}


def initial_batch_size(torch, device, plan):
    """Conservative activation estimate, then OOM backoff in the actual runner."""
    if plan['quality'] != 'deep':
        return 32, None
    if device != 'cuda':
        return min(128, max(16, plan['threads'] * 8)), None
    free, total = torch.cuda.mem_get_info()
    # Leave room for the OS, editor, placement model, and convolution workspace.
    budget = min(int(free * .65), int(total * .50))
    capacity = max(1, budget // (8 * 1024 * 1024))
    count = min(512, 2 ** int(math.log2(capacity)))
    return count, {'freeBytesAtStart': int(free), 'totalBytes': int(total),
                   'activationBudgetBytes': budget, 'estimatedBytesPerPatch': 8 * 1024 * 1024,
                   'headroomPolicy': 'At most 65% of available / 50% of total VRAM for estimated activations; OOM halves the batch.'}


@contextlib.contextmanager
def analysis_thread_scope(torch, plan):
    """Bound temporary CPU pools and restore the existing app thread policy."""
    previous = torch.get_num_threads()
    change = plan['quality'] == 'deep' or bool(plan['cpuThreads'])
    try:
        from threadpoolctl import threadpool_limits
        scope = threadpool_limits(limits=plan['threads'], user_api='blas') if change else contextlib.nullcontext()
    except ImportError:
        scope = contextlib.nullcontext()
    try:
        from scipy.fft import set_workers
        if change and previous != plan['threads']:
            torch.set_num_threads(plan['threads'])
        with scope, set_workers(plan['fftWorkers']):
            yield
    finally:
        if change and torch.get_num_threads() != previous:
            torch.set_num_threads(previous)
