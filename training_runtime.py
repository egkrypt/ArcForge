"""Truthful local runtime selection and corpus-aware training defaults."""
from __future__ import annotations
import contextlib, importlib.util, os, pathlib, subprocess, sys
from functools import lru_cache

ROOT = pathlib.Path(__file__).resolve().parent
GPU_DEPS = ROOT / '.gpu-deps'
if GPU_DEPS.is_dir() and str(GPU_DEPS) not in sys.path:
    sys.path.insert(0, str(GPU_DEPS))

class TrainingCancelled(RuntimeError):
    pass

def check_cancel(callback=None):
    if callback and callback(): raise TrainingCancelled('Cancelled by user; the previous model is unchanged.')

@lru_cache(maxsize=1)
def _probe_gpu():
    result = {'available':False,'name':None,'runtime':None,'reason':'PyTorch CUDA runtime is not installed'}
    try:
        out = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
        if out.returncode == 0 and out.stdout.strip(): result['detectedHardware']=out.stdout.strip().splitlines()[0]
    except (FileNotFoundError,subprocess.TimeoutExpired): pass
    if importlib.util.find_spec('torch') is None: return result
    try:
        import torch
        result['runtime']=f'PyTorch {torch.__version__} / CUDA {torch.version.cuda or "none"}'
        if torch.cuda.is_available():
            device=torch.cuda.get_device_properties(0)
            # A real allocation and multiplication confirms the driver/runtime works.
            x=torch.ones((4,4),device='cuda'); (x@x).sum().item()
            result.update(available=True,name=device.name,memoryBytes=device.total_memory,reason=None)
        else: result['reason']='CUDA is not available to this PyTorch runtime; CPU training remains available'
    except Exception as error: result['reason']=str(error)[:300]
    return result

def runtime_status(refresh=False):
    if refresh: _probe_gpu.cache_clear()
    gpu=dict(_probe_gpu())
    try:
        from threadpoolctl import threadpool_info
        libraries=[{k:lib.get(k) for k in ['internal_api','num_threads','version']} for lib in threadpool_info() if lib.get('user_api')=='blas']
    except ImportError: libraries=[]
    return {'availableBackends':['auto','cpu']+(['gpu'] if gpu['available'] else []),'autoBackend':'gpu' if gpu['available'] else 'cpu',
            'cpu':{'logicalCores':os.cpu_count() or 1,'libraries':libraries,'runtime':'NumPy vectorized BLAS'},'gpu':gpu,
            'usageTelemetry':'Runtime selection is measured. CPU/GPU utilization percentages are not simulated.'}

def select_runtime(backend='auto',threads=None,batch_size=None):
    backend=str(backend).lower()
    if backend not in ['auto','cpu','gpu']: raise ValueError('Training backend must be auto, cpu or gpu')
    status=runtime_status()
    selected=status['autoBackend'] if backend=='auto' else backend
    if selected=='gpu' and not status['gpu']['available']:
        raise ValueError('GPU training is unavailable: '+str(status['gpu']['reason']))
    cores=os.cpu_count() or 1
    count=min(4,cores) if threads is None else int(threads)
    if not 1<=count<=cores: raise ValueError(f'CPU threads must be between 1 and {cores}')
    batch=(4096 if selected=='gpu' else 1024) if batch_size is None else int(batch_size)
    if not 64<=batch<=32768: raise ValueError('Training batch size must be between 64 and 32768')
    return {'requestedBackend':backend,'backend':selected,'threads':count,'batchSize':batch,
            'device':status['gpu'].get('name') if selected=='gpu' else 'CPU','runtime':status['gpu']['runtime'] if selected=='gpu' else 'NumPy BLAS'}

def cpu_thread_limit(threads):
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=threads,user_api='blas')
    except ImportError: return contextlib.nullcontext()

def recommend_training(corpus):
    archives=corpus.get('archives',[])
    usable=sum(a.get('mapCount',a.get('maps',0))>0 and a.get('noteCount',a.get('notes',0))>=24 for a in archives) if isinstance(archives,list) else int(archives)
    notes=corpus.get('noteCount',corpus.get('examples',0))
    epochs=12 if usable<15 else 20 if usable<60 else 24
    return {'usableArchives':usable,'notes':int(notes or 0),'canTrain':usable>=3,'epochs':epochs,'minimumEpochs':8,'maximumEpochs':48,
            'backend':'auto','batchSize':4096 if runtime_status()['gpu']['available'] else 1024,'maxExamples':140000,'patience':6,
            'reason':f'{usable} usable songs: start with {epochs} epochs and retain the checkpoint with the lowest loss on unseen songs. Stop early after 6 epochs without improvement.',
            'qualityMeaning':'Validation loss measures unseen note prediction, not whether a map is fun or rankable.'}
