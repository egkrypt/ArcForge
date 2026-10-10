"""A real locally trained neural placement model with archive-held-out validation.

This model is separate from InfernoSaber's pretrained network. Training this model
does not claim to fine-tune the upstream TensorFlow weights.
"""
from __future__ import annotations
import collections, hashlib, json, pathlib, tempfile, time, zipfile
import numpy as np
from training_runtime import check_cancel, select_runtime, cpu_thread_limit, runtime_status
from .dataset import read_archive, scan_dataset, file_digest
from .features import (audio_context, note_seconds, sample_context, placement_features,
                       token, FEATURE_SIZE, TOKEN_COUNT)

# Changing import/cache plumbing does not change extracted training examples.
# Bump this explicit schema key when placement_features or _collect semantics change.
FEATURE_SCHEMA_VERSION = 'a627f09a7a14a930'


def model_status(model_dir):
    folder = pathlib.Path(model_dir)
    report = folder / 'report.json'
    result = json.loads(report.read_text('utf-8')) if report.exists() else {'state': 'untrained', 'model': 'ArcForge audio-conditioned neural placement'}
    result['ready'] = (folder / 'model.npz').exists()
    composition_report=folder/'composition-report.json'
    if composition_report.is_file():result['composition']=json.loads(composition_report.read_text('utf-8'))
    from .inferno_adapter import status as inferno_status
    result['inferno'] = inferno_status()
    return result


def _collect(folder, model_dir, progress, cancel_callback=None):
    X, Y, source_ids, rhythm = [], [], [], collections.Counter()
    double_counts, slot_counts, light_rates, light_colors = collections.Counter(), collections.Counter(), [], collections.Counter()
    provenance, errors, audio_used = [], [], 0
    cache = pathlib.Path(model_dir) / 'audio-cache'; cache.mkdir(parents=True, exist_ok=True)
    files = sorted(pathlib.Path(folder).glob('*.zip'))
    feature_cache = pathlib.Path(model_dir) / 'feature-cache'; feature_cache.mkdir(parents=True, exist_ok=True)
    extractor_version=FEATURE_SCHEMA_VERSION
    for archive_index, path in enumerate(files):
        check_cancel(cancel_callback)
        if progress: progress({'phase': 'dataset', 'progress': round(archive_index / max(1, len(files)) * .45, 3), 'message': f'Reading {path.name}'})
        try:
            digest=file_digest(path)
            extracted=feature_cache/(digest[:24]+'-'+extractor_version+'.npz')
            if extracted.exists():
                with np.load(extracted,allow_pickle=False) as saved:
                    meta=json.loads(str(saved['metadata']))
                    if saved['X'].ndim!=2 or saved['X'].shape[1]!=FEATURE_SIZE or len(saved['X'])!=len(saved['Y']):raise ValueError('Invalid local feature cache')
                    X.extend(saved['X']);Y.extend(saved['Y'].tolist());source_ids.extend([archive_index]*len(saved['Y']))
                    for k,v in meta['rhythm']:rhythm[tuple(k)]+=v
                    for k,v in meta['slots']:slot_counts[tuple(k)]+=v
                    for k,v in meta['doubles']:double_counts[tuple(k)]+=v
                    light_rates.extend(meta['lightRates'])
                    for k,v in meta['lightColors']:light_colors[tuple(k)]+=v
                    audio_used+=meta['audioUsed']
                    provenance.append(meta['provenance']|{'archive':path.name,'sourceIndex':archive_index,'featureCache':'hit'})
                continue
            x_start=len(X);rhythm_before=rhythm.copy();slot_before=slot_counts.copy();double_before=double_counts.copy();light_before=light_colors.copy();light_start=len(light_rates);audio_before=audio_used
            data = read_archive(path)
            cache_file = cache / (data['sha256'][:16] + '.npz')
            if cache_file.exists():
                cached = np.load(cache_file); times, contexts = cached['times'], cached['values']
            elif data['audioEntry']:
                with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory(dir=cache) as temp:
                    audio_path = pathlib.Path(temp) / 'source.ogg'
                    audio_path.write_bytes(z.read(data['audioEntry']))
                    times, contexts = audio_context(audio_path)
                np.savez_compressed(cache_file, times=times, values=contexts)
            else:
                times, contexts = np.array([], np.float32), np.empty((0, 4), np.float32)
            audio_used += bool(len(times))
            light_rates.extend(data['lighting']['eventRatesPerSecond'])
            for color in data['lighting']['commonColors']: light_colors[tuple(color['rgb'])] += color['count']
            for chart in data['maps']:
                notes = chart['notes']; histories = {0: [], 1: []}
                duration = max(note_seconds(notes[-1]['b'], chart['bpm'], chart['offset'], chart['bpmEvents']), 1)
                nps = len(notes) / duration
                grouped = collections.defaultdict(set)
                for note in notes: grouped[round(note['b'], 4)].add(note['c'])
                for beat, hands in grouped.items():
                    sec = note_seconds(beat, chart['bpm'], chart['offset'], chart['bpmEvents'])
                    energy = sample_context(times, contexts, sec)[0]
                    key = (min(2, int(energy * 2)), int(round(beat * 4)) % 16)
                    slot_counts[key] += 1
                    double_counts[key] += len(hands) == 2
                for note_index,note in enumerate(notes):
                    if note_index%4096==0:check_cancel(cancel_callback)
                    hand = note['c']; history = histories[hand]
                    sec = note_seconds(note['b'], chart['bpm'], chart['offset'], chart['bpmEvents'])
                    gap = sec - history[-1]['seconds'] if history else 1
                    context = sample_context(times, contexts, sec)
                    if .10 <= gap <= 4:
                        X.append(placement_features(history, histories[1 - hand], hand, note['b'], gap, chart['bpm'], nps, context))
                        Y.append(token(note)); source_ids.append(archive_index)
                        energy_bin = min(2, int(context[0] * 2))
                        rhythm[(energy_bin, int(round(note['b'] * 4)) % 16, hand)] += 1
                    histories[hand].append(note | {'seconds': sec})
            record={'archive': path.name, 'sourceIndex': archive_index, 'sha256': data['sha256'], 'maps': len(data['maps']), 'notes': sum(len(c['notes']) for c in data['maps']), 'excluded': data['excluded'],'featureCache':'created'}
            provenance.append(record)
            meta={'provenance':record,'rhythm':[[list(k),v] for k,v in (rhythm-rhythm_before).items()],
                  'slots':[[list(k),v] for k,v in (slot_counts-slot_before).items()], 'doubles':[[list(k),v] for k,v in (double_counts-double_before).items()],
                  'lightColors':[[list(k),v] for k,v in (light_colors-light_before).items()], 'lightRates':light_rates[light_start:],'audioUsed':audio_used-audio_before}
            check_cancel(cancel_callback)
            temp=extracted.with_suffix('.pending.npz')
            np.savez_compressed(temp,X=np.asarray(X[x_start:],dtype=np.float32).reshape(-1,FEATURE_SIZE),Y=np.asarray(Y[x_start:],dtype=np.int32),metadata=json.dumps(meta))
            temp.replace(extracted)
        except Exception as e:
            from training_runtime import TrainingCancelled
            if isinstance(e,TrainingCancelled):raise
            errors.append({'archive': path.name, 'error': str(e)})
    if not X: raise ValueError('No usable human mapping examples were found')
    doubles = np.zeros((3, 16), np.float32)
    for key, count in slot_counts.items(): doubles[key] = double_counts[key] / max(count, 1)
    lighting = {'medianEventsPerSecond': float(np.median(light_rates)) if light_rates else 0,
                'p90EventsPerSecond': float(np.quantile(light_rates, .9)) if light_rates else 0,
                'commonColors': [{'rgb': list(k), 'count': v} for k, v in light_colors.most_common(12)]}
    return np.stack(X), np.array(Y, np.int32), np.array(source_ids), provenance, errors, rhythm, audio_used, doubles, lighting


def _softmax(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    value = np.exp(logits)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), 1e-8)


def train_model(dataset_folder, model_dir, progress_callback=None, epochs=24, backend='auto', threads=None,
                cancel_callback=None, batch_size=None, max_examples=140000, patience=6):
    """Train a checkpoint on CPU BLAS or real CUDA, never silently fake GPU use."""
    runtime=select_runtime(backend,threads,batch_size)
    if not 1<=int(epochs)<=200:raise ValueError('Epochs must be between 1 and 200')
    if not 1000<=int(max_examples)<=1000000:raise ValueError('max_examples must be between 1000 and 1000000')
    if not 1<=int(patience)<=200:raise ValueError('Patience must be between 1 and 200')
    with cpu_thread_limit(runtime['threads']):
        return _train(dataset_folder,model_dir,progress_callback,int(epochs),runtime,cancel_callback,int(max_examples),int(patience))

def _train(dataset_folder, model_dir, progress_callback, epochs, runtime, cancel_callback, max_examples, patience):
    start = time.time(); folder = pathlib.Path(model_dir); folder.mkdir(parents=True, exist_ok=True)
    check_cancel(cancel_callback)
    if progress_callback:progress_callback({'phase':'runtime','progress':0,'message':f"Training on {runtime['device']} · {runtime['runtime']}",'runtime':runtime})
    X, Y, source_ids, provenance, errors, rhythm, audio_used, doubles, lighting = _collect(dataset_folder, folder, progress_callback,cancel_callback)
    # Every difficulty of one song stays in the same split. There is no map leakage.
    archives = sorted(set(source_ids.tolist()))
    if len(archives) < 3: raise ValueError('Train with at least three distinct archives for held-out validation')
    # Content hashes keep the split stable when names change or ZIPs are added.
    hashes={p['sourceIndex']:p['sha256'] for p in provenance}
    held_out={i for i in archives if int(hashes[i][:8],16)%5==0}
    if not held_out:held_out={min(archives,key=lambda i:hashes[i])}
    if len(held_out)==len(archives):held_out.remove(max(archives,key=lambda i:hashes[i]))
    split_digest=hashlib.sha256('|'.join(sorted(hashes[i] for i in held_out)).encode()).hexdigest()
    train_mask = np.array([i not in held_out for i in source_ids])
    validation_mask = ~train_mask
    rng = np.random.default_rng(190205)
    mean, std = X[train_mask].mean(0), np.maximum(X[train_mask].std(0), .1)
    X = np.clip((X - mean) / std, -6, 6)
    Xt, Yt = X[train_mask], Y[train_mask]; Xv, Yv = X[validation_mask], Y[validation_mask]
    # A bounded dataset keeps upload-based retraining responsive on CPU.
    if len(Xt) > max_examples:
        choose = rng.choice(len(Xt), max_examples, replace=False); Xt, Yt = Xt[choose], Yt[choose]
    hidden = 96
    params = [rng.normal(0, .08, (FEATURE_SIZE, hidden)).astype(np.float32), np.zeros(hidden, np.float32),
              rng.normal(0, .05, (hidden, TOKEN_COUNT)).astype(np.float32), np.zeros(TOKEN_COUNT, np.float32)]
    counts = np.bincount(Yt, minlength=TOKEN_COUNT).astype(np.float32)
    params[3] = np.log((counts + 1) / (counts.sum() + TOKEN_COUNT)).astype(np.float32)
    # This is genuine additional training: warm-start prior learned network when available.
    existing = folder / 'model.npz'
    resumed = False
    if existing.exists():
        with np.load(existing) as prior:
            if prior['w1'].shape == params[0].shape and 'splitDigest' in prior and str(prior['splitDigest'])==split_digest:
                params = [prior[k].copy() for k in ['w1', 'b1', 'w2', 'b2']]
                # Change feature normalization without changing the old network's
                # un-clipped outputs. Never warm-start a different holdout split.
                params[1] += ((mean-prior['mean'])/prior['std']) @ params[0]
                params[0] *= (std/prior['std'])[:,None]
                resumed = True
    moments = [np.zeros_like(p) for p in params]; squares = [np.zeros_like(p) for p in params]
    history, step, best_loss, best = [], 0, float('inf'), None
    if resumed:
        initial=_softmax(np.maximum(Xv @ params[0]+params[1],0) @ params[2]+params[3])
        train_initial=_softmax(np.maximum(Xt @ params[0]+params[1],0) @ params[2]+params[3])
        initial_training_loss=float(-np.log(np.maximum(train_initial[np.arange(len(Yt)),Yt],1e-8)).mean())
        best_loss=float(-np.log(np.maximum(initial[np.arange(len(Yv)),Yv],1e-8)).mean())
        best=[p.copy() for p in params]
        history.append({'epoch':0,'trainingLoss':round(initial_training_loss,4),'validationLoss':round(best_loss,4),
                        'validationAccuracy':round(float((initial.argmax(1)==Yv).mean()),4),'checkpoint':'Before additional training'})
    gpu=None
    if runtime['backend']=='gpu':
        gpu=_CudaTrainer(Xt,Yt,Xv,Yv,params,runtime['batchSize'])
    stale=0
    for epoch in range(int(epochs)):
        check_cancel(cancel_callback);epoch_start=time.time()
        order = rng.permutation(len(Xt))
        total_loss = 0
        if gpu is not None:
            total_loss,validation_loss,accuracy=gpu.epoch(order,cancel_callback)
            params=gpu.numpy_parameters()
        for i in range(0,len(order),runtime['batchSize']) if gpu is None else []:
            check_cancel(cancel_callback)
            ids = order[i:i + runtime['batchSize']]; bx, by = Xt[ids], Yt[ids]
            w1, b1, w2, b2 = params
            hidden_values = np.maximum(bx @ w1 + b1, 0)
            probs = _softmax(hidden_values @ w2 + b2)
            total_loss += float(-np.log(np.maximum(probs[np.arange(len(by)), by], 1e-8)).sum())
            probs[np.arange(len(by)), by] -= 1; probs /= len(by)
            grad2 = hidden_values.T @ probs + .0002 * w2
            dh = (probs @ w2.T) * (hidden_values > 0)
            grads = [bx.T @ dh + .0002 * w1, dh.sum(0), grad2, probs.sum(0)]
            step += 1
            for j, grad in enumerate(grads):
                moments[j] = .9 * moments[j] + .1 * grad
                squares[j] = .999 * squares[j] + .001 * grad * grad
                params[j] -= .0015 * (moments[j] / (1 - .9 ** step)) / (np.sqrt(squares[j] / (1 - .999 ** step)) + 1e-8)
        if gpu is None:
            w1, b1, w2, b2 = params
            pv = _softmax(np.maximum(Xv @ w1 + b1, 0) @ w2 + b2)
            validation_loss = float(-np.log(np.maximum(pv[np.arange(len(Yv)), Yv], 1e-8)).mean())
            accuracy = float((pv.argmax(1) == Yv).mean())
        elapsed=time.time()-epoch_start
        record = {'epoch': epoch + 1, 'trainingLoss': round(total_loss / len(Yt), 4), 'validationLoss': round(validation_loss, 4), 'validationAccuracy': round(accuracy, 4),
                  'epochSeconds':round(elapsed,3),'examplesPerSecond':round(len(Yt)/max(elapsed,.001),1),'backend':runtime['backend']}
        history.append(record)
        if validation_loss < best_loss:
            best_loss, best = validation_loss, [p.copy() for p in params];stale=0
        else:stale+=1
        if progress_callback:
            progress_callback({'phase': 'training', 'progress': .45 + .5 * (epoch + 1) / epochs,
                               'message': f"Epoch {epoch + 1}/{epochs} · held-out accuracy {accuracy:.1%}", 'metrics': record,'runtime':runtime})
        if stale>=patience:break
    check_cancel(cancel_callback)
    rhythm_values = np.zeros((3, 16, 2), np.float32)
    for (energy, phase, hand), count in rhythm.items(): rhythm_values[energy, phase, hand] = count
    rhythm_values /= np.maximum(rhythm_values.max(axis=1, keepdims=True), 1)
    from .composition import train_composition
    def composition_progress(value):
        if progress_callback:progress_callback(value|{'progress':.95+.04*float(value.get('progress',0))})
    composition=train_composition(dataset_folder,folder,progress=composition_progress,cancel_callback=cancel_callback)
    np.savez_compressed(folder / 'model.tmp.npz', w1=best[0], b1=best[1], w2=best[2], b2=best[3], mean=mean, std=std, rhythm=rhythm_values, doubles=doubles, splitDigest=split_digest)
    (folder / 'model.tmp.npz').replace(existing)
    baseline = (counts + 1) / (counts.sum() + TOKEN_COUNT)
    baseline_loss = float(-np.log(baseline[Yv]).mean())
    report = {'state': 'trained', 'model': 'ArcForge audio-conditioned neural placement', 'version': 1,
              'architecture': '44 audio/history/phrase inputs → 96 ReLU → 108 note classes',
              'sourceFolder': str(dataset_folder), 'archives': len(provenance), 'maps': sum(p['maps'] for p in provenance),
              'examples': len(X), 'trainingExamples': len(Xt), 'validationExamples': len(Yv),
              'audioConditionedArchives': int(audio_used), 'heldOutArchives': [files['archive'] for files in provenance if files['sourceIndex'] in held_out],
              'heldOutSplit': 'Entire ZIP/song held out, including every difficulty', 'baselineValidationLoss': round(baseline_loss, 4),
              'bestValidationLoss': round(best_loss, 4), 'history': history, 'warmStarted': resumed,
              'trainingSeconds': round(time.time() - start, 2), 'provenance': provenance, 'errors': errors, 'lighting': lighting,'composition':composition,
              'runtime':runtime,'requestedEpochs':epochs,'completedEpochs':len([h for h in history if h['epoch']>0]),'earlyStopped':stale>=patience,
              'featureCacheHits':sum(p.get('featureCache')=='hit' for p in provenance),'patience':patience,
              'qualityClaim': 'Held-out prediction measures are shown; superior gameplay is not established by training loss.'}
    (folder / 'report.json').write_text(json.dumps(report, indent=2), 'utf-8')
    if progress_callback: progress_callback({'phase': 'complete', 'progress': 1, 'message': 'Local model saved', 'metrics': history[-1]})
    return report


class _CudaTrainer:
    """Same dense architecture and Adam objective, computed on a real CUDA device."""
    def __init__(self,Xt,Yt,Xv,Yv,params,batch_size):
        import os
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        self.torch=torch;self.batch_size=batch_size
        torch.manual_seed(190205);torch.cuda.manual_seed_all(190205)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.Xt=torch.as_tensor(Xt,dtype=torch.float32,device='cuda');self.Yt=torch.as_tensor(Yt,dtype=torch.long,device='cuda')
        self.Xv=torch.as_tensor(Xv,dtype=torch.float32,device='cuda');self.Yv=torch.as_tensor(Yv,dtype=torch.long,device='cuda')
        self.params=[torch.tensor(p,dtype=torch.float32,device='cuda',requires_grad=True) for p in params]
        self.optimizer=torch.optim.Adam([{'params':[self.params[0],self.params[2]],'weight_decay':.0002},
                                         {'params':[self.params[1],self.params[3]],'weight_decay':0}],lr=.0015,betas=(.9,.999),eps=1e-8)
    def logits(self,x):
        w1,b1,w2,b2=self.params
        return self.torch.relu(x@w1+b1)@w2+b2
    def epoch(self,order,cancel_callback):
        torch=self.torch;losses=[]
        for i in range(0,len(order),self.batch_size):
            check_cancel(cancel_callback);ids=torch.as_tensor(order[i:i+self.batch_size],device='cuda',dtype=torch.long)
            self.optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(self.logits(self.Xt[ids]),self.Yt[ids]);loss.backward();self.optimizer.step()
            losses.append(loss.detach()*len(ids))
        with torch.no_grad():
            validation=self.logits(self.Xv)
            loss=torch.nn.functional.cross_entropy(validation,self.Yv).item()
            accuracy=(validation.argmax(1)==self.Yv).float().mean().item()
            total=torch.stack(losses).sum().item()
        return total,loss,accuracy
    def numpy_parameters(self):return [p.detach().cpu().numpy().copy() for p in self.params]
