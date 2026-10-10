"""Local PyTorch execution of Google's Apache-2.0 YAMNet weights.

Architecture and signal parameters follow the TensorFlow Authors' YAMNet
implementation (2019), reimplemented for ArcForge's existing CUDA runtime.
https://github.com/tensorflow/models/tree/master/research/audioset/yamnet
The weights are never trained or invented by this module.
"""
from __future__ import annotations
import math, pathlib
import numpy as np

SAMPLE_RATE = 16000
WINDOW_SECONDS = .96
HOP_SECONDS = .48


def log_mel_patches(samples, hop_frames=48, workers=1):
    """Match TF's periodic Hann / magnitude STFT / HTK mel features."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if isinstance(hop_frames, bool) or not isinstance(hop_frames, int) or not 1 <= hop_frames <= 96:
        raise ValueError('YAMNet patch hop must be 1–96 spectrogram frames.')
    minimum = 15600  # .96 + final .025-.010 STFT window.
    hop_samples = hop_frames * 160
    length = minimum + max(0, math.ceil((len(samples) - minimum) / hop_samples)) * hop_samples
    samples = np.pad(samples, (0, max(0, length - len(samples))))
    frames = np.lib.stride_tricks.sliding_window_view(samples, 400)[::160]
    hann = (.5 - .5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    if workers > 1:
        from scipy.fft import rfft
        # Match the original double-precision FFT path before storing float32;
        # parallel CPU FFT is preprocessing, independent of inference device.
        spectrum = np.abs(rfft((frames * hann).astype(np.float64), n=512, axis=1, workers=workers)).astype(np.float32)
    else:
        spectrum = np.abs(np.fft.rfft(frames * hann, n=512, axis=1)).astype(np.float32)
    mel = lambda hz: 1127. * np.log1p(np.asarray(hz) / 700.)
    edges = np.linspace(mel(125.), mel(7500.), 66)
    bins = mel(np.linspace(0., 8000., 257))[:, None]
    lower = (bins - edges[:-2]) / (edges[1:-1] - edges[:-2])
    upper = (edges[2:] - bins) / (edges[2:] - edges[1:-1])
    weights = np.maximum(0., np.minimum(lower, upper)).astype(np.float32)
    weights[0] = 0.  # TensorFlow always excludes the DC spectrogram bin.
    feature = np.log(spectrum @ weights + .001).astype(np.float32)
    count = 1 + (len(feature) - 96) // hop_frames
    return np.stack([feature[i * hop_frames:i * hop_frames + 96] for i in range(count)])


def build_model(weight_path, device='cpu'):
    import torch
    from torch import nn
    import torch.nn.functional as F
    import h5py

    class SameConv(nn.Conv2d):
        def forward(self, value):
            height, width = value.shape[-2:]
            sh, sw = self.stride; kh, kw = self.kernel_size
            ph = max((math.ceil(height / sh) - 1) * sh + kh - height, 0)
            pw = max((math.ceil(width / sw) - 1) * sw + kw - width, 0)
            return super().forward(F.pad(value, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)))

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.blocks = nn.ModuleList(); self.weight_layers = []
            sizes = [(2, 32), (1, 64), (2, 128), (1, 128), (2, 256), (1, 256),
                     (2, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512), (2, 1024), (1, 1024)]
            channels = 1
            for index, (stride, out) in enumerate(sizes, 1):
                prefix = f'layer{index}'
                if index == 1:
                    conv = SameConv(channels, out, 3, stride=stride, bias=False)
                    bn = nn.BatchNorm2d(out, eps=1e-4)
                    block = nn.Sequential(conv, bn, nn.ReLU())
                    self.weight_layers.extend([(prefix + '/conv', conv, 'conv'), (prefix + '/conv/bn', bn, 'bn')])
                else:
                    depth = SameConv(channels, channels, 3, stride=stride, groups=channels, bias=False)
                    dbn = nn.BatchNorm2d(channels, eps=1e-4)
                    point = SameConv(channels, out, 1, bias=False)
                    pbn = nn.BatchNorm2d(out, eps=1e-4)
                    block = nn.Sequential(depth, dbn, nn.ReLU(), point, pbn, nn.ReLU())
                    self.weight_layers.extend([(prefix + '/depthwise_conv', depth, 'depth'), (prefix + '/depthwise_conv/bn', dbn, 'bn'),
                                               (prefix + '/pointwise_conv', point, 'conv'), (prefix + '/pointwise_conv/bn', pbn, 'bn')])
                self.blocks.append(block); channels = out
            self.dense = nn.Linear(1024, 521)

        def forward(self, patches):
            value = patches[:, None, :, :]
            for block in self.blocks: value = block(value)
            embedding = value.mean(dim=(2, 3))
            return torch.sigmoid(self.dense(embedding)), embedding

    model = Net()
    with h5py.File(pathlib.Path(weight_path), 'r') as saved:
        datasets = {}
        saved.visititems(lambda name, value: datasets.__setitem__(name, np.asarray(value)) if isinstance(value, h5py.Dataset) else None)
        def get(prefix, variable):
            suffix = prefix + '/' + variable + ':0'
            matches = [value for name, value in datasets.items() if name.endswith(suffix)]
            if len(matches) != 1: raise ValueError(f'YAMNet weight missing or ambiguous: {suffix}')
            return torch.from_numpy(matches[0].copy())
        with torch.no_grad():
            for prefix, layer, kind in model.weight_layers:
                if kind == 'bn':
                    layer.weight.fill_(1.)
                    layer.bias.copy_(get(prefix, 'beta'))
                    layer.running_mean.copy_(get(prefix, 'moving_mean'))
                    layer.running_var.copy_(get(prefix, 'moving_variance'))
                else:
                    kernel = get(prefix, 'depthwise_kernel' if kind == 'depth' else 'kernel')
                    layer.weight.copy_(kernel.permute(2, 3, 0, 1) if kind == 'depth' else kernel.permute(3, 2, 0, 1))
            # Keras releases assign dense/dense_1 differently. Identify the
            # unique 1024 x 521 trained head by shape, never by iteration order.
            heads = [value for name, value in datasets.items() if name.endswith('kernel:0') and value.shape == (1024, 521)]
            biases = [value for name, value in datasets.items() if name.endswith('bias:0') and value.shape == (521,)]
            if len(heads) != 1 or len(biases) != 1: raise ValueError('YAMNet 521-class head is missing or ambiguous')
            model.dense.weight.copy_(torch.from_numpy(heads[0].copy()).T)
            model.dense.bias.copy_(torch.from_numpy(biases[0].copy()))
    return model.to(device).eval()
