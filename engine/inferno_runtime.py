"""Initialize only ArcForge's trusted dependency directories for the TF worker.

Python 3.12 removed stdlib distutils. TensorFlow 2.16 still imports it, so
setuptools' installed .pth hook must run when packages live in a --target folder.
Merely adding that folder to sys.path skips normal site-package initialization.
"""
from __future__ import annotations

import pathlib
import site
import sys

RUNTIME_VERSION = '1.2-site-packages'


def initialize_runtime(root: pathlib.Path):
    """Process dependency .pth files, preserving the worker's package precedence.

These are application dependency folders, never uploaded maps or user audio.
Normal app-local virtual environments already process their own site directory.
"""
    directories = (root / '.deps', root / '.server-deps', root.parent / '.tools')
    for directory in directories:
        if directory.is_dir():
            # addsitedir also processes .pth files when the directory was already
            # placed on sys.path by bootstrap.py or another application entry point.
            site.addsitedir(str(directory))
    for directory in reversed(directories):
        if directory.is_dir():
            value = str(directory)
            if value in sys.path:
                sys.path.remove(value)
            sys.path.insert(0, value)


def normalize_probabilities(values):
    """Normalize small sigmoid evidence without suppressing an entire hand.

The network can emit probabilities far below 1e-8. A fixed 1e-8 denominator
turns those valid distributions into nearly zero weight in Hybrid placement.
An entirely empty row stays empty rather than inventing model evidence.
"""
    import numpy as np
    evidence = np.asarray(values, np.float64)
    totals = evidence.sum(axis=-1, keepdims=True)
    result = np.divide(evidence, totals, out=np.zeros_like(evidence), where=totals > 0)
    return result.astype(np.float32)

