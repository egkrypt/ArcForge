"""Use app-scoped dependencies without modifying the user's Python installation."""
import os, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parent
for directory in [ROOT.parent / '.tools', ROOT / '.deps', ROOT / '.server-deps']:
    if directory.exists():
        sys.path.insert(0, str(directory))
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
