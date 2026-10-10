"""Small, normalized controls for musical attention, independent of map difficulty."""
from __future__ import annotations
import math

VERSION = '2.0'
SOURCES = ('kick', 'snare', 'percussion', 'bass', 'drums', 'piano', 'chords',
           'melody', 'lead', 'vocals', 'spoken', 'background', 'other')
COMPETITORS = ('drums', 'bass', 'piano', 'chords', 'melody', 'lead', 'vocals',
               'spoken', 'background', 'other')
SOURCE_GROUP = {'kick': 'drums', 'snare': 'drums', 'percussion': 'drums'}
DEFAULTS = {
    'vocalPriority': .65, 'drumPriority': .60, 'attentionStrength': .80,
    'smoothingSeconds': .65, 'competition': .75, 'maxDensity': 1.,
    'minImportance': .24, 'secondaryAllowance': .35, 'phraseAwareness': .80,
    'backgroundSuppression': .80, 'repetitionSuppression': .50,
}


def normalize_settings(settings=None):
    """Ignore unknown fields and clamp non-finite input; all goals use [0, 1]."""
    result = DEFAULTS.copy()
    for key, fallback in DEFAULTS.items():
        try: value = float((settings or {}).get(key, fallback))
        except (TypeError, ValueError): value = fallback
        if not math.isfinite(value): value = fallback
        result[key] = round(min(3.0 if key == 'smoothingSeconds' else 1.,
                                max(.10 if key == 'smoothingSeconds' else 0., value)), 5)
    return result


def canonical_mode(mode):
    value = str(mode or 'off').lower().replace(' ', '').replace('_', '').replace('-', '')
    return {'off': 'off', 'legacy': 'legacy', 'arcaudio': 'arcaudio', 'arcaudiov2.0': 'arcaudio',
            'v2': 'arcaudio', 'hybrid': 'hybrid', 'arcaudio3':'arcaudio3', 'arcaudiov3.0':'arcaudio3', 'v3':'arcaudio3'}.get(value, 'off')


def attention_source(source):
    return SOURCE_GROUP.get(source, source if source in COMPETITORS else 'other')
