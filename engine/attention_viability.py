"""Detect contradictory attention filters before spending time on placement.

This is a narrowly scoped configuration guard, not a musical quality score or
an instruction to fill quiet audio. User thresholds are never changed here.
"""
from __future__ import annotations

from .arcaudio.settings import canonical_mode, normalize_settings, DEFAULTS


def assess_attention_viability(analysis, config, candidates):
    mode = canonical_mode(config.get('arcAudioMode', 'legacy'))
    if mode in ('off', 'legacy') or config.get('manual'):
        return {'status': 'not_applicable', 'policy': 'Legacy and manual mapping are unchanged.'}
    knobs = normalize_settings(config.get('arcAudioSettings'))
    report = analysis.get('_arcAudioMapperReport', {})
    attention = analysis.get('_arcAudioAttention', {})
    summary = report.get('summary') or attention.get('summary', {})
    detected = int(report.get('detectedEvents', summary.get('candidateEvents', 0)))
    eligible = int(report.get('eligibleForegroundEvents', summary.get('selectedEvents', 0)))
    low = int(summary.get('rejectionReasons', {}).get('low_importance', 0))
    duration = max(.001, float(analysis.get('duration', 0)))
    requested = max(0., float(config.get('nps', 10)))
    hits = sum(2 if p.get('chord') else 1 for p in candidates)
    states = summary.get('states', {})
    active_fraction = sum(v for k, v in states.items() if k != 'silence') / max(1, sum(states.values()))
    warnings = []
    if knobs['minImportance'] >= .5:
        warnings.append('Minimum event importance is a strict rejection threshold, not a musicality percentage.')
    if knobs['maxDensity'] < 1:
        warnings.append(f"Density ceiling {knobs['maxDensity']:.0%} limits a {requested:g} NPS request to at most {requested * knobs['maxDensity']:g} NPS before hand and evidence limits.")
    # Only block a severe, demonstrable threshold conflict in a substantial,
    # mostly active full-song high-density request. Sparse arrangements, short
    # excerpts, continuation and deliberately quiet regions are not diagnosed
    # from low note count alone. These are explicit product guard thresholds.
    blocked = (
        not config.get('mappingRegions') and not config.get('sourceFocusRegions') and not config.get('continuationContext')
        and duration >= 30 and requested >= 5 and knobs['minImportance'] >= .5
        and detected >= max(100, duration * 2) and active_fraction >= .5
        and low / max(1, detected) >= .5 and eligible / max(1, detected) < .02
        and hits / duration < min(.5, requested * .08)
    )
    result = {'status': 'blocked' if blocked else 'passed', 'requestedNps': requested,
        'minimumImportance': knobs['minImportance'], 'densityCeiling': knobs['maxDensity'],
        'activeNpsCeiling': round(requested * knobs['maxDensity'], 3),
        'detectedEvents': detected, 'eligibleForegroundEvents': eligible,
        'lowImportanceRejected': low, 'candidateHits': hits,
        'candidateAverageNps': round(hits / duration, 3),
        'warnings': warnings, 'settingsChanged': False,
        'policy': 'Severe full-song high-density/strict-threshold conflict; no threshold overrides or invented timings.'}
    if blocked:
        result['message'] = (
            f"Audio attention settings would produce only {hits} notes across {duration:.1f} seconds "
            f"for a {requested:g} NPS request. Of {detected:,} detected attacks, only {eligible:,} "
            f"are eligible; {low:,} were rejected for low importance. Minimum event importance "
            f"is {knobs['minImportance']:.2f}. Use Reset attention defaults "
            f"(minimum importance {DEFAULTS['minImportance']:.2f}, density ceiling 100%) or lower "
            "the rejection threshold and inspect the analysis before retrying. "
            "Your settings and existing levels were not changed. NJS controls approach speed, not note density."
        )
    return result


def require_viable_attention(analysis, config, candidates):
    result = assess_attention_viability(analysis, config, candidates)
    if result['status'] == 'blocked':
        raise ValueError(result['message'])
    return result
