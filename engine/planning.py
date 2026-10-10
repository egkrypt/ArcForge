"""Inspectable musical intent and a measured, section-level generation critic.

Plans bind existing analysis and user intent to actual candidate selection.
Diagnostics are measurements and limitations, never a human-likeness score.
"""
from __future__ import annotations
import collections, itertools, math
import numpy as np
from .arcaudio.settings import canonical_mode
from .regions import config_at_time


def build_map_plan(analysis, config, candidates, minimum, clock):
    mode = canonical_mode(config.get('arcAudioMode', 'legacy'))
    attention = analysis.get('_arcAudioMapperReport', {})
    source = attention if mode not in ('off','legacy') else analysis.get('_arcForgeCandidateReport', {})
    curves = source.get('densityCurve', [])
    music = analysis.get('musicAnalysis', {})
    if not curves:
        curves = [{'start':s['start'],'end':s['end'],'label':s.get('label','Drive'),
            'requestedNps':config.get('nps',10),'targetNps':config.get('nps',10)*s.get('densityFactor',1)}
            for s in music.get('sections', [])]
    rows=[]; tempo=analysis.get('tempoTimeline', [])
    for index, curve in enumerate(curves):
        start,end=float(curve['start']),float(curve['end'])
        if end <= start: continue
        center=(start+end)/2; intent=config_at_time(config, center)
        section=next((s for s in music.get('sections',[]) if s['start']<=center<s['end']),{})
        hits=[p for p in candidates if start<=p['time']<end]
        before_composition=int(curve.get('selectedEvents',curve.get('chosenEvents',len(hits))))
        composition_pruned=max(0,before_composition-len(hits)) if config.get('learnWholeLevel') else 0
        selected_hits=sum(2 if p.get('chord') else 1 for p in hits)
        before_composition_hits=int(curve.get('selectedHits',curve.get('chosenHits',before_composition)))
        composition_pruned_hits=max(0,before_composition_hits-selected_hits) if config.get('learnWholeLevel') else 0
        source_counts=dict(collections.Counter(p.get('source',p.get('role','unknown')) for p in hits))
        goal=intent['styleGoal']; target=float(curve.get('targetNps',0))
        vocabulary=[]
        if target>=8: vocabulary.append('compact alternating streams / bursts')
        if any(p.get('chord') for p in hits): vocabulary.append('spaced accent doubles')
        if goal[1]>=.7: vocabulary.append('parity-preserving angular changes')
        if goal[2]>=.7: vocabulary.append('recovery-limited expressive travel')
        if target<3: vocabulary.append('sparse phrase accents / intentional rests')
        if not vocabulary: vocabulary.append('alternating phrase flow')
        measured_tempo=next((t for t in tempo if t['start']<=center<t['end']),{})
        local_bpm=float(measured_tempo.get('bpm') or (clock.beat(end)-clock.beat(start))*60/(end-start))
        rows.append({'id':f'plan-{index+1}','start':start,'end':end,
            'musicalRole':curve.get('label',section.get('label','Drive')),
            'energy':float(section.get('energy',0)), 'rhythmicActivity':float(section.get('activity',0)),
            'localBpm':round(local_bpm,3), 'tempoConfidence':measured_tempo.get('confidence'),
            'mappingStyle':intent['style'],'secondaryStyles':intent['secondaryStyles'],'goals':intent['philosophy'],
            'requestedNps':float(curve.get('requestedNps',intent.get('nps',10))), 'targetNps':round(target,3),
            'availableEvents':int(curve.get('availableEvents',len(hits))),
            'selectedEvents':len(hits),'selectedHits':selected_hits,'sourceCounts':source_counts,
            'compositionPrunedEvents':composition_pruned,
            'compositionPrunedHits':composition_pruned_hits,
            'selectionLimit':curve.get('limitedBy'), 'backfilledEvents':int(curve.get('backfilledEvents',0)),
            'vocabularyObjectives':vocabulary,
            'lightingIntent':'impact / phrase accents' if section.get('label')=='Drop' else 'build progression' if section.get('label')=='Build' else 'foreground rhythm with space'})
    return {'version':'1.0','audioEngine':mode,'mappingStyle':config.get('style','Modern'),
        'requestedNps':float(config.get('nps',10)), 'dynamicDensity':bool(config.get('dynamicDensity',True)),
        'minimumHandIntervalMs':round(minimum*1000,3), 'sections':rows,
        'timingPolicy':'Only analyzed musical opportunities reach placement; selection and motion goals do not create random note times.',
        'jumpStrategy':{'mode':'manual' if config.get('njs') else 'measured recommendation',
            'scope':'per-difficulty native metadata; per-section recommendations do not claim unsupported native NJS animation.'}}


def critique_generation(chart, analysis, config, plan, candidates, clock):
    from .variety import analyze_variety
    notes=sorted(chart.get('colorNotes',[]),key=lambda n:(n['b'],n['c']))
    times=np.array([clock.time(n['b']) for n in notes])
    colors=[int(n['c']) for n in notes]; counts=collections.Counter(colors)
    runs=[(h,len(list(g))) for h,g in itertools.groupby(colors)]
    findings=[]; rows=[]
    for section in plan['sections']:
        start,end=section['start'],section['end']; duration=end-start
        local=[n for n,t in zip(notes,times) if n.get('customData',{}).get('arcforgePlanSection')==section['id']]
        target=section['targetNps']; actual=len(local)/duration
        selected=sum(2 if p.get('chord') else 1 for p in candidates if start<=p['time']<end)
        desired=int(round(target*duration)); missing=max(0,desired-len(local))
        cause=('wholeLevelComposition' if section.get('compositionPrunedHits',section.get('compositionPrunedEvents')) else section.get('selectionLimit')
            or ('placementRecovery' if len(local)<selected else 'densitySelection')) if missing else None
        row={'id':section['id'],'start':start,'end':end,'role':section['musicalRole'],
            'targetNps':target,'actualNps':round(actual,3),'noteCount':len(local),'plannedHits':selected,
            'densityDeficit':missing,'limitedBy':cause,'colorCounts':dict(collections.Counter(n['c'] for n in local))}
        rows.append(row)
        # A short rounding difference is not an important underpowered section.
        if target>=5 and missing>=2 and actual<target*.75:
            findings.append({'code':'densityDeficit','severity':'warning','start':start,'end':end,
                'description':f"{section['musicalRole']}: planned {target:.2f} NPS, generated {actual:.2f} NPS.",
                'evidence':{'requestedNps':section['requestedNps'],'targetNps':target,'actualNps':round(actual,3),
                    'availableEvents':section['availableEvents'],'missingHits':missing,'cause':cause}})
    imbalance=abs(counts[0]-counts[1])/max(1,len(notes))
    if config.get('mode')!='OneSaber' and len(notes)>=24 and imbalance>.2:
        findings.append({'code':'colorImbalance','severity':'warning','start':0.,'end':float(analysis['duration']),
            'description':'One hand receives substantially more hits; inspect intentional phrase ownership.',
            'evidence':{'redNotes':counts[0],'blueNotes':counts[1],'absoluteShareDifference':round(imbalance,4)}})
    variety=analyze_variety(chart,analysis,config)
    if variety['unmatchedConsecutiveRepeatedBars']:
        findings.append({'code':'unmatchedPatternRepetition','severity':'warning',
            'description':'Adjacent identical relative-motion phrases have no detected matching musical motif.',
            'evidence':{'count':variety['unmatchedConsecutiveRepeatedBars'],'bars':variety['repeatedBars'][:16]}})
    active=[r for r in rows if r['targetNps']>=5]
    recovery_drops=max(0,sum(2 if p.get('chord') else 1 for p in candidates)-len(notes))
    return {'version':'1.0','findings':findings,'sections':rows,
        'density':{'activeSections':len(active),'sectionsBelow75PercentTarget':sum(r['actualNps']<r['targetNps']*.75 for r in active),
            'placementRejectedHits':recovery_drops,'schedulerBackfilledEvents':sum(r['backfilledEvents'] for r in plan['sections']),
            'targetNpsRange':[min((r['targetNps'] for r in rows),default=0),max((r['targetNps'] for r in rows),default=0)],
            'actualNpsRange':[min((r['actualNps'] for r in rows),default=0),max((r['actualNps'] for r in rows),default=0)]},
        'hands':{'redNotes':counts[0],'blueNotes':counts[1],'absoluteShareDifference':round(imbalance,4),
            'longestColorRun':max((n for _,n in runs),default=0)},
        'variety':variety,
        'refinement':{'performed':['two-hand recovery-aware opportunity selection','reachable-density backfill','bounded melodic hand ownership'],
            'preservesMusicalTimes':True,'inventedHits':0},
        'interpretation':'Technical counts and evidence, not a comfort guarantee, perceptual musicality score, or ranking certification.'}
