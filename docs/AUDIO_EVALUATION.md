# Audio and engine evaluation

ArcAudio detects evidence, allocates musical attention, and supplies timing
opportunities to the mapper. Source names from mixed audio are hypotheses, not
verified instrument stems. YAMNet class posteriors and DSP component evidence
are kept separately so that an uncertain classifier does not erase measured
harmonic or rhythmic activity.

## Reproducible checks

Run these commands from the application folder with the configured Python:

```console
python -X utf8 -m unittest tests.test_attention_dsp_quality tests.test_tempo_timeline tests.test_arcaudio tests.test_recognition_import
python -X utf8 tests/evaluate_audio_attention_quality.py AUDIO_ID --settings defaults --phase current-defaults
python -X utf8 tests/evaluate_audio_attention_quality.py AUDIO_ID --settings pdf --phase current-strict
python -X utf8 tests/evaluate_engine_modes.py --audio-id AUDIO_ID --phase current
```

The evaluation scripts read local audio records and the current placement
checkpoint. They write private artifacts under `data/test-artifacts`, outside
the level library. The engine comparison exports genuine playable-format ZIPs,
records the model checksum, uses identical configuration and seed across modes,
and verifies that saved project files remain unchanged. A strict-settings run
must be compared with another strict run, rather than with defaults.

Distance to an observed onset is a timing-evidence measurement. It is not a
musicality score, instrument-recognition accuracy, ranking assessment, or a
substitute for listening and headset playtesting. Fewer notes are not inherently
better, and higher density is not inherently better.

## Controlled DSP and context coverage

The six source tests construct independent harmonic note attacks, broadband
noise attacks, overlapping harmonic/percussive audio, and absolute silence. They
verify that pitched attacks are not blindly treated as three separate drum
streams, that broadband attacks retain rhythmic evidence, and that overlapping
components exist before attention selection. They also verify that Hybrid adds
only observed, spectrally supported legacy attacks and never mutates the input.
Rapid transient tests also shift the attack phase relative to the context
sample clock. Source strength uses pooled energy rather than a maximum-presence
value: the latter was useful as a presence diagnostic but incorrectly allowed
brief drum attacks to dominate nearly every frame of real mixed audio.

The context fixtures separately test quiet voice with piano, voice with full
drums, instrumental drops, dense choruses, spoken sections, piano-only and
drum-only passages, competing instruments, sparse music, transitions, repeated
phrases, uncertainty, and stable source selection. Those fixtures use known
evidence to test decisions; they do not establish real-world source detection
accuracy.

## Tempo interpretation

The local detector uses onset periodicity and retains a supported existing
meter. It distinguishes the raw rhythmic-rate hypothesis from a change to the
beat clock. An exact doubling of attacks can mean eighth-note percussion at
the existing tempo or a genuine doubled tempo: audio periodicity alone cannot
prove which interpretation the composer intended. Such aliases do not rewrite
the clock by themselves.

Controlled tests include sustained 120→180→90 BPM changes, deceleration,
gradual acceleration, brief fills, silence, malformed evidence, and stable
120/190/235 BPM with changing percussion subdivisions. The gradual-ramp check
requires the generated clock to remain aligned to measured attacks. Stored
real-audio diagnostics must also be inspected for implausible octave jumps;
passing synthetic clicks alone is insufficient.

## Observed limits

The strict screenshot configuration used minimum importance 0.75, substantially
above the default 0.24. This deliberately admits far fewer events. The analyzer
must report this rejection cause rather than fill the map with unsupported
timings to satisfy an NPS request.

The screenshot configuration was reproduced separately from defaults. It
demonstrated sensitivity to minimum importance, but its historical event counts
are not a release acceptance target. Source pooling and event-local spectral
evidence subsequently changed; compare runs using the stored source hashes.
Vocal identification remains weak on this recording, and inferred piano,
speech, and lead labels must retain their uncertainty and provenance.

The first complete engine comparison also exposed an output defect: older
ArcAudio 2 and Hybrid candidate selection applied uncertainty reduction after
the attention engine had already limited the evidence. This produced
underpowered high-BPM output. The generation critic reports actual-versus-target
density per phrase so that correction can be measured on native output, rather
than inferred from the presence of a setting.

The mapper now applies one difficulty-aware density budget to valid foreground
events. It may reconsider an event excluded only by the analyzer's generic
attention quota. It cannot rescue background, sustained, low-importance, or
unsupported events. Hybrid's legacy grid assistance must be tied to compatible
foreground evidence within 35 ms. Whole-level composition retention is measured
separately so that repeated density reductions can be identified. These guards
are intentionally retained even when they reveal that the analyzer supplies too
few meaningful events to reach the requested active NPS.

## Real-output checkpoint: 9 October 2026

The `oct9-source-strength` evaluation generated fifteen native map archives:
Neon Phoenix (190 BPM), The One That Got Away (134 BPM), and When the Sky Goes
Silent (182.5 BPM), each using OFF, Legacy, ArcAudio 2, Hybrid, and ArcAudio 3.
Every run used the same existing placement checkpoint, seed 32026, Modern
style, Expert+ target, and requested active NPS of 12. Whole-level composition
and learned lighting remained enabled. The evaluation records source and model
hashes before and after, as well as saved-project hashes. The one project
present during this run and all model/source fingerprints remained unchanged.

All fifteen archives passed the current motion checks without errors or
warnings. Thirty independent schema checks passed across their Info and V3
difficulty files. OFF and Legacy were byte-identical by design; the three
opt-in modes produced distinct timing hashes. No source mode introduced tempo
changes on these stable-tempo recordings.

Corrected source evidence no longer made drums primary almost everywhere:
drum-primary frame shares ranged from 0.4% to 6.2% across these nine opt-in
analyses. The component layers still overlap. Bass and melodic evidence became
prominent, but these measurements do not prove the resulting named instrument
labels are correct. In particular, real vocal/piano recognition is not solved.

The generated maps exposed a remaining quality defect. The following values
are actual NPS inside the **same analyzer-defined high-activity sections** of
each song, not manually annotated drops:

| Song | Legacy | ArcAudio 2 | Hybrid | ArcAudio 3 |
| --- | ---: | ---: | ---: | ---: |
| Neon Phoenix | 10.73 | 7.48 | 7.46 | 5.62 |
| The One That Got Away | 10.46 | 3.68 | 3.69 | 6.02 |
| When the Sky Goes Silent | 10.57 | 6.74 | 6.74 | 5.70 |

Across active planned phrases, actual-to-target hit ratios were 96–98% for
Legacy, 51–78% for ArcAudio 2/Hybrid, and 66–75% for ArcAudio 3. These results
are **not a quality acceptance pass**. They show that foreground eligibility
and learned composition still remove too many opportunities on these songs.
For Neon Phoenix in ArcAudio 2, 57 under-target phrases were evidence-limited
and 25 were limited by whole-level composition; none were lost to the final
placement recovery guard. Increasing training epochs would not repair that
specific coordination defect.

Known sparse sections remained sparse: all engines placed zero notes in the
5.08-second low-activity region of The One That Got Away. On When the Sky Goes
Silent, the shared 21.79 seconds of low-activity sections averaged 1.15–1.52
NPS. That guard must remain in place while addressing the dense-section
starvation; copying every legacy event into the attention pipeline would not
demonstrate better musical selection.

Private reproducibility artifacts are under
`data/test-artifacts/engine-comparison/oct9-source-strength/`: `results.json`,
`preservation.json`, native ZIPs, generation reports, and independent schema
results. They are excluded from the source distribution because they contain
user audio. Analyzer-defined section roles and source confidence are
heuristics; they are not ground-truth genre labels or calibrated accuracy.

## Secondary-event correction: 10 October 2026

The next measured change fixes a coordination error rather than adding
training data. A sustained source could previously occupy the secondary slot,
while uncertain instrument identity prevented observed competing accents from
being admitted. Secondary selection now requires actual source attacks. In
busy mixed contexts, an independent competing layer may contribute observed
accents even when instrument identity is uncertain; event strength, confidence,
relative attention, minimum importance, coincidence suppression, and the
downstream single density budget still apply. This does not admit a third
source or enable background streams in quiet vocal space.

Nine opt-in maps were generated first and compared with the frozen 9 October
baseline. Six OFF/Legacy maps then completed the fifteen-map native-export
comparison. All source, model, and saved-project fingerprints matched across
both stages. Thirty independent Info/V3 schema checks and all fifteen current
motion checks passed. No checkpoint or saved level was modified.

| Song | ArcAudio 2 high-activity NPS | Hybrid high-activity NPS | ArcAudio 3 high-activity NPS |
| --- | ---: | ---: | ---: |
| Neon Phoenix | 7.48 → 8.18 | 7.46 → 8.19 | 5.62 → 7.93 |
| The One That Got Away | 3.68 → 6.02 | 3.69 → 6.06 | 6.02 → 7.01 |
| When the Sky Goes Silent | 6.74 → 7.33 | 6.74 → 7.32 | 5.70 → 7.69 |

These are the same measured sections as the prior table. ArcAudio 3's active
actual-to-target hit ratios increased from 66–75% to 79–84%. ArcAudio 2/Hybrid
increased from 51–78% to 74–81%. Neon Phoenix reached 10.75–11 NPS over its
densest four-second windows; The One That Got Away reached 8.25–9.25 NPS and
When the Sky Goes Silent reached 10 NPS. OFF/Legacy output remained unchanged.
Maximum same-color runs remained two notes in these Standard-mode fixtures.

The zero-note 5.08-second sparse region remained empty. The screenshot song's
21.79 seconds of shared sparse sections remained at 1.42–1.56 NPS. Repeating
the screenshot's strict 0.75 importance configuration selected only 91 events
in ArcAudio 2, 91 in Hybrid, and 25 in ArcAudio 3; secondary admission did not
bypass that control.

This is a measured improvement, not a claim that the remaining quality work
is finished. Evidence eligibility and learned whole-level composition still
leave some phrases below their targets, and source labels remain uncertain.
For example, the revised Neon Phoenix ArcAudio 3 critic identifies 29
evidence-limited and 34 composition-limited under-target phrases. These
diagnostics remain visible rather than silently reducing the reported target
or inventing unsupported note timings. This intermediate pass's private artifacts are
under `data/test-artifacts/engine-comparison/oct10-secondary-final/`, including
an `assembly.json` that documents the two fingerprint-matched stages.

## Observed accents, recurring motifs and one active note budget

The following pass traced three more information losses at the analyzer/mapper
boundary. Foreground candidates always disabled accent doubles, even when real
admitted attacks and the learned human double distribution supported them.
Their unique phrase IDs also prevented the existing placement model from
recognizing recurring musical motif IDs. Finally, whole-level composition
applied another density/double reduction after attention and physical recovery
had already allocated a note budget.

Candidates now retain observed event ID/time, strength, salience, source
confidence and admission evidence. Coincident background detections are still
diagnostic only: only independently admitted observed sources can support an
accent. Learned phase/energy double frequency, style, difficulty and movement
goals determine a bounded accent allowance. Each double consumes two physical
notes, needs recovery for both hands on both sides, and cannot displace an
already selected single hit or add a new timestamp. One Saber and quiet vocal
space keep single hits. Recurring motif identity now reaches recall and
variation conditioning, with the unique phrase ID retained separately.

An intermediate three-song ArcAudio 3 run emitted 12, 21 and 14 accent doubles
for Neon Phoenix, the vocal recording and the screenshot song respectively,
with zero current motion errors/warnings and no placement rejections. The
5.08-second quiet vocal region stayed empty. Its artifacts are retained in
`data/test-artifacts/engine-comparison/oct10-accents-v3/` so the subsequent
composition change can be assessed separately.

Active foreground groups with planned NPS of at least five now retain their
already allocated hits and learned accents through composition. The original
target is not lowered to conceal a deficit. Learned motion, complexity, motif
variation and lighting still condition the output; quiet/rest envelopes remain
available. Diagnostics count timing events and physical notes separately.

Real output inspection also found a reporting bug: rounding a phrase start
from 7.78599 to 7.786 seconds excluded its actual first attack and blamed
composition for removing it. Internal window boundaries now retain precision;
display formatting is separate. The regression checks first-hit ownership and
physical-note counts. Fifteen isolated accent contracts cover these behaviors,
including actual neural placement and native V4 provenance roundtrip.

## Final frozen-source comparison: 10 October 2026

`oct10-precise-plan-final` generated all fifteen maps again after the exact
boundary correction, with one fixed source/model fingerprint set. All thirty
external Info/V3 schema checks passed. Every map retained zero current motion
errors and warnings, and the maximum same-color run was two. Saved projects,
the existing placement checkpoint and composition corpus stayed unchanged.
OFF/Legacy charts remained byte-identical to the 9 October baseline.

Actual high-activity NPS in the same measured music sections:

| Song | ArcAudio 2: baseline → final | Hybrid: baseline → final | ArcAudio 3: baseline → final |
| --- | ---: | ---: | ---: |
| Neon Phoenix | 7.48 → 8.29 | 7.46 → 8.30 | 5.62 → 8.01 |
| The One That Got Away | 3.68 → 6.16 | 3.69 → 6.18 | 6.02 → 7.11 |
| When the Sky Goes Silent | 6.74 → 7.54 | 6.74 → 7.51 | 5.70 → 7.80 |

ArcAudio 3 retained 12/21/14 observed accent doubles on these songs, with all
double notes accounted for in the physical quota. The other opt-in modes used
19–35 doubles according to their admitted events and learned accent evidence.
Their densest four-second windows reached 10.75–11.25 NPS on Neon Phoenix,
8.25–9.5 NPS on the vocal recording, and 10–10.25 NPS on the screenshot song.
The 5.08-second sparse vocal section still has zero notes. The screenshot song's
21.79 seconds of sparse sections average 1.52–1.74 NPS; they were not filled to
the requested 12-NPS active target.

Active actual-to-target ratios are 81–85% for ArcAudio 3 and 77–83% for
ArcAudio 2/Hybrid. The corrected critic now attributes the remaining
underpowered active phrases to insufficient eligible musical evidence, rather
than falsely claiming composition removed first attacks. These are still
meaningful deficits. They are not concealed by changing the target or solved
by inventing hit times. Instrument labels and perceptual foreground choice
remain uncertain, and these engineering checks do not establish superior
musicality or headset comfort.

The final private artifacts contain native ZIPs, generation reports, schema
results, `preservation.json`, source/model hashes and `final-delta.json` under
`data/test-artifacts/engine-comparison/oct10-precise-plan-final/`. Earlier runs
remain available as failed/intermediate comparisons. None of their songs or
private model data is redistributed in the source release.

## Final HTTP generation and native V4 export

The final source also ran through the actual generation/export APIs in an
isolated workspace, using the screenshot song, the installed local placement
checkpoint and ArcAudio 3 with YAMNet. The requested and effective analysis and
placement engines matched; no fallback occurred. Echoing the displayed BPM
preserved the existing tempo analysis, the saved user project stayed unchanged,
and the exported Ogg hash matched the input Ogg.

| Independently generated difficulty | Notes | Whole-song average NPS | Densest four-second NPS | Native NJS events |
| --- | ---: | ---: | ---: | ---: |
| Easy | 639 | 2.12 | 3.00 | 23 |
| Hard | 1,320 | 4.38 | 6.00 | 47 |
| Expert+ | 1,928 | 6.40 | 10.25 | 59 |

All eight external schemas passed: Info 4.0.1, AudioData 4.0.0, and the three
indexed gameplay/lightshow pairs. Each difficulty had zero current motion
errors and warnings. The native files retained arcs, chains, walls, Chroma
colors, Noodle note animations and 92 `AnimateTrack` events per chart, plus
635 basic light-rotation events per chart. These are file-level evidence of
connected generation/export, not proof that every effect renders identically
in the browser or game. Optional native NJS was enabled for this test; its
bounded recommendations are not a comfort guarantee.

The Expert+ critic still reports 71 of 214 active sections below 75% of their
planned density, with zero placement-rejected hits. Its maximum same-color run
is two, and no consecutive bars have identical relative motion. Those measures
do not establish human-like authorship. In particular, the density deficits
remain an upstream eligible-event limitation and require further attention
evaluation; successful export and geometric validation do not resolve them.

This run took 90.13 seconds while the fifteen-map comparison was also running,
so it is not a standalone performance benchmark. Reproducible private reports,
native files and independent schema results are under
`data/test-artifacts/master-end-to-end/1791635539516736800/`. None are bundled
with the public source archive. Perceptual synchronization, instrument labels,
foreground selection and headset playability still need human review.

## Reported 26-note chart: reproduced and repaired

The saved Expert+ request correctly contained 13 NPS. Its minimum event
importance was 0.75 and density ceiling 0.70, rather than the defaults 0.24 and
1.00. The original report detected 8,407 attacks but admitted only 25; 6,511
were rejected for low importance and 1,845 as supporting layers. Those 25
attacks became 26 notes. The automatic 14 NJS recommendation was a consequence
of that sparse chart, not a conversion of NPS into NJS. NPS and NJS remain
separate throughout the request, generation and jump recommendation.

The UI now explains the threshold, exposes an explicit reset, reports
requested/actual NPS separately from approach speed, and warns that a 70%
density ceiling caps a 13 NPS active target at 9.1 NPS. No preferences are
silently rewritten. A narrowly scoped generation guard rejects a near-empty
full-song high-density request when measured rejection evidence demonstrates a
strict-threshold conflict. Low note count alone does not trigger it; short
excerpts, sparse arrangements, guided regions, continuation, manual mapping and
Legacy are preserved. Six tests include the actual generator stopping before
Inferno/placement, NJS independence and unchanged input settings.

After the source-family fusion correction described below, the same strict
saved request was tested through the actual API in an isolated workspace. It
would still produce only 27 notes from 26 eligible attacks, with 6,506
low-importance rejections. The new guard returned an actionable error instead
of saving that result as a successful generated chart.

A second isolated API run changed only minimum importance to 0.24 and density
ceiling to 1.00, retaining the saved Tech style, Expert+ 13 NPS request,
Ranked-compatible target, whole-level learning and Hybrid placement. It produced
1,934 notes, averaging 6.42 NPS with a 10.25 NPS four-second peak. Requested
ArcAudio 3 and Inferno + ArcForge matched the effective engines with no fallback.
The chart had 78 arcs, balanced hands (976 red/958 blue), no consecutive
identical relative-motion bars and zero current motion errors/warnings. All
four external Info/AudioData/gameplay/lightshow schemas passed. The original
saved chart stayed unchanged. NJS was 20.7 m/s and JD 27.84 m; the ranking
target correctly kept native NJS events disabled and reported that reason.

This fixes the catastrophic filtering/runtime failure, not the entire density
problem: 97 of 217 active sections remained below 75% of planned density, with
zero placement-rejected hits. The 13 NPS active goal was not fully achieved.
The original target remains visible. Source identity and musical foreground
still require listening and headset review. The run took 46.542 seconds and
is retained at `data/test-artifacts/master-end-to-end/1791637662846223400/`;
configuration snapshots and the mocked browser regression are under
`data/test-artifacts/attention-starvation-fix/`.

Inferno's Python 3.12 worker now initializes its app-local dependency site
directories, including the installed setuptools `.pth` compatibility hook and
Pillow dependencies. No stdlib was copied and no global Python installation was
changed. Small genuine sigmoid probabilities now normalize in float64 instead
of being attenuated by a fixed 1e-8 denominator; empty rows remain empty. A real
80-event inference returned finite, normalized [80, 2, 108] distributions from
the unchanged upstream weights. Seven runtime regressions cover these fixes.

The updated installed service then generated a separate library project named
`When the sky goes silent — recovered` (ID
`dcc02d9fe4de4b4280dda0567e752f78`) through its normal asynchronous API. It
reproduced 1,934 notes, 6.42 average NPS, 10.25 four-second peak NPS and effective
Inferno + ArcForge with no fallback. Its native export is `map-r1-v4.zip`.
Hashes of the original 26-note project, saved preferences and original audio
analysis all remained unchanged. The actual recovered project and private
`live-recovered-result.json` are not bundled with the source release.

## October 10: 26-note report and instrument uncertainty

The reported 301.03-second, 182.5 BPM song is the imported **When the sky goes
silent** recording. Its first twelve seconds reproduce weak named-piano
classification: the cached YAMNet Piano posterior averages approximately 0.01,
while Harp averages approximately 0.20. These are actual model scores, not
verified instrument labels. The functional melody evidence is substantially
stronger, so a nearly flat Piano lane does not mean that pitched attacks were
absent. Neither this analysis nor the harmonic/percussive DSP produces isolated
instrument stems or proves that the opening is piano. An experimental harmonic
audio view produced inconsistent wind/saxophone labels and was rejected.

The 26-note failure has a separate, reproducible cause. The saved attention
configuration used `minImportance: 0.75` and `maxDensity: 0.7`. The importance
threshold rejected 6,511 low-importance events out of 8,407 candidates, leaving
only 25 selected events after the other attention filters. That upstream
starvation produced the tiny chart; NPS and NJS are separate quantities. The
failed Inferno process then fell back to local placement, as recorded in the
generation report. Fixing its runtime cannot recover events already rejected
by audio attention.

A second defect was corrected in source fusion: one confident named class,
such as vocals, previously suppressed measured evidence from unrelated musical
families. Fusion now considers rhythm, pitched instrumentation, bass and voice
independently, while retaining finite attention competition downstream. Six
new controlled regression tests cover overlapping sources, silence, truthful
piano uncertainty and guitar-class evidence without asserting a lead role.
The same real recording with default attention settings selected **1,596 events
before and 1,612 after** this change; raw Piano scores were unchanged. New
`instrumentEvidence` fields expose direct timbre-class scores separately from
mapping roles. This is not a claim of improved labeled instrument accuracy.

These results validate source-analysis behavior only. Recovered generated maps
and their export/motion checks are not yet claimed by this subsection. Private
before/after reports and the rejected experiment are saved under
`data/test-artifacts/piano-lead-diagnosis/`.


## Deep analysis coverage and compute — 10 October 2026

A full-song benchmark used the same private 301.03-second recording as the sparse-chart investigation, on 12 logical CPU threads and an RTX 3050 with 8 GB VRAM. The audio, saved analysis and trained recognition weights retained their hashes. No model retraining or restored library entry was involved.

| Measurement | Existing Balanced | Deep |
| --- | ---: | ---: |
| Actual classifier windows | 627 | 2,502 |
| Window stride | 0.48 s | 0.12 s |
| CPU analysis threads | 6 existing policy | 12 |
| CUDA batch | 32 | 256 |
| Classifier wall time, cold process | 2.40 s | 6.80 s |
| Full Deep recognition, cold | not measured by the same wrapper | 9.30 s |
| Full Deep recognition, forced warm repeat | — | 1.79 s |

Deep's cold classifier included 2.88 seconds model loading, 0.17 seconds features and 3.59 seconds network work. The CPU HPSS/source worker ran concurrently for 8.07 seconds; fusion took 0.63 seconds. Peak CUDA tensor allocation was about 284 MB, peak reserved CUDA memory about 520 MB, and peak process RSS about 1.73 GB. There were zero OOM retries. These numbers are useful-work measurements, not a claim that every device resource should stay saturated.

Forced repeat was real inference, not a recognition-cache hit; scores were bit-identical on this machine. Both modes selected 1,612 attacks in the comparison, with 1,560 common selections and 17 of 1,205 primary-source frames changing. Equal counts therefore do not imply identical attention. There were 70 primary-source switches in both runs. No reference source labels exist here, so this is **coverage, performance and reproducibility evidence**, not proof of better instrument accuracy.

The opening's uncertain synthetic piano-like timbre remains classified mostly as melodic/plucked material: piano posterior did not improve (mean about 0.0085 Balanced versus 0.0059 Deep in 0–15 seconds). More compute alone does not fix this classifier limitation. Regional Custom Sources also cannot manufacture missing piano detections.

Private reproducible artifacts: `data/test-artifacts/analysis-compute/e0757df0cb77450282d3dec730d292d2/{before-balanced,after-deep}.json`. They and original audio/model files are excluded from the source archive.


## Regional attention and NPS — verified real generation

Two full maps used the same 301.03-second song, trained placement model, seed, compute profile and regional density/style settings. The control kept regional attention Auto; the directed version used Full Mix or strict Custom Sources. This isolates source selection from a merely changed NPS setting. The existing local model generated actual notes and native V4 exports.

| Range and directive | Auto control | Directed result |
| --- | --- | --- |
| 0:30–0:46, Speed, NPS range 8–13 | 115 notes, 7.19 NPS | 164 notes, 10.25 NPS |
| 1:00–1:16, Drums only, target 10 NPS | 118 mixed-source notes | 49 kick/snare/percussion notes, 3.06 NPS |
| 4:14–4:20, Spoken only | 26 mixed-source notes, 4 spoken | 7 notes, all spoken |

The strict drum range deliberately remains below its requested NPS because the selected drum evidence supplies too few eligible attacks. The system reports this limitation instead of inventing timing or inserting another instrument. The sparse spoken region demonstrates the same evidence-first principle. These are technical musical-attention results, not a headset enjoyment rating.

A first run found three automatic selections just outside the edited regions affected by a shared sliding attention quota. The fix restores untouched automatic event admission outside exact half-open ranges without rerunning detection. The final run matched all **1,400 outside-region selected events exactly**, with identical source/timing decisions. Full-chart note bytes can still differ after a changed region because hand recovery and pattern history are validated through the remainder of a complete generation; partial revisions preserve authored content outside their explicit merge range.

The control generated 1,939 notes; the directed version generated 1,901. Both passed static motion and internal native-format checks. All eight external native V4 schema checks passed with zero errors or warnings. Original audio, saved analysis, user projects and trained model hashes remained unchanged. Evaluation did not restore deleted library media.

Private final report: `data/test-artifacts/mapping-source-regions/1791657171500760500/evaluation.json`. Both reproducible evaluation tools are distributed in the source release; private songs, exports, score arrays and model weights are excluded.
