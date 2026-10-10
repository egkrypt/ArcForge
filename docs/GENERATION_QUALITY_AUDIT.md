# Generation quality audit — 10 October 2026

This audit records failures traced through real data, changes made to the
mapping engine, and measured output. It is not a declaration that every
requested feature or subjective mapping goal is complete.

## Lifecycle and information losses

| Stage | Existing implementation | Concrete loss or bottleneck | Change / remaining work |
| --- | --- | --- | --- |
| Audio → evidence | `audio.py`, `engine/music.py`, `engine/arcaudio/`, `engine/recognition/` | The supplied screenshots show overconfident percussion and uncertain `other` classifications, combined with restrictive user thresholds. | Audio investigation and before/after evidence are maintained separately below. No new training is assumed necessary. |
| Evidence → opportunities | `engine/generate.py`, `engine/attention_candidates.py` | Fixed quarter-beat grid limited faster observed rhythms at low BPM. At high BPM, increasing the entire subdivision step discarded every other opportunity. Attention selection treated minimum global spacing as though it were two-hand recovery. | Keep the observed high-BPM grid; add genuinely detected faster attacks when the coarse grid is limiting; schedule existing opportunities against independent hand cooldowns. |
| Tempo → native beats | `generate._mapping_clock` | Averaging 32 beat periods smeared abrupt tempo changes and displaced subdivisions near the boundary. | Compress only adjacent equivalent periods with a bounded timing deviation; segment means preserve the elapsed time at each boundary. Measured beat times remain authoritative. |
| Creative intent → density | Styles, global goals, optional regions | Candidate quotas counted a double accent once while placement emitted two notes. Confidence was multiplied into attention and mapper quotas repeatedly. Whole-map density protection compared a song label with an attention-state label and missed opt-in drops. | Doubles consume two budget units. All attention modes share one difficulty-aware budget over eligible foreground events; source uncertainty is not repeatedly treated as slow rhythm. Already budgeted active attention hits and learned accents survive the composition stage; quiet/supporting envelopes can still reduce. |
| Opportunities → hands | Neural placement with deterministic hand assignment | A missing melodic motif defaulted to red; a repeated numeric motif could monopolize one hand. | Missing motifs use normal alternation. Evidence-backed phrase ownership is bounded by recent balance, recovery and color runs. One Saber remains one-handed. |
| Hands → motion | Learned token probabilities plus geometric masks | Every hand was confined to its own half. A universal distance penalty favored small inline movement regardless of expressive style. | Learned style and movement goals determine a travel objective; rapid density keeps it compact. Expert expressive styles can use an adjacent cross-lane when recovery permits. Fast parity remains strict. |
| Motion → validation | `audit_map`, geometric playtest / repair | Rapid note-to-note reach did not capture all cut-endpoint recovery spikes. | Placement masks constrain rapid cut-endpoint recovery before emitting notes. Validation and authored-map advisory checks remain separate; manual drafts are not automatically changed. |
| Generation → feedback | Section report plus static checks | Displayed target NPS was recomputed from raw section density, ignoring actual style/region quotas. Placement skips were not distinguished from audio scarcity. | A plan is built before placement and consumed by local-tempo/NPS model inputs and travel objectives. The critic compares actual output with this plan and exposes evidence, physical limits, backfill and color counts. |
| Engine control → inference | Local, Inferno, Hybrid / Auto placement | Explicitly selecting Inferno could silently run Local after failure. ArcAudio 3 source provenance was omitted from notes. | Explicit unavailable/failing Inferno now errors. Hybrid/Auto fallback has a reason. `report.engineSelection` identifies requested/effective engines. ArcAudio 3 provenance reaches native note custom data. |
| Attention → accents / motifs | `engine/attention_candidates.py` | Foreground candidates discarded strength/salience/coincidence evidence and always disabled doubles. Unique phrase IDs were passed where placement expected recurring motif identity. Rounded internal phrase boundaries misassigned first attacks and falsely reported composition pruning. | Preserve admitted observed evidence; learned phase/energy accent frequency spends only spare note quota with both-hand recovery. Use source-aware recurring motif IDs, retaining separate phrase provenance. Internal boundaries retain precision through planning. |
| Gameplay → lighting | `modding.py`, learned human-lighting distributions | Separate lighting decisions could fail to share the gameplay attention and section plan. | Root integration passes the generation plan to lighting. Lighting research and real-output evidence are recorded below. |
| Native data → editor / export | Editable canonical data, V3 exporter, native V4 codec | Version conversion and native event behavior require format validation rather than a renamed version field. | Native-format and UI integration results are recorded below. Existing user projects are not rewritten by this audit. |

## Mapping research used

The [BSMG Basic Mapping guide](https://bsmg.wiki/mapping/basic-mapping.html)
connects rhythm choices to foreground instruments, recommends alternating
forehand/backhand motion, and calls for tighter angular relationships at faster
swing speeds. It permits broader positioning in advanced difficulties with
adequate reaction and recovery. Those principles informed the retained rapid
parity guard, capacity-based hit selection and conditional cross-lane movement.

The [Intermediate Mapping guide](https://bsmg.wiki/mapping/intermediate-mapping.html)
distinguishes musical representation from gameplay, describes faster play as
more linear, and warns that jump speed changes timing precision as well as
visual separation. Accordingly, the implementation does not equate density,
movement, NJS or randomness with quality.

The [Downmapping guide](https://bsmg.wiki/mapping/downmapping.html) treats rhythm,
pattern simplification and jump readability as connected difficulty decisions.
The regression spread checks density **and** direction vocabulary across five
difficulties; each difficulty is generated independently.

These are design constraints, not certification of comfort, ranking eligibility,
or superiority to human authors. No authored chart layouts were copied.

## Existing human corpus

The already extracted style corpus contains **1,767 charts**. Read-only measured
whole-chart medians were 4.6239 NPS, 0.3967 nonvertical-note fraction, 1.7238 grid
units of same-hand travel, and 0.1344 simultaneous-double fraction. The 5th–95th
NPS range was 1.8605–8.9138. This is a heterogeneous, unlabeled collection, not a
target every map must match. The existing placement checkpoint was reused;
this work did not retrain or replace it.

## Real before/after generation

`tests/quality_benchmark.py` generates isolated native charts from the existing
audio records for Fresh! (235 BPM), Fallen Winner 2 (190 BPM), and Katy Perry —
The One That Got Away (134 BPM). Each run uses the same trained checkpoint,
seed `32026`, Modern style, Expert+ rank, requested 12 NPS, dynamic density and
the same six creative goals. It does not create or overwrite Library projects.

Private local artifacts are in `data/test-artifacts/master-quality/`:

- `before-summary.json` and `before-*.json`: actual pre-change generated charts.
- `after-summary.json` and `after-*.json`: current generated charts and reports.
- `style-profile-effects.json`: isolated same-audio style/profile comparison.

The summaries include chart measurements, human-corpus reference distributions,
and current source/model fingerprints. Attack-distance measurements are
distance to detected spectral onsets, not a perceptual synchronization score.

| Song | Observed result |
| --- | --- |
| Fresh! | Rapid endpoint-recovery warnings fell from 18 to 0 while retaining the same timing language. Modern hand travel increased from 0.872 grid units to about 1.09. The final map contains 1,449 notes, 7.4285 average NPS. |
| Fallen Winner 2 | No geometric errors before or after. Modern hand travel increased from 0.831 to about 1.05 without adding random timing. The final map contains 2,229 notes, 9.2157 average NPS. |
| Katy Perry | Faster real attacks previously discarded by the coarse grid can now be selected. Median detected-attack distance fell from 11.53 ms to 0.01 ms; 95th percentile fell from 62.90 ms to about 32.29 ms. The final map contains 2,270 notes, 9.9786 average NPS, split equally between hands. |

After double-budget correction, the exact final counts and density curves are
in `after-summary.json`; counting independent musical opportunities as notes
would overstate the requested density. Final checks found no active section
below 75% of its **planned** NPS in these three Legacy-engine runs. This does not
mean the requested global NPS should be maintained through silence or quiet
phrases, nor that another audio engine has the same timing or event choices.

## Controlled regression evidence

`tests/test_mapping_quality.py` covers:

- 300 BPM observed quarter-beat attacks: controlled output exceeds 11.5 NPS
  while respecting the configured hand recovery; the old global subdivision
  reduction could not exceed 10 NPS. No hit times are invented.
- Two close accents can use separate hands; three hits cannot violate their
  combined cooldown capacity. Greedy salience selection is backfilled only from
  existing feasible opportunities when it unnecessarily misses the quota.
- Missing melodic motif evidence no longer forces a sparse vocal map to red.
- A requested 4 NPS over 24 seconds stays within 96 notes while retaining real
  accent doubles.
- An abrupt 120→240 BPM boundary preserves all measured beat timestamps and the
  initial offset in the native clock.
- Easy→Expert+ independently changes density and directional structure.
- Explicit unknown/unavailable engines fail clearly; effective engine and
  ArcAudio 3 source provenance are inspectable.

On the same controlled rhythm after note-budget correction, Speed versus Dance
produced mean travel **1.2723 versus 1.8048** grid units. Tech versus Accuracy
produced nonvertical fractions **0.2421 versus 0.1895**. Dance used **0.1684**
cross-lane fraction while Accuracy used zero. Movement goal 0→100 changed Dance
travel **1.5230→1.9149**. All comparisons passed the geometric error/warning
checks. These are reproducible behavioral differences, not named style labels
learned from verified corpus annotations.

## Verified checks and compatibility

The targeted mapping suites passed: 10 master mapping tests, 12 attention/region
tests, 10 existing musical/playtest tests, 7 whole-map variation tests, 3 native
jump tests, and 15
region/continuation unit tests. The real continuation API workflow also passed:
preview, regenerate, accept, undo, native ZIP, and authored-content preservation.
Windows sandbox socket restrictions can stall `TestClient`; that isolated API
test was rerun successfully outside the sandbox, without touching user maps.

Default, Off and Legacy analysis routes remain exactly equivalent with the same
current checkpoint. Historical pre-master hashes/counts are retained as evidence,
not overwritten to conceal the intentional scheduler changes.

Attention regression coverage also checks that all enabled engines can approach
10 requested NPS from a busy observed foreground even when the instrument label
has only 0.4 confidence. This does not admit background or low-importance events.
Quiet voice retains its seven observed attacks; rejected piano, sustained and
simultaneous duplicate events remain rejected. Hybrid/native-v3 timing assistance
requires an eligible foreground attack within 35 ms with a compatible musical
role. Actual V2/Hybrid analyzer-to-map tests confirm increasing minimum importance
changes emitted note counts; fallback cannot bypass that setting.

## Optional native V4.1 jump events

`engine/jump.py` consumes the actual section NPS, local tempo and angular demand
after base NJS/JD have been chosen. When explicitly enabled for automatic NJS and
native V4, it emits canonical native NJS events, with four-second section smoothing,
changes capped at ±2 m/s, at least 1.5 seconds per eased transition and peak slew
bounded at 1.5 m/s per second. Manual NJS, V3, ranked targets and rotating maps keep
their existing chart data. Existing authored NJS events are always preserved.

The [native NJS schema](https://bsmg.wiki/mapping/map-format/beatmap.html#njs-events)
uses differences from Info.dat's base NJS and eases toward the current event.
The helper follows those semantics: below base speed JD stays constant; above
base speed reaction time stays constant. These are bounded recommendations,
not a human comfort guarantee. Tests verify native indexed V4 encoding,
validation and decoding preserve the events. ArcViewer's V3 preview conversion
does not render this native feature and must state that limitation.

## Remaining limits

- Geometric diagnostics approximate motion, not a person's full arm, wrist,
  tracking hardware or headset sight-reading behavior.
- No invented overall “human-like” percentage, instrument accuracy or perceptual
  musicality score is reported.
- Weak spectral/source evidence can still choose the wrong foreground. The
  supplied PDF problem cases require the audio fixes and real comparisons below.
- The current style corpus has measured archetypes, not verified human style
  labels; musical phrase/motif interpretation remains uncertain.
- Native V3 NJS is static difficulty metadata. Per-section recommendations must
  not be presented as changing game speed unless a supported native/modded
  representation is actually emitted and preserved.

## Audio, tempo, lighting, format and UI integration findings

### PDF reproduction and source evidence

The three supplied PDF screenshots were rendered and inspected before changing
the analyzers. They showed two different defects: spectral-band attacks being
treated as drums regardless of harmonic evidence, and a high generic Music
posterior being converted into an independent `other` stream. Neither category
was fixed by renaming its label.

Observed attacks now carry local harmonic/percussive and band-support evidence.
A short forward spectral window distinguishes pitched attack tails from noise;
inaudible band leakage is excluded. Context strength pools the full 250 ms
window instead of sampling one STFT frame, while attack times retain their
original resolution. Transient presence is a separate diagnostic from source
strength: a failed max-pooling experiment made drums dominate over 92% of two
vocal recordings and was rejected after real-song testing. RMS source strength
removed that takeover. Onset counts describe rhythm density and cannot directly
override measured source strength.

ArcAudio 3 retains actual YAMNet class posteriors separately from functional DSP
components. A weak instrument posterior no longer erases measured bass, melody
or percussion; uncertain timbre remains explicitly uncertain. This is not stem
separation or verified piano/vocal transcription. The [official YAMNet
documentation](https://www.tensorflow.org/hub/tutorials/yamnet) describes sound
classification; numerical agreement with that network is not instrument accuracy
on a commercial mix.

The independent source tests cover pitched attacks, broadband transients,
overlapping components at known attack times, absolute silence, observed-only
Hybrid supplementation and fast attacks shifted relative to the context clock.
Quiet residuals are weighted by energy when evaluating source evidence; tests
must not force continuously strong drum importance between isolated hits.

### Tempo and settings contracts

The tempo detector retains supported meter and rejects octave/subdivision aliases
unless there is independent evidence of a clock change. Tests cover sustained
changes, acceleration, deceleration, changing percussion subdivisions, short
fills and separated bursts. The wizard no longer erases a dynamic beat clock
merely by echoing its displayed BPM and offset. HTTP regressions verify unchanged
values preserve the clock, offset changes shift it, and explicit BPM overrides
create the requested constant grid.

Audio analysis and note placement have separate selectors. Creation,
continuation and section operations receive the current choices; saved reports
record requested/effective engines and any placement fallback. An explicit
unavailable Inferno request fails rather than silently becoming local. Off and
Legacy intentionally retain identical timing behavior. Switching level tabs or
refreshing diagnostics preserves pending analysis settings; playback follows the
level's audio rather than an unrelated inspection track.

### Lighting shares the musical plan

The lightshow consumes the same observed musical events, tempo timeline,
foreground selection and mapping plan as gameplay. It does not run an unrelated
loudness detector. Foreground musical roles select fixture groups; phrases
alternate native red/blue banks; local tempo governs rotations and controlled
flicker timing. User style policies change spacing, fades, motion and impact
eligibility. Tests compare actual event counts and both color banks for Minimal,
Classic, Cinematic, Energetic and High-BPM policies, and verify that rejected
background piano does not become an independent light stream under a vocal.
Learned native-event lighting statistics remain optional. Modern box/filter and
arbitrary custom choreography are preserved, not claimed to be semantically
learned or perfectly rendered.

### Native output and editor verification

Native V4 uses indexed gameplay, a separate lightshow and sample-addressed shared
AudioData. Import, editable conversion, validation, training parsing and export
use the same codec. Actual exports found and fixed an arc tail-direction/color
confusion; regressions also cover a red tail encoded as zero on a blue arc. A
V3 downgrade refuses native NJS, distinct tail colors, object rotations and other
unsupported semantics instead of silently discarding them. Serializer changes
invalidate export caches.

The full HTTP pipeline generated three independently mapped difficulties and
exported native V4 with arcs, chains, Chroma and Noodle data. The final real-song
run produced 639/1,320/1,928 notes for Easy/Hard/Expert+, with four-second peaks
of 3/6/10.25 NPS and 23/47/59 optional native NJS events. All eight independent
Info, AudioData, gameplay and lightshow schemas passed, with zero current motion
errors/warnings. The saved user project and exported audio stayed unchanged,
and requested ArcAudio 3/local placement matched the effective engines without
fallback. This verifies the route and format, not game runtime. Expert+ still
has 71 of 214 active sections below 75% planned density, with no placement
rejections; the remaining attention/evidence limitation is not claimed solved.

The installed ArcViewer takes explicit V3 previews. Native-only content routes
to the browser Mod preview with a specific capability notice. Its sampled NJS,
local BPM, lane rotations, rewind behavior and paused edit-plane selection have
isolated browser regressions. The display states its approximation; it does not
claim to recreate Unity assets, shaders or exact game trajectories. Existing
editor tests cover light editing, mod tracks, scripts, stale-draft protection
and viewer snapshots without writing user maps.

### Real-output quality gate

Format validity and zero geometric errors are necessary, not sufficient. The
October 9 same-model/same-seed 15-map comparison passed 30 external schema checks
and current motion checks, but revealed substantial density starvation in the
opt-in attention routes. That run is retained as a failing musical-attention
baseline. Primary-only admission, a secondary slot occupied by inactive aliases,
and repeated composition budgets must be evaluated by actual active NPS and
source-selection evidence. Requiring observed attacks for the secondary slot
and permitting supported competing accents increased ArcAudio 3 high-activity
NPS from 5.62 to 7.93 on Neon Phoenix, 6.02 to 7.01 on the vocal recording,
and 5.70 to 7.69 on the screenshot song. The same fifteen-map comparison passed
thirty native schema checks and retained zero current motion errors/warnings.
The tested quiet vocal region stayed empty, strict importance filtering remained
effective, and OFF/Legacy output stayed unchanged.

The installed application also passed a read-only browser check on October 10:
actual ArcViewer Unity initialization, Mod preview, Light grid, Analytics and
Settings, with zero browser errors or writes to user projects. The separate
editor fixture checks note editing, scripts and unsaved draft handling.

[Audio evaluation](AUDIO_EVALUATION.md) records the current comparisons and
unresolved limits; no blanket quality percentage is assigned to these charts.

The final frozen-source fifteen-map pass after the accent, motif, budget and
precise-boundary corrections passed all thirty external schema checks, with
zero current motion errors/warnings. ArcAudio 3 high-activity NPS was 8.01,
7.11 and 7.80 on the same three songs (baseline 5.62, 6.02 and 5.70). The quiet
vocal region remained empty. Off/Legacy stayed byte-identical, and saved
projects plus the existing checkpoint/corpus remained unchanged. The remaining
active deficits are explicitly evidence-limited; they are not claimed solved.
The full table, artifact fingerprints and limits are in the audio evaluation.

### Severe strict-filter conflict and Inferno runtime repair

The reported 26-note Expert+ chart was traced from the persisted UI request:
13 NPS reached generation correctly, but minimum importance 0.75 rejected
6,511 of 8,407 detected attacks, leaving only 25 eligible timings. A 70%
density ceiling also capped the active target. The recommended 14 NJS was a
downstream response to low density; NJS was never used as a note-count target.
Explicit reset/help, review warnings, saved requested/measured NPS and a
measured severe-conflict guard now prevent this failure from appearing as a
successful near-empty map. Existing preferences and charts are not rewritten.

Inferno's Python 3.12 worker had skipped app-local `.pth` initialization and
Pillow dependencies. Proper site-directory setup now runs actual upstream
inference. Small sigmoid distributions also normalize without fixed-epsilon
attenuation, so Hybrid placement receives genuine model evidence. Stale runtime
verification and inference caches are versioned; optional setup installs the
compatibility dependency explicitly.

An isolated real API rerun with the saved request and only the importance/
ceiling controls reset produced 1,934 notes, 6.42 average NPS and 10.25 peak
four-second NPS. ArcAudio 3 + Inferno + ArcForge was effective without fallback;
all four native schemas and current motion checks passed. The original chart
remained unchanged. However, 97/217 active sections still fell below 75% planned
density: the 13 NPS goal is not claimed achieved. The audio evaluation records
the reproduction, recovery artifacts and explicit limits.

## October 10 follow-up: attention starvation versus timbre recognition

The reported 26-note Expert+ chart was traced to the saved attention settings,
not a conversion between NPS and NJS. On the 301.03-second, 182.5 BPM screenshot
recording, `minImportance: 0.75` rejected 6,511 of 8,407 candidates as
low-importance; the remaining attention checks left only 25 selected events.
`maxDensity: 0.7` was also saved. Inferno's failed runtime and explicit local
fallback are a separate placement problem; changing placement cannot restore
events filtered out earlier.

The weak Piano graph is a genuine classifier limitation: first-twelve-second
YAMNet Piano scores average approximately 0.01, versus approximately 0.20 for
Harp. Functional pitched timing evidence remains available. These model scores
are neither verified piano recognition nor source-separated stems. A harmonic
audio-view experiment produced worse timbre hypotheses and was not integrated.

Source fusion did contain an independent architecture defect: strong recognition
of one class suppressed DSP components from every family. Family-specific
fusion now preserves overlapping rhythm, pitched, bass and voice evidence.
Raw timbre scores are exposed as `instrumentEvidence`; guitar detection does
not automatically assert a lead role, and harmonic attacks are not relabeled
as proven piano. Six controlled regressions passed. Default-setting analysis
of the real recording selected 1,596 events before versus 1,612 after, with the
Piano curve unchanged. These are source-analysis results; recovered map output
has not yet been claimed validated here. See
[Audio evaluation](AUDIO_EVALUATION.md) and private artifacts under
`data/test-artifacts/piano-lead-diagnosis/`.
