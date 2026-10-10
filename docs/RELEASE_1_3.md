# ArcForge 1.3.0 source release — 10 October 2026

This source snapshot includes the generation-quality audit and fixes within
the existing 1.3 release. It is a self-hostable development release, with
measured behavior and explicit runtime boundaries. It is not a claim of
perfection, human-level authorship, ranking approval or headset validation.

## What changed in this quality pass

- Audio analysis and placement engines are independent choices. Effective
  engines, fallbacks, rejection causes and source provenance are reported.
  Explicit unavailable Inferno selection fails instead of silently using Local.
- Harmonic/percussive evidence is measured before contextual attention; unknown
  classifier labels no longer erase useful pitched or rhythmic evidence.
  Legacy, ArcAudio 2, Hybrid and ArcAudio 3 retain different candidate policies.
- Tempo analysis preserves established meter through percussion subdivision
  changes, while supported acceleration/deceleration can change the beat clock.
- A section plan and critic track actual-versus-planned note density, hand
  balance, recovery constraints and variation. Supported musical opportunities
  are scheduled across hands; extra notes are not invented to satisfy quotas.
  Foreground accent doubles use learned emphasis at existing attack times and
  consume two note-budget units. Attention and learned composition share one
  active-phrase budget instead of reducing the same density repeatedly; exact
  phrase boundaries retain ownership of their opening attacks. Recurring
  musical motifs retain a shared identity across separate phrase IDs so their
  mapping can develop a recognizable variation.
- Gameplay and lighting share tempo, attention and section structure. Lighting
  styles change event articulation and intensity, with native red/blue banks.
- Genuine native V4 uses indexed gameplay, split lightshows and a shared sample
  clock. Import, training input and export preserve native data. Unsupported
  downgrade semantics fail explicitly. V3 remains the default.
- Setup and launch resolve supported Python from the app-local virtual
  environment or standard Windows Python installation, without a Codex cache
  dependency. Launching an already running service opens its page only.
  Stop identifies the local service and uses its graceful lifecycle endpoint,
  preserving active-job guards rather than killing a recorded process ID.
- The loopback service rejects unrelated, duplicate or malformed Host headers
  before any route, including reads. The source archive verifier also rejects
  unsafe Windows path forms and verifies every declared file hash.

See [generation audit](GENERATION_QUALITY_AUDIT.md),
[audio evaluation](AUDIO_EVALUATION.md) and [native formats](NATIVE_FORMATS.md)
for mechanisms, measurements and limitations.

## October 10 sparse-chart hotfix

- Reproduced a 26-note, five-minute Expert+ result as an attention-filter
  conflict: minimum importance 0.75 discarded almost all attacks, while a 70%
  density ceiling also constrained the 13 NPS active-phrase request. NJS was
  not used as the NPS target. Long, active, high-density requests with this
  demonstrable severe rejection conflict now fail before placement and report
  the detected, eligible and rejected counts. Sparse music, short excerpts,
  continuation, manual regions and Legacy retain their existing behavior.
- The attention UI explains strict rejection thresholds and density ceilings,
  provides an explicit reset to defaults, and displays requested NPS separately
  from saved-map average/peak NPS and approach-speed NJS. No user setting is
  silently overridden and no random timestamps are added to satisfy a quota.
- Optional Inferno startup processes app-local package hooks and initializes
  the setuptools compatibility layer needed for Python 3.12 TensorFlow.
  `Setup ArcForge.ps1 -Inferno` installs the pinned optional CPU dependencies;
  stale or failed runtime verification no longer appears as verified. Tiny
  nonzero model outputs retain their relative probability evidence.
- Recognition fusion no longer lets one named sound family erase unrelated
  measured components in an overlapping mix. The inspector distinguishes
  piano, guitar and plucked-string classifier hypotheses from functional
  melody/lead evidence. These are sound-class hypotheses, not validated piano
  transcription, source separation or a claim that this song's piano is now
  recognized correctly.

The release remains version 1.3.0. Its rebuilt source archive and adjacent
checksum identify this hotfix snapshot; earlier 1.3.0 archives have a different
hash. See the audio evaluation for the actual song rerun and residual limits.

## Source archive boundaries

The release is assembled by an explicit whitelist. It contains application
source, the built frontend and its full source, setup/launch scripts, selected
upstream source references and licenses, documentation and evaluation tools.
Exact locked React, React DOM, Scheduler, Three.js and Lucide license texts
accompany the bundled frontend in [frontend notices](FRONTEND_LICENSES.md).
Retained ArcViewer source includes its original short interface hit sounds.

It contains no user songs, level ZIPs, projects, training corpus, private
checkpoints, classifier weights, installed dependency directories, logs,
runtime caches or local settings. The separately installed ArcViewer WebGL
WASM/data build is excluded because its exact corresponding source is not
established. Its retained source snapshot is a separate reference project.
Prebuilt upstream DLL/SO plugins are also excluded; their retained source
references, metadata and licenses do not substitute for those build inputs.
Rebuilding that separate Unity project requires restoring its dependencies.
No server, public upload or publication is performed by packaging.

Each archive includes `SOURCE_MANIFEST.json` with every source file's SHA-256
and byte count. ZIP entry timestamps and order are fixed. The adjacent
`.sha256` file identifies the exact archive. A source change requires a new
archive and checksum even when the application version remains 1.3.0.

## Verification commands

From a checkout with the pinned server requirements installed:

```console
python -X utf8 tools/verify_release.py
powershell -NoProfile -File tests/test_launchers.ps1
python -X utf8 tools/package_source.py --audit-only
python -X utf8 tools/package_source.py
python -X utf8 tools/verify_release.py --archive release/ArcForge-1.3.0-source.zip --archive-only
```

The fixture suite uses synthetic audio and temporary map data. It does not
require a trained checkpoint or download recognition weights. Checkpoint-based
mapping regressions and real-audio comparisons are separately reproducible
with locally imported songs and a trained placement model. Their private
artifacts stay under `data/test-artifacts` and are not bundled.

Run `npm ci` then `npm run build` in `frontend` after UI changes. Optional
browser checks require an independently installed Playwright runtime. Native
schema checks require bsmap 3.0.3 and Valibot; no validation package is bundled.
`node tests/test_native_preview.mjs` checks native NJS easing and lane rotation
semantics using the installed TypeScript compiler. The optional
`tests/browser_native_preview.mjs` exercises the real WebGL editor with mocked
APIs; it does not load or save user projects.

## Remaining runtime limits

Mixed-audio instrument names remain uncertain, and YAMNet is not a separated
stem or lyric model. High importance thresholds can legitimately yield sparse
maps. Strong notes-per-second goals cannot override insufficient musical
evidence or physical guards. Similarity and motion metrics are technical
diagnostics rather than calibrated musical-quality scores.

V4.1 NJS and per-object rotations need a compatible game/runtime. Legacy
ArcViewer receives V3-compatible previews and refuses unsupported downgrades.
Browser mod previews approximate documented properties; Vivify shaders,
prefabs, post-processing and arbitrary Unity choreography still require
game-side testing. Ranking compatibility and comfortable VR play require
human review. Training hardware and recognition dependencies are optional,
installed separately and never redistributed in the source archive.


## Regional mapping and Deep analysis follow-up

The creation wizard's existing Mapping regions step now controls local NPS targets/ranges, style, profile goals and Auto/Full Mix/weighted Custom Sources. The same settings save with each level and feed section revision and continuation. Exact boundaries, coincident-event merging, missing-evidence warnings and neighboring fades are covered by regression tests. Map Details displays requested and actual regional NPS. Source controls require an ArcAudio attention engine; style/NPS-only regions remain available in Legacy.

ArcAudio 3.0 Deep uses a 0.12-second classifier stride, available CPU threads, adaptive CUDA batches and a bounded concurrent CPU source-feature worker. Existing projects retain Balanced's original 0.48-second stride. Full-song evaluation verified 2,502 versus 627 windows and actual changed attention; no improved instrument-identity accuracy is claimed. The opening piano timbre remains uncertain.

This follow-up passed 213 release fixture tests, extended browser control contracts, two full real-song generations and eight external native V4 schema checks. Full Mix achieved 10.25 NPS versus the 7.19 Auto control in the selected drop; strict Spoken produced only seven spoken-evidence notes in 4:14–4:20. See AUDIO_EVALUATION.md for evidence, limits and reproducible tooling.
