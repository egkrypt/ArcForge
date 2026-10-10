# ArcAudio 2.0 / 3.0 and guided mapping

ArcAudio is an optional decision layer between the existing music analysis and hit selection. It keeps source evidence, musical attention, and physical note placement separate. Off and Legacy use the prior candidate pipeline; ArcAudio selects observed important attacks; Hybrid admits compatible legacy rhythm within the same attention policy.

## Analysis contract

`POST /api/audio/{audioId}/attention` accepts `mode` and `settings`. Time-aligned frames expose source hypotheses, source confidence, energy, spectral activity, rhythmic activity, context state, primary/secondary attention, finite attention shares, density target and decision confidence. The response also contains all event decisions with reasons, selected events, phrases, transitions, limitations and measured execution time.

The source layer uses the existing spectral roles plus harmonic/percussive masks, relative harmonic concentration and envelope modulation. Vocal, spoken and piano evidence from mixed recordings is uncertain. There is no installed trained stem separator or lyric transcriber. The adapter also accepts independently separated or annotated `sourceEvidence`; its confidence and provenance are explicit. Controlled attention tests are not evidence of real instrument recognition accuracy.

Context decisions use both preceding and following frames, smoothing, minimum residence and a switching margin. Sources compete for a finite budget. Sustained background, coincident aliases, weak supporting layers and crowded nearby events can be rejected before reaching the mapper. Energy is recorded separately from importance. Observed expressive attacks stay off-grid when appropriate; only close measured beat matches may be snapped by the downstream mapper.

Settings cover vocal/drum priorities, attention strength, competition, density ceiling, minimum importance, secondary accents and smoothing. Three advanced controls cover phrase awareness, background suppression and repetition suppression. Settings-independent source features are cached in process; decision responses are cached on disk by source identity, code fingerprint, grid and settings.

## Guided creation

AI creation includes an optional Mapping regions step immediately after the song. Automatic/Skip requires no regions. From Scratch keeps the existing seven-step workflow.

Each half-open audio-time region inherits the global style and six goals unless explicitly overridden. Primary and up to two secondary philosophies form one weighted objective. Regions may be moved, resized, duplicated, edited or removed. Overlapping ranges are rejected. Styles and goals affect supported event selection and learned placements; they do not manufacture note times.

## Let AI Continue

Studio and Map Details offer Fill, Replace and Continue operations for multiple ranges. The default Fill protects existing notes both inside and outside the ranges. The generator studies local density, hand trajectories, rhythm and vocabulary on both sides. The adapter reranks new placements against the fixed boundary notes and maps exact audio times into the original chart clock, including imported song offsets and tempo changes.

Generation creates an unsaved comparison candidate. Accept writes a new revision; Regenerate creates another proposal; Discard leaves the saved map unchanged. Undo restores the pre-accept chart as a new revision. A preview based on an older revision is rejected. A later manual edit also blocks direct continuation undo. Dirty editor work must be saved before requesting a candidate and cannot silently be replaced by accepting a stale preview.

Existing lighting, bombs, walls and custom data remain intact in continuation by default. A boundary that cannot be reconciled with fixed authored timing is reported for review. This is a static geometric estimate, not a simulation of a person's body in VR.

## Manual draft checks

Check current draft runs timing, reach, recovery, parity, obstruction and collision checks on the unsaved chart. A repeated blue down-cut can produce a possible repeated-direction warning. Intentional resets with enough recovery may be valid. Findings seek to their timestamps; checks never apply fixes or save edits.

## Human-map attention supervision

`POST /api/projects/{projectId}/attention/supervision` exports weak audio-event/context/human-hit relationships. A human hit indicates represented timing, not a verified instrument label. Coincident sources share ambiguous positive weight; omitted events are not proof that their source was unimportant. Decorative non-interactable notes are excluded and original song offsets are included. The export is training-compatible evidence, not a trained attention classifier.

## Evaluation

Run `python -X utf8 -m unittest discover -s tests -p 'test_arcaudio.py'` for decision tests including quiet vocal/piano, voice/drums, drops, dense choruses, speech, piano, drums, busy instrumental, competing sources, transitions, recurrence and sparse arrangements. `tests/evaluate_arcaudio.py` records controlled attention agreement separately from full-song event reduction, timing consistency, stability and cold/warm performance in `data/arcaudio-evaluation.json`.

`test_generation_attention_regions.py` checks identical Default/Off/Legacy maps with the same current checkpoint, observed vocal times, hybrid suppression and guided boundaries. `test_regions_continuation.py` checks both boundaries and preservation; `test_attention_api.py` checks cache invalidation, advisory-only edits, preview acceptance conflicts and undo.

Real instrument detection accuracy and subjective musical improvement remain unmeasured without annotated stems and human listening/playtests. The UI reports these limits alongside the decisions.


## ArcAudio 3.0 recognition

ArcForge 1.3 adds optional `arcaudio3`, locally executed pretrained source classification feeding the existing attention abstraction. Legacy remains the default. Version 3 does not replace placement, editing or native formats.

`engine/recognition` independently executes the official Apache-2.0 YAMNet architecture with PyTorch, using Google's unchanged HDF5 weights. Installation is optional and checksum-verified. Its 521 AudioSet class scores feed the existing source vocabulary, with explicit provenance and uncertainty. The approximately 15.3 MB weight download contains no repository archive; audio never leaves the computer. CPU is supported and CUDA is used when available/requested by the local runtime.

The model uses 16 kHz log-mel input, approximately 0.96-second windows and 0.48-second hops. Those windows describe source evidence; they are **not musical-event timestamps**. Classification attaches to measured DSP/legacy attacks. It cannot identify individual piano pitches, words, syllables or isolated stems. Generic music plus observed transient activity can remain `other` with capped confidence; it never becomes a confident drum label merely because its spectrum has a strong attack.

```text
Decoded audio → DSP / optional trained classification → source evidence
  → temporal context / phrases → competing attention → important observed events
  → compatible beat candidates → existing neural placement / motion guards
```

Attention supports single focus, primary with secondary accents, and justified combinations. Smoothing and minimum residence avoid frame-by-frame switching. High energy can coexist with low importance. Low-confidence classification encourages conservative decisions while unknown-music evidence avoids incorrectly treating audible music as silence.

Version 3 candidate selection uses measured phrase boundaries and tempo changes rather than assuming a fixed global four-beat duration. Manual region edges split groups exactly. Requested NPS applies to active musical phrases, with dynamic density and evidence/physical-spacing limits. It does not force every source into the map or invent note times to satisfy a quota. Historical golden hashes remain unchanged; a user-trained replacement checkpoint may change placements independently of implementation.

### APIs and persistence

- `GET /api/recognition/status`: dependencies, installed/verified model and runtime status.
- `POST /api/recognition/install {}`: asynchronous checksum-verified optional model installation.
- `POST /api/audio/{audioId}/attention`: `mode: "arcaudio3"` selects trained evidence plus attention.
- `PATCH /api/projects/{projectId}/analysis`: updates analysis/learning configuration without replacing charts or global creation defaults.
- `POST /api/projects/{projectId}/preview`: exports an unsaved draft snapshot without saving the project.

Recognition score caches depend on audio identity, weights and implementation version. Attention responses also depend on grid, settings and evidence fingerprints. Legacy/Off do not invoke recognition. The public release contains neither weights nor user caches.

### Measured evidence, 2026-10-07

The independent TensorFlow/PyTorch check on four patches compared all 521 classes: maximum absolute feature error **1.31 × 10⁻⁵**, maximum class-score error **9.89 × 10⁻⁶**, below the 3 × 10⁻⁴ tolerance. This establishes numerical agreement, not real-song source accuracy.

Two inspected local full songs reduced 5,761 → 1,430 and 7,327 → 1,684 candidate events, with approximately 2.1 source switches per minute. Much of the dense-mix evidence remained unknown-instrument music. These results establish conservative selection/stability, not correct identification of every singer or piano, nor perceptual superiority to Legacy.

A 24-second / 49-window fixture measured CPU first inference **2.099 s**, warm **0.066 s**; CUDA first inference after CPU initialization **3.711 s**, warm **0.030 s**. These specific runtime numbers exclude a fresh process's framework/device initialization and are not a hardware-independent speed guarantee. Earlier first-use execution took substantially longer. Cache hits and cold initialization must be reported separately.

The final real-generation check compared Legacy and v3 on two current songs with the same target settings. Median/p95 distances to detected attacks were 11.32/60.39 ms versus 7.22/34.83 ms on one track, and 7.40/26.49 ms versus 7.31/46.44 ms on the other. The result is mixed; v3 is not universally better. All four generated fixtures passed the implemented guards and eight native Info/V3 schema checks, with zero consecutive identical four-beat signatures. These checks do not establish absence of musical repetition or perfect perceptual synchronization. Average NPS varied with supported evidence and dynamic density. No human VR quality score or ranking certificate is claimed.

`test_recognition_import.py` covers unknown fallback, piano-label versus drum-band disagreement, optional real inference/numerical agreement and incremental import. `test_continuation_workflow13.py` verifies real neural generation, review without saving, regeneration, acceptance, undo and native ZIP export using an isolated authored fixture. `tools/verify_release.py` selects corpus-independent checks for fresh installs. Recognition-dependent tests skip when optional weights are absent; other model-dependent tests need a locally trained checkpoint.

### Future learning boundary

Human-map supervision exports audio/context/event-to-hit evidence with ambiguous source attribution. It is suitable for future learned attention training; the current attention policy is a transparent temporal/DSP decision system with optional pretrained source evidence, **not a newly trained human-attention classifier**. Whole-map and lighting learners are separate statistical models trained from complete human charts. They do not supply verified instrument labels.

Primary model reference: [official YAMNet source](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet). See [research provenance](integrations/PROVENANCE.md) for the additional repositories read without downloading code or restricted model assets.


## Analysis compute and regional controls — 10 October 2026

ArcAudio 3.0 has independent **Balanced / Deep** quality and **Auto / CPU / CUDA** device controls. Balanced preserves the existing 0.48-second classifier stride. Deep uses a 0.12-second stride with the same trained 0.96-second input window, all available CPU threads by default, threaded spectral features and adaptive CUDA batches. Deep CUDA overlaps one bounded CPU source-feature/HPSS worker with classifier inference. Batches leave VRAM headroom and retry the same unconsumed windows with a smaller batch after CUDA out-of-memory. CPU threads are restored afterward. Explicit CUDA requests fail if CUDA is unavailable; Auto may use CPU. Existing projects without compute settings remain Balanced. The new creation defaults offer Deep when ArcAudio 3.0 is selected; Legacy remains the selected engine unless changed.

No extra model weights are needed. More overlapping windows improve temporal coverage, **not guaranteed instrument identity, source separation or precise vocal transcription**. Exact note opportunities still come from observed attacks, not classifier-window centers. Measured device, patch count, stride, threads, batch, timings, overlap and memory are recorded. Cached measurements identify that inference did not run again.

Optional Mapping regions now support the following in the existing draggable/resizable timeline:

- Inherited NPS, a target NPS, or an activity-responsive minimum/maximum NPS range.
- Inherited or local mapping style, up to two secondary influences and per-goal movement/profile overrides.
- **Auto** keeps the selected engine's automatic dominant-source attention.
- **All Sources / Full Mix** considers all observed components under one finite information/note budget. Coincident detections remain one musical instant, and sustained background or insufficient evidence remains rejectable.
- **Custom Sources** selects weighted sources, either exclusively or as primary sources with supporting accents. Drums includes kick, snare and percussion; selecting Kick alone does not include all drums. Lead is a broad guitar/synth evidence role, not verified guitar transcription.
- Optional edge transitions blend density, movement and style objectives inside each range. Adjacent ranges with transitions meet coherently; zero preserves intentional changes. Strict source allowlists never extend beyond the exact half-open `[start,end)` range.

An example creative directive is:

```json
{
  "analysisCompute": {"quality":"deep","device":"auto","cpuThreads":0},
  "arcAudioMode":"arcaudio3",
  "mappingRegions":[
    {"id":"speech","name":"Spoken bridge","start":254,"end":260,
     "primaryStyle":"Flow","nps":3.5,"analysisMode":"custom",
     "sources":{"spoken":1},"sourcePolicy":"only"},
    {"id":"drop","name":"Full mix drop","start":30,"end":46,
     "primaryStyle":"Speed","npsRange":[8,13],"analysisMode":"all"}
  ]
}
```

Source weights express creative intent; detection confidence remains unchanged. A missing source yields no invented hits and a warning. Source controls require ArcAudio 2.0, Hybrid or ArcAudio 3.0; Legacy/Off remain available for style/NPS-only regions and reject unsupported source directives explicitly. Regions are cached, saved with the level, and carried into generation, section revisions and continuation. A continuation candidate retains surrounding audio attention and intersects the gap with existing regional style/density goals. Existing notes outside a selected revision/continuation interval remain preserved by the existing merge workflow.

Actual NPS is measured after generation and displayed per region in Map Details. It includes rests and can be lower than the goal because of source evidence, density ceilings, difficulty or hand recovery. Range goals use musical activity once rather than applying the density reduction twice. Neither targets nor ranges create random timestamps to fill a quota.

### APIs and reproducible evaluation

- `GET /api/analysis/runtime`: actual CPU/CUDA availability and policy.
- `POST /api/audio/{id}/attention`: `mode`, `settings`, `compute`, `mappingRegions`, optional legacy `sourceFocusRegions`.
- Generation uses `analysisCompute` and `mappingRegions`; section/continuation analysis overrides accept those same fields.
- `PATCH /api/projects/{id}/analysis` saves these fields without replacing notes.
- `tests/benchmark_analysis_compute.py` measures actual classifier/full-recognizer coverage, cold/warm timings, memory and repeatability without claiming detection accuracy.
- `tests/evaluate_mapping_source_regions.py` compares real generated regional note evidence, NPS, boundary ownership, outside-region attention and native V4 output using private local audio/model assets.

The compute policy follows the official [YAMNet parameters](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/params.py), [PyTorch thread controls](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html) and [CUDA free-memory API](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.mem_get_info.html).
