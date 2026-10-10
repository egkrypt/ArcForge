# ArcForge 1.3

ngl tho this project was vibe coded with GPT 6.1 sol + My own super intelligence
https://media1.tenor.com/m/m3r_8NhxM9MAAAAC/cat-clapping-ai-cat-clapping.gif


ArcForge is a local Beat Saber creation studio: audio analysis, learned map generation, manual note and light editing, native ZIP export, and reviewable AI continuation. It runs without an API key. Music, projects and training stay on your computer; playlist downloads and optional model installation use the internet.

The existing installation opens with **Launch ArcForge.cmd** or at [127.0.0.1:8769](http://127.0.0.1:8769). Version 1.3 keeps the previous generator and editor, and adds optional **ArcAudio 3.0** classification to the ArcAudio 2.0 attention system. **Legacy remains the default.** The October 8–10 quality pass connects engine selection, temporal evidence, section planning, density feedback, shared lighting and native V4 formats. [Release scope and limitations](docs/RELEASE_1_3.md) distinguish measured behavior from runtime claims.

## Install from the source release

Use Python **3.10–3.12**, Node.js **22 or newer**, and npm. In PowerShell:

```powershell
.\"Setup ArcForge.ps1" -BuildFrontend
.\"Launch ArcForge.ps1"
```

The setup script creates `.venv`, installs pinned server requirements, and builds the frontend with `npm ci`. Add `-Recognition` for optional local classifier dependencies, or `-Cuda` for a supported NVIDIA CUDA runtime. Install checksum-verified YAMNet weights from the **ArcAudio** page when needed. Audio is not sent to Google. CPU recognition works without CUDA.

The public source archive deliberately contains **no songs, projects, training corpus, private trained checkpoints or installed model weights**. Import human-made level ZIPs in Training and train the local placement model before AI generation; at least three usable archives are required. Manual creation and ZIP import do not need a trained model. The installed workspace's existing trained models are not included in the release.

The source archive also omits the locally installed, separately hosted ArcViewer WebGL binary: its exact corresponding build source has not been established. The upstream reference project excludes prebuilt DLL/SO plugins, so a Unity rebuild requires restoring those dependencies. The browser **Mod preview** and **Legacy 3D** remain usable. See [viewer installation and build boundaries](integrations/VIEWER_SETUP.md). No game mods are installed by setup.

## Create and organize levels

**Levels** combines collection and creation. **Open level ZIP** imports V2/V3/V4 maps; **New level → AI** or **From Scratch** starts creation. Drop any audio format FFmpeg can decode; conversion produces Vorbis OGG at the original speed. Imported audio is forgotten by the next creation session unless **Remember imported audio** is enabled. Existing projects retain their audio.

Choose five native difficulty slots with custom names, Standard / 90° / 360° / No Arrows / One Saber, a mapping philosophy and musicality, technique, movement, variety, contrast and stamina goals. Difficulties are generated independently rather than by deleting notes from the hardest map. NPS is an active-phrase target, constrained by observed musical events, pauses and human-motion guards; the report explains evidence or spacing limits. Dynamic density preserves section contrast. NJS, jump distance, BPM, grid offset and preview start can be automatic or manually overridden.

The optional **Mapping regions** step supports movable, resizable, duplicated time ranges, primary and secondary philosophies, and inherited goal overrides. Skip it for automatic mapping. A style changes how an event is represented; it does not invent musical timing. Generated maps open automatically by default, configurable in Settings.

The mapper plans sections before placement, measures the generated density and reports musical-evidence or recovery limits. High-BPM observed subdivisions can support streams and bursts without halving every opportunity. Strong foreground accents can use learned two-hand emphasis at existing event times, consuming two note-budget units. Attention and learned composition share one phrase budget. Hand assignment, movement goals and repeated-motif evidence influence placement; the critic reports color balance, relative-motion repetition and actual-versus-planned NPS. These diagnostics are not calibrated human-likeness scores. [Generation audit and real comparisons](docs/GENERATION_QUALITY_AUDIT.md) document the changes.

Collection menus offer Open in Studio, Open in Explorer, Duplicate, Rename and Delete. Deletion uses recoverable trash by default; permanent deletion and trash purge are explicit Settings options. Shared or active audio/assets are preserved. Playtest, Analytics, Details and level-specific ArcAudio appear inside the selected project. Analysis preferences save independently of the chart and creation defaults.

## Audio understanding and attention

ArcAudio separates **what is audible** from **what deserves mapping attention**. Time-aligned source evidence feeds contextual states, a finite competing-source attention budget, phrase analysis and redundant/background-event suppression before candidates reach the existing mapper. Confidence and energy are distinct from importance.

- **Off / Legacy:** preserve the existing candidate pipeline.
- **ArcAudio 2.0:** mixed-audio DSP hypotheses plus contextual attention.
- **Hybrid:** compatible legacy rhythm evidence within the attention policy.
- **ArcAudio 3.0:** official pretrained YAMNet sound classification plus observed attack timing, attention and conservative compatible legacy evidence.

The inspector shares playback with waveform, beat grid, source lanes, selected/rejected events, context, attention, density, confidence and phrase boundaries. Controls cover musical priorities, competition, smoothing, density ceiling and supporting accents. They are stored per project when inspecting a level.

Audio-analysis and note-placement engines are independent controls. Explicitly unavailable InfernoSaber selection fails; Hybrid/Auto may use Local with a reported reason. Generated reports identify the requested and effective engines. The tempo inspector separates an attack-rate hypothesis from a supported meter change, so changing percussion subdivisions do not automatically double the beat clock. [Audio evaluation](docs/AUDIO_EVALUATION.md) explains controlled tests and real-source limitations.

The October 10 sparse-chart hotfix checks severe attention-filter conflicts before placement. A strict minimum-importance threshold that discards nearly all attacks in a long, active high-density request now stops generation with the actual rejection counts instead of exporting a near-empty chart as success. Attention controls include an explicit **Reset attention defaults** action; settings are never silently relaxed. A density ceiling below 100% reduces the requested active-phrase NPS ceiling. Saved-map details distinguish requested NPS, measured average/peak NPS and **NJS**, the notes' approach speed.

YAMNet is a **521-class sound classifier**, not a stem separator, lyric transcriber or piano-note recognizer. Classification windows never become note timestamps. Unrecognized instruments remain explicitly uncertain musical evidence instead of being relabeled as drums. Dense mixes may still yield mostly unknown-instrument evidence; the optional classifier is not a claim of perfect detection or better perceptual sync. Listen and inspect the grid. [Architecture, contracts and evaluation](ARCAUDIO.md) describe the implementation.

## Manual studio

Hover a grid cell to see a placement ghost and click to add a note. Red is left hand and blue is right hand in player orientation. The waveform sits beside the beat grid. The palette starts collapsed; Beat grid, Light grid and Mods panels can be shown independently. Snap accepts fractions or decimal beat intervals.

| Control | Action |
| --- | --- |
| Q W E / A S D / Z X C | Up-left, up, up-right / left, dot, right / down-left, down, down-right |
| F | Invert cut direction |
| Tab | Switch red/blue hand |
| Wheel up / down | Forward / backward in song time |
| Ctrl + wheel | Zoom |
| Left / right arrow | Step on beat snap |
| Space | Play / pause |
| Escape / Delete | Clear / remove selection |
| Ctrl + Z / Ctrl + Y or Ctrl + Shift + Z | Undo / redo |
| Ctrl + S | Save |

Direction keys change the placement ghost while hovering. Shortcuts & scroll includes inverse direction keys and the wheel multiplier. Notes and ghosts anchor to the edit plane while paused. Place or edit notes, bombs, walls, arcs, chains, BPM markers and tempo ramps. Light grid edits native events, light IDs, colors, fades, ring/laser motion, boosts and modern event-box JSON.

**Check current draft** reports possible repeated same-hand directions, insufficient recovery, geometry, visibility and collision risks. Findings seek to the affected timestamp. Checks do not edit or save the map. Intentional resets may be valid; these are advisory geometric estimates.

**Let AI Continue** fills one or several gaps using surrounding rhythm, density, movement and both hand boundaries. Fill preserves existing content; Replace and Continue are explicit alternatives. An unsaved proposal supports Accept, Regenerate, Discard and revision-aware Undo. Existing lighting and other authored objects are retained by default. A stale preview cannot silently overwrite later edits.

## Preview, effects and compatibility

ArcViewer is a separate Unity WebGL player. Installed builds receive an **unsaved native ZIP snapshot** without saving the project; optional idle refresh reloads that snapshot. The live browser Mod preview reflects supported transforms immediately. Legacy 3D remains available. ArcViewer reconstructs 13 environments plus the Crab Rave alias; unsupported scenes visibly fall back to Default. All 45 catalogued environment names remain available for export.

| Component | Editing and export | Preview boundary |
| --- | --- | --- |
| Chroma | Colors, IDs, native lights, rings/lasers, environment/geometry data | Supported lights and transforms; not every Unity environment enhancement |
| Noodle Extensions | Object/track/path keyframes, positions, rotations, scale, dissolve, time | Documented properties approximated in browser; arbitrary choreography needs game testing |
| Mapping Extensions | Precision coordinates and custom directions | Precision positions/directions; incompatible with Noodle selection |
| Vivify | Imported bundle metadata/assets, prefab lifecycle and custom-event data | Lifecycle inspection/placeholder transforms; compiled meshes/shaders/post-processing require Beat Saber |
| CustomJSONData / Heck | Editable custom data and framework dependencies | Frameworks, not independent visual effects |
| Synapse | Identified as an external network plugin | Not a map-visual renderer |

Generated lighting uses learned human basic-event behavior, balanced red/blue banks, accent rotations, laser motion, fades, brightness and boosts. Gameplay and lighting share tempo, attention and section planning. Lighting styles change articulation and intensity rather than just a label. Optional Noodle note entrances settle before the hit plane. Imported modern light boxes and arbitrary custom events are preserved and editable; their complete semantics are not learned or reconstructed.

Mods exposes transforms, tracks, keyframes, environment geometry, asset metadata and JSON. Scripts use **Run → Review → Apply** with undo; treat them as local project tools, not a Unity plugin host. Vanilla, Ranked-compatible, Modded and Unrestricted targets constrain export features. Ranked-compatible is not a ranking certificate.

Advanced legacy V2 mod charts support viewing and unchanged export; convert them to V3 in ChroMapper before editing their animations. **Native export format** selects V3 (default) or genuine indexed V4 gameplay with separate lightshows and shared AudioData timing. V4 import and training input are supported. Imported native metadata is retained, and unsupported V4-only downgrades fail explicitly. Native V4.1 NJS or per-object rotation support still depends on the game/viewer runtime. See [native formats and preservation rules](docs/NATIVE_FORMATS.md).

## Learn from human maps

Upload ZIPs, import **Learn from these levels**, or upload a BeatSaver `.bplist`. Playlist downloads use the official API, verify map hashes, reject declared AI maps and report failures. Incremental import skips validation/copy for unchanged archives; metadata and feature caches invalidate changed content. Cancellation retains the last good model.

The current development corpus contains **605 archives / 1,767 charts**. It is private local input, not bundled release data. The placement network learns audio-conditioned positions/directions; optional whole-map learning also learns phrase density, movement, complexity, doubles, recurrence variation and empty phrases. Lighting training uses actual basic events, colors, phase, brightness and boosts across complete lightshows. Whole-map conditioning is a statistical model, not unrestricted planning or copied block templates.

The measured local composition run used **261,851 phrase examples**, 605 decoded songs and **10,852,398 basic lighting events**. Archive-held-out prediction losses are reported separately by target; they do not establish enjoyable gameplay. Modern box counts and custom events are recorded without claiming their choreography has been learned. Training shows actual model reports, device and recommended epochs.

Auto / CPU / GPU training chooses a verified runtime. Explicit GPU selection fails clearly if unavailable. CPU uses bounded numerical-library threads; CUDA accelerates the local placement network. Different phases use different resources, so low CPU or memory consumption alone does not indicate a stalled trainer. The app does not promise to combine CPU and GPU for every operation.

Optional InfernoSaber v1.7.1 inference uses frozen upstream `fav_15` TensorFlow weights in a separate CPU process when installed. Uploads train the separate ArcForge network, not those upstream weights. The source release omits all weights. [Training, caches, playlist security and runtime boundaries](integrations/TRAINING_RUNTIME.md) contain details.

Use `Setup ArcForge.ps1 -Inferno` to install the optional Inferno CPU runtime. Its startup initializes app-local package hooks, including the pinned setuptools compatibility needed by TensorFlow on Python 3.12. Failed runtime checks invalidate stale success records. Inferno and Local placement consume audio-selected candidate times; a placement-engine fallback does not create missing timing candidates.

## Development and release checks

Original backend/engine code is MIT. The ChroMapper-derived frontend is GPL-2.0-only with complete frontend source and adapted original reference files. ArcViewer is a separate GPL-3.0 program. Licenses and attribution remain with each component; user music/maps are not relicensed. See [third-party notices](THIRD_PARTY_NOTICES.md), [frontend dependency licenses](docs/FRONTEND_LICENSES.md), [provenance](integrations/PROVENANCE.md) and [source manifest](integrations/source-manifest.json).

`server.py` serves the built frontend on loopback port 8769 and validates exact loopback hostnames for both reads and writes. React/TypeScript source is in `frontend/src`; audio/generation modules are in `engine`. `.venv`, installed dependencies, `data` and `models` are local runtime directories. Settings include themes, reduced motion, audio retention, workflow and guarded restart/stop controls.

Use `python -X utf8 tools/verify_release.py` for fixture-only DSP, tempo, engine/API, continuation, lighting, native-format and storage checks; `npm run build` in `frontend`; and `python -X utf8 tools/package_source.py --audit-only` to inspect the private-data-free release whitelist. Build the deterministic source ZIP with `python -X utf8 tools/package_source.py`. Audit every archive entry/hash with `python -X utf8 tools/verify_release.py --archive release/ArcForge-1.3.0-source.zip --archive-only`. Trained-checkpoint and real-song evaluations remain separate; do not run arbitrary old example scripts against authored projects.

The [1.3 checklist](UPDATE_1_3_CHECKLIST.md), [ArcAudio checklist](ARCAUDIO_V2_CHECKLIST.md), [1.1 checklist](UPDATE_1_1_CHECKLIST.md) and in-app Changelog record scope. Static motion checks, attack-distance measurements and numerical-model agreement cannot replace listening or human VR playtesting.

Optional browser checks use an installed Playwright package and Edge by default. Install test-only packages separately (for example `npm install --prefix .test-deps --no-save playwright bsmap@3.0.3 valibot`), set `ARCFORGE_PLAYWRIGHT_MODULE` to its Playwright package directory, and optionally set `ARCFORGE_BROWSER_CHANNEL`. Fixture browser checks use mocked APIs; `browser_13_live.mjs` is a read-only installed-server smoke test. For `validate_exports.mjs`, set `ARCFORGE_BSMAP_ROOT` to the installed bsmap 3.0.3 package directory and pass an isolated extracted map folder; it writes a schema report there. `validate_real_v4.py --summary PATH` accepts a locally generated benchmark summary, uses the same schema setting, and resolves Node from PATH or `ARCFORGE_NODE`. These optional packages are not app runtime dependencies.
