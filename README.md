# ArcForge 1.3

> **In short:** ArcForge is a local, open-source Beat Saber map creator that can analyze songs, generate maps using a model trained on human-made levels, edit notes and lighting, continue parts of a map with AI, and create AI-generated modded content such as Noodle Extensions and Chroma data. You can import BPLISTs to build your own training dataset without manually downloading hundreds or thousands of maps one by one. It runs locally, doesn't need an API key, and can run on fairly normal hardware.

ngl tho this project was vibe coded with GPT 6.1 sol + my own super intelligence.

https://media1.tenor.com/m/m3r_8NhxM9MAAAAC/cat-clapping-ai-cat-clapping.gif

## What is ArcForge?

ArcForge is a Beat Saber creation studio I'm working on to see how far AI-generated maps can go, especially when the model learns from actual human-made maps, including ranked ones. The goal isn't just to put notes on a beat. I want the maps to feel more like someone actually mapped the song instead of the usual obvious AI-generated patterns.

It runs on your own computer. Your music, projects, and training data stay local. You don't need an API key or a cloud AI service. The app uses the internet only for things like downloading playlist maps or installing optional model components.

ArcForge includes:

- **AI map generation** using a locally trained placement model.
- **Training from human-made maps**, including maps imported from BeatSaver playlists (`.bplist`).
- **Audio analysis** to help find musical events, sections, rhythm, and mapping attention.
- **Mapping regions** to give different parts of a song different styles and NPS goals.
- **AI continuation** to fill selected gaps while taking the surrounding map into account.
- **Manual map editing** for notes, bombs, walls, arcs, chains, BPM markers, and tempo ramps.
- **Lighting creation and editing**, including basic events and modern light-event data.
- **AI-generated modding data** for supported features such as Chroma and Noodle Extensions, plus tools to inspect and edit modded map data.
- **3D and mod previews** for checking supported content before exporting.
- **Native ZIP export** for Beat Saber map formats, including V3 and V4.
- **Local training and model options**, with CPU support and optional NVIDIA CUDA acceleration.

This is still a work in progress. Some modded features can be edited or exported even when the browser preview can't fully reproduce how they look in Beat Saber. Always test modded maps in-game.

## Run ArcForge

### If you already have an installed build

Start it with **`Launch ArcForge.cmd`**, then open [http://127.0.0.1:8769](http://127.0.0.1:8769).

### Build it from the source release

You'll need:

- Python **3.10–3.12**
- Node.js **22 or newer**
- npm

Open PowerShell in the ArcForge folder and run:

```powershell
.\Setup ArcForge.ps1 -BuildFrontend
.\Launch ArcForge.ps1
```

The setup script creates the Python virtual environment, installs the pinned server dependencies, and builds the frontend.

Optional components:

```powershell
.\Setup ArcForge.ps1 -Recognition
.\Setup ArcForge.ps1 -Cuda
.\Setup ArcForge.ps1 -Inferno
```

Use the relevant option when you want the optional local sound classifier, supported NVIDIA CUDA runtime, or Inferno CPU inference runtime. Install checksum-verified YAMNet weights from the **ArcAudio** page when needed. Audio isn't sent to Google.

**Important:** the public source release does not include songs, projects, private training data, trained checkpoints, or installed model weights. Import human-made map ZIPs in Training and train the local placement model before using AI generation. At least three usable map archives are required. You can still create maps manually or import ZIPs without training a model.

The source release also doesn't include the separately hosted, locally installed ArcViewer WebGL binary. The browser **Mod preview** and **Legacy 3D** are still available, and no game mods are installed by setup. See [viewer installation and build boundaries](integrations/VIEWER_SETUP.md).

## Make a map

1. Open **Levels**.
2. Choose **Open level ZIP** to import a map, or **New level → AI** to generate one. You can also start **From Scratch**.
3. Drop in an audio file that FFmpeg can decode.
4. Choose your difficulty slots, game mode, mapping goals, and other settings.
5. Generate the map, then open it in Studio to review and edit it.
6. Export the map as a ZIP and test it in Beat Saber.

You can set up to five difficulty slots with custom names. Available modes include Standard, 90°, 360°, No Arrows, and One Saber. Mapping goals include musicality, technique, movement, variety, contrast, and stamina.

Difficulties are generated independently; ArcForge doesn't just create the hardest map and delete notes to make easier ones. NPS can be targeted dynamically, but it still depends on the musical events the audio analysis finds, pauses in the song, and movement constraints. The generation report explains when the evidence or spacing limits what it can make.

You can choose NJS, jump distance, BPM, grid offset, and preview start automatically or set them yourself. **Mapping regions** are optional: add, resize, duplicate, and move time ranges, then assign different mapping styles or goals to each part of the song. Skip this step if you want ArcForge to handle the whole song automatically.

## Train it with human-made maps

This is one of the main reasons I'm building ArcForge: I want to see how much training on real human-made maps, including ranked maps, improves the results.

In **Training**, you can upload map ZIPs, choose **Learn from these levels**, or import a BeatSaver `.bplist`. ArcForge can download the maps in a playlist for you, check their hashes, and report failed downloads. You don't need to manually install 1,000 maps in Beat Saber just to build a dataset.

The current development corpus contains **605 archives / 1,767 charts**. That's my local development dataset, not something bundled with the public source release.

The model learns audio-conditioned note positions and directions. Optional whole-map learning also looks at things like phrase density, movement, complexity, doubles, pattern variation, and empty phrases. Lighting training learns from basic events, colors, timing, brightness, and boosts.

Training results and prediction losses can help measure what the model is learning, but they don't prove that a map is fun, comfortable, well-synced, or genuinely human-like. That's why actual listening and VR playtesting still matter.

ArcForge can use CPU training, and supported NVIDIA GPUs can accelerate the local placement network. You can choose Auto, CPU, or GPU where available. If a selected runtime isn't available, ArcForge should report that rather than silently pretending it used it.

## Audio analysis: ArcAudio

ArcAudio tries to separate **what can be heard** from **what deserves mapping attention**. Its analysis can feed musical events, source clues, section context, and phrase information into the mapper before notes are placed.

Available modes:

- **Off / Legacy** — uses the existing candidate pipeline.
- **ArcAudio 2.0** — uses mixed-audio DSP hypotheses and contextual attention.
- **Hybrid** — combines compatible legacy rhythm evidence with the attention system.
- **ArcAudio 3.0** — adds pretrained YAMNet sound classification alongside observed attack timing, attention, and compatible legacy evidence.

You can adjust musical priorities, source competition, smoothing, density ceiling, and secondary accents. The audio inspector lets you check the waveform, beat grid, source lanes, accepted and rejected events, context, attention, density, confidence, and phrase boundaries.

The audio-analysis engine and note-placement engine are separate settings. ArcForge reports which engine was requested and which one was actually used. If an engine can't run, it should explain the fallback or error.

A quick note about ArcAudio 3.0: YAMNet is a 521-class sound classifier. It is **not** a stem separator, lyric transcriber, or piano-note recognizer. Its classification windows don't directly become note timestamps, and it won't identify every instrument perfectly. Dense songs can still be difficult to analyze, so check the grid and listen to the result instead of assuming detection is perfect.

## Edit maps yourself

The Studio editor lets you place and edit notes, bombs, walls, arcs, chains, BPM markers, and tempo ramps. The waveform is displayed beside the beat grid, and you can show or hide the Beat grid, Light grid, and Mods panels independently.

Hover over a grid cell to see a placement ghost, then click to add a note. In player orientation, red is the left hand and blue is the right hand.

### Default keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| Q W E / A S D / Z X C | Up-left, up, up-right / left, dot, right / down-left, down, down-right |
| F | Invert cut direction |
| Tab | Switch red/blue hand |
| Mouse wheel | Move forward/backward in song time |
| Ctrl + mouse wheel | Zoom |
| Left / right arrow | Step using the beat snap |
| Space | Play / pause |
| Escape / Delete | Clear / remove selection |
| Ctrl + Z | Undo |
| Ctrl + Y or Ctrl + Shift + Z | Redo |
| Ctrl + S | Save |

**Check current draft** checks for things like repeated same-hand directions, insufficient recovery, geometry, visibility, and possible collisions. These checks are advisory, not a perfect human-comfort detector, and they don't change or save the map.

### Let AI Continue

Select a gap and use **Let AI Continue** to fill it based on the surrounding rhythm, density, movement, and hand patterns on both sides. You can fill gaps while keeping existing content, or explicitly choose Replace and Continue.

You can review a proposal before accepting it, regenerate it, or discard it. Existing lighting and other authored objects are preserved by default, and a stale preview shouldn't overwrite newer edits.

## Lighting, mods, and previews

ArcForge can create and edit lighting as well as gameplay. Its lighting generation uses learned basic-event behavior, balanced red/blue banks, accent rotations, laser movement, fades, brightness, and boosts. Gameplay and lighting share tempo and section planning, so lighting styles can change the way a section is articulated instead of only changing a label.

The **Mods** tools let you inspect and edit supported modded data, including transforms, tracks, keyframes, environment geometry, asset metadata, and JSON. AI-generated modding content is intended to help create more than just vanilla note maps. The exact result depends on the mod, the data generated, and the preview/runtime support.

| Mod / format | What ArcForge supports | Important limitation |
| --- | --- | --- |
| **Chroma** | Colors, IDs, native lights, rings/lasers, environment and geometry data | The preview doesn't reproduce every Unity environment effect |
| **Noodle Extensions** | Object and track keyframes, positions, rotations, scale, dissolve, and time data | The browser preview approximates supported properties; test complex choreography in-game |
| **Mapping Extensions** | Precision coordinates and custom directions | Some combinations are incompatible with Noodle Extensions |
| **Vivify** | Imported bundle metadata/assets, prefab lifecycle, and custom-event data | Compiled meshes, shaders, and post-processing need Beat Saber to render properly |
| **CustomJSONData / Heck** | Editable custom data and framework dependencies | These are frameworks, not standalone visual effects |
| **Synapse** | Recognition of the external network plugin | It isn't a map visual renderer |

The browser Mod preview updates supported transforms, and Legacy 3D is available as another preview option. ArcViewer is a separate Unity WebGL player and can preview an unsaved ZIP snapshot when installed. It supports 13 environments plus the Crab Rave alias; unsupported scenes fall back to Default in ArcViewer, although all 45 catalogued environment names remain available for export.

For scripts in Mods, use **Run → Review → Apply** and keep the undo option in mind. These are local project tools, not a general Unity plugin host.

Export targets include **Vanilla**, **Ranked-compatible**, **Modded**, and **Unrestricted**. Ranked-compatible only limits certain features; it doesn't mean a map is guaranteed to be accepted or ranked.

## V3 and V4 support

ArcForge can import V2, V3, and V4 maps. In **Settings → Native export format**, choose V3 (the default) or native V4 export. V4 uses indexed gameplay data with separate lightshows and shared AudioData timing.

Imported metadata is retained where supported. If a downgrade can't safely preserve a V4-only feature, ArcForge should report that instead of silently pretending nothing changed. Some V4.1 features, including NJS or per-object rotation behavior, still depend on the game or viewer runtime.

Advanced legacy V2 mod charts can be viewed and exported unchanged, but convert them to V3 in ChroMapper before editing their animations.

## Privacy and local files

ArcForge is designed to run locally:

- Music, projects, and training data stay on your computer.
- You don't need an API key.
- Playlist import downloads map archives through the official API.
- Optional model components and dependencies may need an internet connection.
- The public source archive does not include the private development corpus, songs, projects, trained checkpoints, or model weights.

The app runs on loopback at port `8769`. The local server validates loopback hostnames for reads and writes.

## Development and release checks

The backend and engine code are MIT licensed. The ChroMapper-derived frontend is GPL-2.0-only, and ArcViewer is a separate GPL-3.0 program. Each component keeps its own license and attribution; your music and maps aren't relicensed by ArcForge.

See [third-party notices](THIRD_PARTY_NOTICES.md), [frontend dependency licenses](docs/FRONTEND_LICENSES.md), [provenance](integrations/PROVENANCE.md), and the [source manifest](integrations/source-manifest.json).

Useful release checks:

```powershell
python -X utf8 tools/verify_release.py
npm run build
python -X utf8 tools/package_source.py --audit-only
python -X utf8 tools/package_source.py
python -X utf8 tools/verify_release.py --archive release/ArcForge-1.3.0-source.zip --archive-only
```

The repository also includes release and evaluation notes:

- [Release 1.3 scope and limitations](docs/RELEASE_1_3.md)
- [Generation quality audit](docs/GENERATION_QUALITY_AUDIT.md)
- [Audio evaluation](docs/AUDIO_EVALUATION.md)
- [ArcAudio architecture](ARCAUDIO.md)
- [Training and runtime boundaries](integrations/TRAINING_RUNTIME.md)
- [Native formats and preservation rules](docs/NATIVE_FORMATS.md)
- [Viewer setup](integrations/VIEWER_SETUP.md)

Static checks and numerical model results are useful, but they can't replace listening to maps or testing them in VR.

---

ArcForge is still being developed, so expect things to change. The whole point is to keep improving it, test how much human-map training actually helps, and make AI-generated Beat Saber maps that don't instantly scream "AI-generated."
