# Source integrations

The development workspace retains the upstream source releases requested earlier, with hashes in `source-manifest.json`. The public archive includes retained ArcViewer/Inferno source snapshots, complete ArcForge frontend source, the original adapted ChroMapper reference files and licenses; it is not a binary distribution of the full Unity editors.

## ArcViewer v0.8.1-beta

- Upstream: https://github.com/AllPoland/ArcViewer/tree/v0.8.1-beta
- License: GPL-3.0, complete text at `vendor/arcviewer/LICENSE`.
- Unity version: 6000.0.67f1.
- The local player under `integrations/arcviewer-web/` is the public upstream WebGL player whose loader declares productVersion 0.8.1. It was downloaded from https://allpoland.github.io/ArcViewer/ on 2026-10-02, not rebuilt or claimed to be a byte-for-byte tagged build. Source and asset hashes are recorded separately.
- Only the HTML wrapper was modified: same-origin ready/error notifications and a readable startup failure message. Unity data, WASM, loader and framework remain upstream bytes.
- The player is served at `/viewer/`, in a separate iframe. The app sends a same-origin exported ZIP using the source-supported `url` and `noProxy=true` query parameters. Playback, environment selection, scrubbing and settings inside this player are its own controls. Refreshing the player can load an unsaved native draft ZIP snapshot without saving the project. Optional idle refresh is available.
- Source facts: `Assets/__Scripts/Loading/Web/URLParsing/UrlArgHandler.cs` (URL loader), `Assets/__Scripts/Previewer/MapControl/Environments/EnvironmentManager.cs` (environment list and fallbacks), `ProjectSettings/EditorBuildSettings.asset` (13 built scenes).
- ArcViewer includes 13 reconstructed environment scenes; Crab Rave uses the Monstercat scene. Other known environments export their correct names but preview in ArcViewer's Default scene. They are clearly marked in the selector. The native Studio viewer uses original procedural scenery rather than proprietary game models.
- Supported upstream: standard notes, bombs, walls, arcs, chains, BPM changes, basic/V2 lighting, Chroma colors/light IDs/ring and laser events, and precision mod positioning. Advanced Noodle track choreography, Chroma environment enhancement and Vivify asset bundles are **not fully reproduced** by upstream ArcViewer.

## ChroMapper 0.13.892

- Upstream: https://github.com/Caeden117/ChroMapper/tree/0.13.892
- License: GPL-2.0, complete text at `vendor/chromapper/LICENSE`.
- Unity version: 2021.3.45f1. The full Unity editor has not been compiled or embedded; the editor in ArcForge is a browser implementation with explicitly adapted source algorithms.
- `frontend/src/chromapper.ts` ports `Assets/__Scripts/Beatmap/Helper/SpawnParameterHelper.cs`: half-jump duration and jump distance, preserving the 17.999 m halving threshold and 0.25 beat minimum.
- The same module ports the note-only portion of `Assets/__Scripts/MapEditor/UI/MirrorSelection.cs`: grid/precision-position mirroring, color swap, horizontal direction map, angle offset and mod rotation reversal. Slider/event/obstacle operations are not claimed as ports.
- New input validation and the inverse jump-distance helper were added. BPM integration is original ArcForge code, separately identified in the source file.
- `tests/test_chromapper.mjs` checks known jump values, inverse settings and mirror involution including mod coordinates.

## InfernoSaber v1.7.1 and local training

- Upstream: https://github.com/fred-brenner/InfernoSaber---BeatSaber-Automapper/tree/v1.7.1
- License: MIT, retained with the complete requested source under `vendor/inferno/InfernoSaber---BeatSaber-Automapper-1.7.1`.
- `engine/inferno_adapter.py` invokes actual upstream preprocessing, timing and decoding functions in a separate TensorFlow CPU worker. The upstream `fav_15` encoder, mapper, beat generator and decoder files have fixed SHA-256 checks. FFmpeg decoding and legacy Keras compatibility are identified adaptations; the upstream neural weights remain unchanged.
- ArcForge's independently implemented placement trainer learns from the uploaded human-chart corpus. The optional PyTorch CUDA backend trains this local network, not Inferno's TensorFlow weights. Its observed device, versions, held-out split, losses and checkpoint history are recorded in `models/style/report.json`.
- Human-chart style conditioning learns density, angle, movement and double statistics without verified style labels. Preset names are target goals, not a claim that the corpus has been authoritatively classified by mapping style.
- BeatSaver playlist verification follows the published map-hashing specification examined in SongCore's MIT-licensed source. That source and license are retained under `integrations/beatsaver/SongCore-master/`. Implementation details, official references and runtime boundaries are documented in [TRAINING_RUNTIME.md](TRAINING_RUNTIME.md).

## License boundaries

The ChroMapper-derived frontend is GPL-2.0-only; modifications and complete source are provided locally. ArcViewer is separately served GPL-3.0 software with a retained source tag and license; exact corresponding source for the installed hosted binary has not been established, communicating by a URL-loaded map file. ArcViewer C# source is not linked or translated into the GPL-2.0 frontend. Backend and other standalone components retain their declared licenses. This directory is an aggregate of separate programs, not a relicensing of upstream projects.

Training maps and uploaded audio remain local user content. Source attribution is not a grant to redistribute third-party music, Beat Saber assets, or other maps.


## ArcAudio 3.0 and 1.3 research

The new recognition runtime follows [Google YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet), Apache-2.0. Architecture/log-mel execution is independently expressed in PyTorch and compared numerically with TensorFlow. The optional installer downloads only official unchanged weights, 15,296,092 bytes, SHA-256 `13c3308955bbfaef262f175ac9c40e47b134573a93984f009220dd7cc12a1744`. Source classes indicate sound presence, not separated stems or exact piano/syllable events. The public archive contains source/license, not weights.

These requested references were read during the 1.3 work without downloading new repository archives:

- [audio-seg-data-synth](https://github.com/satvik-venkatesh/audio-seg-data-synth): temporal speech/music segmentation and augmentation ideas. Its pretrained assets were not obtained or incorporated; code and model terms are distinct.
- [soundview](https://github.com/pirate/soundview): spectrum, pitch/harmonic and chroma inspection concepts; not used as a pretrained instrument recognizer.
- [instrument_recognition_AST](https://github.com/gilpanal/instrument_recognition_AST): multilabel AudioSet/instrument classification and evaluation boundaries. Its large AST/Essentia pretrained assets were not downloaded or incorporated.
- [audio-classification](https://github.com/apasagic/audio-classification): separation between feature extraction, labels, classification and evaluation; no project asset copied.
- [audio-recognition topics](https://github.com/topics/audio-recognition) and [sound-detection topics](https://github.com/topics/sound-detection?o=desc&s=updated): discovery references, not licensed model assets.
- [Chroma](https://github.com/Aeroluna/Chroma), [Noodle Extensions](https://github.com/Aeroluna/NoodleExtensions), [Heck](https://github.com/Aeroluna/Heck) and [current Heck documentation](https://heck.aeroluna.dev/): documented custom-event, track/path, lighting and animation properties. Old archived repositories are not represented as current binaries.
- [Mapping Extensions](https://github.com/Kylemc1413/MappingExtensions): precision-coordinate and direction encodings.
- [Vivify](https://github.com/Aeroluna/Vivify): bundle metadata and prefab lifecycle. Unity shaders, post-processing and compiled meshes are not claimed to render generically in the browser.
- [CustomJSONData](https://github.com/Aeroluna/CustomJSONData): custom-data framework boundary.
- [Synapse](https://github.com/Aeroluna/Synapse): network/synchronized-play plugin; it is not a map visual-rendering system.

Browser mod editing/preview is original ArcForge TypeScript that interprets supported documented JSON properties; the game plugins are not installed or embedded into the browser. Exact runtime behavior for unsupported properties remains a Beat Saber test boundary. See [VIEWER_SETUP.md](VIEWER_SETUP.md) and the in-app capability matrix.

## Public release policy, 2026-10-07

`tools/package_source.py` selects explicit source/build/license paths and writes a per-file SHA-256 manifest. It excludes runtime data, models/weights, dependency directories, imported assets, user projects, logs, backups and historical patch scripts. Complete ArcForge frontend source accompanies its built JS. The original ChroMapper source references and GPL-2.0 license are retained. The ArcViewer tag is supplied as a separate reference project; opaque hosted WebGL binary files are omitted because exact corresponding build source is unproven. The installed local viewer is left intact.
