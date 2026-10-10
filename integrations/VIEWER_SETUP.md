# ArcViewer installation and source-release boundary

The existing local installation retains its ArcViewer 0.8.1 WebGL player. It is served as a separate iframe, not linked into the GPL-2.0 frontend. The app can export an unsaved native ZIP snapshot for that player without saving the project. An optional idle refresh creates a new snapshot; Unity reload is more expensive than the live browser preview.

The public source ZIP **does not contain the hosted player's WASM, Unity data, framework or loader**. A product version of 0.8.1 does not establish which exact source commit and build inputs produced those binary bytes. The GPL-3.0 source tag under `vendor/arcviewer` is retained as a separate upstream reference, not represented as proven corresponding source for the hosted binary.

That reference snapshot also omits four prebuilt dependency libraries:
`Winista.MimeDetect.dll`, `Ookii.Dialogs.dll`, `System.Windows.Forms.dll` and
`libStandaloneFileBrowser.so`. Their source references, metadata and available
upstream notices remain. Restore the dependencies from their matching upstream
projects before building; their binaries are not supplied by the source ZIP.
Packaging does not remove them from the installed local workspace.

Without an installed player, use **Mod preview** or **Legacy 3D**. ArcForge reports viewer availability through `/api/capabilities` and does not attempt to load its own app recursively as a Unity player.

## Build a separate player

The retained upstream project identifies Unity **6000.0.67f1** in `vendor/arcviewer/ProjectSettings/ProjectVersion.txt`. Review that project's README, license, package references and WebGL build template. A matching Unity installation with WebGL support is required; npm does not compile it.

1. Review and restore the omitted native/.NET plugins from the upstream project's documented dependencies. Open the retained ArcViewer project in its matching Unity version and let Unity restore its declared packages.
2. Build the declared WebGL scenes into a separate directory. Keep the build inputs and modified source together so that you can provide corresponding source if distributing that build.
3. Place the resulting WebGL output beneath `integrations/arcviewer-web`, retaining its `Build` and template paths. The current availability probe expects `Build/ArcViewer.wasm`.
4. The iframe wrapper must accept the upstream `url` and `noProxy=true` query parameters. Keep its same-origin ready/error notifications if adding the ArcForge wrapper. Restart ArcForge and verify a draft ZIP, playback and environment selection.

`integrations/arcviewer-upstream-index.html` and `prepare_viewer.py` document the existing wrapper adaptation. Running the preparation helper does not compile or obtain a Unity build, and should not be treated as an installer.

## Preview support

ArcViewer reconstructs thirteen environment scenes. Crab Rave aliases the Monstercat scene; unsupported environments visibly fall back to Default while retaining their native export names. The catalogue contains 45 export environment names.

The live browser Mod preview supports documented object/track/path transforms, colors, dissolve and common basic lighting. It approximates these properties; it is not the Beat Saber runtime. Vivify compiled prefab meshes, shaders, particles and post-processing require the game. Arbitrary custom-event choreography, modern lighting filter semantics and complete Chroma environment replacement also require game-side inspection.

Primary source: [ArcViewer v0.8.1-beta](https://github.com/AllPoland/ArcViewer/tree/v0.8.1-beta). No new upstream repository was downloaded during the 1.3 research work.
