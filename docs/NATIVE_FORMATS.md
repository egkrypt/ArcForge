# Native beatmap formats

ArcForge keeps one editable, V3-shaped representation and serializes it through
`engine/formats`. `config.beatmapVersion` controls the real exported format.
It is separate from the placement model and from the ArcAudio engine.

| Choice | Files written |
| --- | --- |
| `3` (default) | Info 2.1 plus combined Beatmap 3.3 gameplay and lighting |
| `4` | Info 4.0.1, shared AudioData 4.0, Beatmap 4.1, separate Lightshow 4.0 per difficulty |

`GET /api/projects/{id}/export?version=3` or `?version=4` overrides the saved
format for that download. It does not change or save the chart.

V4 is an indexed format, not a renamed V3 file. Gameplay references resolve
into color-note, bomb, obstacle, arc, chain, rotation and NJS metadata tables.
Lighting references resolve into separate basic events, boosts, waypoints,
index filters, color boxes, rotation boxes, translation boxes and float FX.
Repeated equal metadata is deduplicated on export. Explicit invalid indices,
nonfinite values, reversed durations and zero-squish chains fail validation.
Native omitted-value defaults and bounded gzip metadata are supported.

## Tempo and loudness

V4 stores a song-wide sample-to-beat timeline in AudioData. Charts in one map
must agree on tempo; export refuses conflicting timelines. ArcForge converts
its existing BPM markers and song offset into contiguous sample segments and
recovers them on import. Gradual tempo can be represented by successive native
segments. No `bpmEvents` array is invented inside the V4 difficulty file.

For newly exported V4 songs, integrated LUFS is measured by FFmpeg's EBU R128
filter and cached by audio revision. Imported native loudness metadata is
preserved when its original clock remains applicable. This is integrated
song loudness, not a claim of per-instrument or per-frame loudness analysis.

## Editing and preservation

V4 gameplay and native advanced lighting become editable through the existing
note and light grids. The internal representation additionally retains:

- `r` for per-object lane rotation; `hr`/`tr` for arc and chain endpoints.
- `njsEvents` containing native V4.1 `{b,d,p,e}` events.
- V4 color-event easing and independent occurrence/metadata custom data.
- Independent V4 arc endpoint color/angle metadata. Standard V3 `tc` remains
  the tail cut direction; it is never interpreted as a saber color.

These retained values do not imply that every browser preview implements the
game's V4 runtime. An older ArcViewer build may need a compatible V3 preview;
V4-only behavior must be disclosed instead of silently removed.

An unchanged imported V4 chart exports its original gameplay, lightshow and
AudioData JSON, retaining custom metadata, author lists and separate preview
audio. Known objects can be reserialized after editing. Unknown extension
fields are identified; an edited conversion refuses them rather than losing
their meaning. Native V4.1 NJS, per-object lane rotations and V4-only lighting
metadata likewise refuse an unsupported V3 downgrade.

V2 mod animations retain the existing unchanged-export safeguard. V4 support
does not certify a mod loader, older game build or ranking service supports a
particular feature combination. V3 remains the conservative default for older
mod targets.

## Developer API and verification

`decode_v4(difficulty, lightshow, audio_data)` resolves native data into the
editing representation. `encode_v4(chart)` returns separate `difficulty` and
`lightshow` documents. `validate_v4` checks metadata references and native
object ranges. `build_audio_data` and `audio_data_to_tempo` translate the shared
clock. `ensure_v3_compatible` protects against an unsupported downgrade.

`tests/test_formats_v4.py` covers native objects, all four light-box types,
metadata deduplication, NJS/rotation/easing preservation, compressed metadata,
default values, invalid references, tempo changes, loudness measurement,
actual project import/export and V4 training input. The schema validator in
`tests/validate_exports.mjs` supports both V3 and V4 archives. Set
`ARCFORGE_BSMAP_ROOT` to an installed bsmap package before running it against
an extracted test map.

Format references: [BSMG overview](https://bsmg.wiki/mapping/map-format.html),
[Info](https://bsmg.wiki/mapping/map-format/info.html),
[AudioData](https://bsmg.wiki/mapping/map-format/audio.html),
[Beatmap](https://bsmg.wiki/mapping/map-format/beatmap.html),
[Lightshow](https://bsmg.wiki/mapping/map-format/lightshow.html).
