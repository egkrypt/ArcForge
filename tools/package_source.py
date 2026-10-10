"""Deterministic, explicit-whitelist source release. Never includes runtime data.

Run from any directory with Python 3.10+: package_source.py [--audit-only].
All output is contained in the app's release directory. No network is used.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION = '1.3.0'
TOP = f'ArcForge-{VERSION}'
DEST = ROOT / 'release'
STAMP = (2026, 10, 10, 0, 0, 0)
ROOT_FILES = {
    '.gitignore', 'README.md', 'ARCAUDIO.md', 'ARCAUDIO_V2_CHECKLIST.md',
    'UPDATE_1_1_CHECKLIST.md', 'UPDATE_1_3_CHECKLIST.md', 'THIRD_PARTY_NOTICES.md', 'LICENSE',
    'server.py', 'bootstrap.py', 'audio.py', 'attention_api.py', 'services_api.py',
    'projects.py', 'studio.py', 'modding.py', 'gameplay.py', 'playlists.py',
    'training_runtime.py', 'service_restart.py', 'Setup ArcForge.ps1',
    'Launch ArcForge.ps1', 'Launch ArcForge.cmd', 'Stop ArcForge.ps1', 'Stop ArcForge.cmd',
    'requirements-server.txt', 'requirements-gpu.txt', 'requirements-recognition.txt',
    'requirements-inferno.txt',
}
FRONTEND_FILES = {'package.json', 'package-lock.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml',
                  'tsconfig.json', 'vite.config.ts', 'index.html', 'LICENSE'}
INTEGRATION_FILES = {'PROVENANCE.md', 'TRAINING_RUNTIME.md', 'VIEWER_SETUP.md',
                     'changelog.json', 'environments.json', 'source-manifest.json',
                     'runtime-versions.json', 'arcviewer-upstream-index.html', 'prepare_viewer.py'}
TEST_FILES = {
    'test_arcaudio.py', 'test_attention_api.py', 'test_regions_continuation.py',
    'test_recognition_import.py', 'test_music_mapping.py', 'test_meter_stereo.py',
    'test_composition.py', 'test_whole_map_variety.py', 'test_services_api.py',
    'test_mod_choreography13.py', 'test_generation_attention_regions.py',
    'test_continuation_workflow13.py', 'test_training_runtime.py',
    'test_chromapper.mjs', 'test_mod_runtime.mjs', 'validate_exports.mjs',
    'browser_13_ux.mjs', 'browser_13_editor.mjs', 'browser_13_live.mjs', 'evaluate_arcaudio.py', 'benchmark_import13.py',
    'evaluate_real_generation13.py', 'test_attention_dsp_quality.py',
    'test_tempo_timeline.py', 'test_engine_selection_api.py', 'test_mapping_quality.py',
    'test_lighting_shared.py', 'test_formats_v4.py', 'evaluate_audio_attention_quality.py',
    'evaluate_engine_modes.py', 'evaluate_generation_api.py', 'quality_benchmark.py',
    'validate_real_v4.py', 'browser_engine_selection.mjs',
    'test_source_release.py',
    'test_dynamic_jump.py',
    'test_native_preview.mjs', 'browser_native_preview.mjs', 'test_launchers.ps1',
    'test_attention_admission.py',
    'test_local_access.py',
    'test_attention_accents.py',
    'test_validator_portability.py',
    'test_attention_viability.py', 'test_inferno_runtime.py', 'test_recognition_components.py',
    'browser_attention_starvation.mjs',
    'test_analysis_runtime.py', 'test_analysis_compute_api.py',
    'test_mapping_source_regions.py', 'test_mapping_region_safety.py', 'test_region_controls_api.py',
    'benchmark_analysis_compute.py', 'evaluate_mapping_source_regions.py',
}
DOC_FILES = {'AUDIO_EVALUATION.md', 'GENERATION_QUALITY_AUDIT.md', 'NATIVE_FORMATS.md', 'RELEASE_1_3.md', 'FRONTEND_LICENSES.md'}
DENY_PARTS = {'data', 'models', 'node_modules', '__pycache__', '.git', '.venv', '.deps',
              '.gpu-deps', '.server-deps', '.install-temp', '.test-deps', 'Library', 'Temp', 'Logs'}
DENY_SUFFIXES = {'.pyc', '.pyo', '.h5', '.npz', '.pkl', '.log', '.pid', '.tsbuildinfo', '.zip',
                 '.mp3', '.ogg', '.egg', '.flac', '.vivify', '.m4a', '.aac', '.opus', '.wma',
                 '.aif', '.aiff', '.mp4', '.mkv', '.npy', '.pt', '.pth', '.onnx', '.tflite',
                 '.ckpt', '.safetensors', '.pem', '.key', '.pfx', '.env', '.wasm', '.unityweb',
                 '.exe', '.msi', '.dll', '.so', '.dylib', '.a', '.lib', '.pdb',
                 '.data', '.bundle', '.dat', '.bak', '.sqlite', '.db'}


def admitted(rel: pathlib.PurePosixPath) -> bool:
    parts = rel.parts
    if any(p in DENY_PARTS or p.lower().startswith('.env') for p in parts) or rel.suffix.lower() in DENY_SUFFIXES:
        return False
    # The only distributed audio is the retained upstream viewer's short SFX,
    # which form part of its corresponding source. No user audio is admitted.
    if rel.suffix.lower()=='.wav' and not (parts[:3]==('vendor','arcviewer','Assets') and 'SFX' in parts):
        return False
    if len(parts) == 1:
        return parts[0] in ROOT_FILES
    if parts[0] == 'engine':
        return rel.suffix in {'.py', '.txt'}
    if parts[0] == 'frontend':
        return ((len(parts) == 2 and parts[1] in FRONTEND_FILES) or parts[1]=='src'
                or parts[1:] == ('public','arcforge-logo.png')
                or parts[1]=='dist' and rel.suffix.lower() in {'.html','.js','.css','.png','.svg','.ico','.woff','.woff2','.ttf','.otf'})
    if parts[0] == 'tools':
        return len(parts) == 2 and parts[1] in {'package_source.py', 'verify_release.py', 'python_runtime.ps1'}
    if parts[0] == 'tests':
        return len(parts) == 2 and parts[1] in TEST_FILES
    if parts[0] == 'docs':
        return len(parts)==2 and parts[1] in DOC_FILES
    if parts[0] == '.github':
        return parts[1:] == ('workflows', 'ci.yml')
    if parts[0] == 'integrations':
        return (len(parts) == 2 and parts[1] in INTEGRATION_FILES) or parts[1:3] == ('beatsaver','SongCore-master')
    if parts[0] == 'vendor':
        if parts[1] in {'arcviewer', 'inferno'}:
            # These are the retained upstream source snapshots, not a Unity WebGL build.
            return not any(p in {'Build', 'Builds', 'build'} for p in parts[2:])
        if parts[1] == 'chromapper':
            # Complete ArcForge frontend source is above; retain exact reference C# files.
            return str(rel) in {
                'vendor/chromapper/LICENSE', 'vendor/chromapper/README.md',
                'vendor/chromapper/Assets/__Scripts/Beatmap/Helper/SpawnParameterHelper.cs',
                'vendor/chromapper/Assets/__Scripts/MapEditor/UI/MirrorSelection.cs',
                'vendor/chromapper/Assets/__Scripts/UI/SongInfoEditUI.cs',
            }
    return False


def selected_files() -> list[pathlib.Path]:
    result = []
    roots = {'engine', 'frontend', 'tools', 'tests', 'docs', '.github', 'integrations', 'vendor'}
    for folder, directories, names in os.walk(ROOT):
        current = pathlib.Path(folder)
        directories[:] = sorted(d for d in directories if d not in DENY_PARTS
            and (current != ROOT or d in roots)
            and not (current == ROOT / 'frontend' and d not in {'src', 'public', 'dist'})
            and not (current == ROOT / 'integrations' and d != 'beatsaver')
            and not (current == ROOT / 'vendor' and d not in {'arcviewer', 'chromapper', 'inferno'}))
        for name in names:
            path = current / name
            rel = pathlib.PurePosixPath(path.relative_to(ROOT).as_posix())
            if admitted(rel):
                if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError(f'Release inputs cannot escape the app: {rel}')
                result.append(path)
    return sorted(result, key=lambda p: p.relative_to(ROOT).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    files = selected_files()
    required = [ROOT / 'server.py', ROOT / 'frontend/package-lock.json', ROOT / 'frontend/dist/index.html',
                ROOT / 'frontend/LICENSE', ROOT / 'vendor/arcviewer/LICENSE',
                ROOT / 'vendor/chromapper/LICENSE', ROOT / 'tools/verify_release.py', ROOT/'tools/python_runtime.ps1',
                *(ROOT/'tests'/name for name in TEST_FILES), *(ROOT/'docs'/name for name in DOC_FILES)]
    missing = [str(p.relative_to(ROOT)) for p in required if p not in files]
    if missing:
        raise SystemExit('Missing required release inputs: ' + ', '.join(missing))
    manifest = {'version': VERSION, 'date': '2026-10-10', 'format': 'explicit-whitelist-source',
        'privateRuntimeIncluded': False, 'hostedArcViewerBuildIncluded': False,
        'files': [{'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    DEST.mkdir(exist_ok=True)
    manifest_path = DEST / f'ArcForge-{VERSION}-manifest.json'
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    if args.audit_only:
        (DEST / 'source-audit.json').write_bytes(manifest_bytes)
        print(json.dumps({'audit': 'passed', 'files': len(files), 'bytes': sum(f['bytes'] for f in manifest['files']),
                          'privateRuntimeIncluded': False, 'hostedArcViewerBuildIncluded': False}))
        return
    archive_path = DEST / f'ArcForge-{VERSION}-source.zip'
    temporary = archive_path.with_suffix('.zip.tmp')
    def add(z, name, payload):
        info = zipfile.ZipInfo(f'{TOP}/{name}', STAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        info.create_system = 3
        z.writestr(info, payload, compresslevel=9)
    with zipfile.ZipFile(temporary, 'w') as z:
        for path, expected in zip(files, manifest['files']):
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected['sha256']:
                raise RuntimeError(f'Release input changed during packaging: {expected["path"]}')
            add(z, expected['path'], payload)
        add(z, 'SOURCE_MANIFEST.json', manifest_bytes)
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:
            raise RuntimeError('ZIP CRC verification failed')
    temporary.replace(archive_path)
    manifest_path.write_bytes(manifest_bytes)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (DEST / f'ArcForge-{VERSION}-source.sha256').write_text(f'{digest}  {archive_path.name}\n', encoding='utf-8')
    print(json.dumps({'archive': archive_path.name, 'sha256': digest, 'files': len(files),
                      'compressedBytes': archive_path.stat().st_size, 'privateRuntimeIncluded': False}))


if __name__ == '__main__':
    main()
