"""Fixture-only smoke suite, suitable for a source checkout without trained models.

Optional recognition weights are not downloaded. Corpus/model-dependent tests
remain separate from this release suite. Tests only create their own fixtures.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, re, sys, unittest, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests')); sys.path.insert(0,str(ROOT/'tools'))
import bootstrap  # noqa: E402
from package_source import TOP, VERSION, admitted, TEST_FILES, DOC_FILES  # noqa: E402


def verify_archive(path):
    """Audit ZIP entries and every declared hash without extracting or executing."""
    with zipfile.ZipFile(path) as archive:
        entries=archive.infolist();names=[entry.filename for entry in entries]
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP entries')
        if len(names)!=len({name.casefold() for name in names}):raise ValueError('Case-insensitive ZIP path collision')
        manifest_name=f'{TOP}/SOURCE_MANIFEST.json'
        manifest=json.loads(archive.read(manifest_name))
        if (manifest.get('version')!=VERSION or manifest.get('privateRuntimeIncluded') is not False
                or manifest.get('hostedArcViewerBuildIncluded') is not False):
            raise ValueError('Unexpected release identity or private-data declaration')
        expected={f'{TOP}/{item["path"]}':item for item in manifest['files']}
        if len(expected)!=len(manifest['files']):raise ValueError('Duplicate manifest paths')
        if set(names)!={manifest_name,*expected}:raise ValueError('Archive entries do not match its manifest')
        for entry in entries:
            entry_path=pathlib.PurePosixPath(entry.filename)
            if (entry_path.is_absolute() or '..' in entry_path.parts or '\\' in entry.filename
                    or entry_path.as_posix()!=entry.filename
                    or entry_path.parts[0]!=TOP or any(':' in part or part.rstrip(' .')!=part
                    or any(ord(char)<32 for char in part)
                    or re.fullmatch(r'(?i:CON|PRN|AUX|NUL|COM[1-9\u00b9\u00b2\u00b3]|LPT[1-9\u00b9\u00b2\u00b3])',part.split('.')[0])
                    for part in entry_path.parts)):
                raise ValueError(f'Unsafe archive path: {entry.filename}')
            if (entry.external_attr>>16)&0o170000==0o120000:raise ValueError('Symlink in archive')
            if entry.filename==manifest_name:continue
            rel=pathlib.PurePosixPath(*entry_path.parts[1:])
            if not admitted(rel):raise ValueError(f'Non-whitelisted release entry: {rel}')
            payload=archive.read(entry);item=expected[entry.filename]
            if len(payload)!=item['bytes'] or hashlib.sha256(payload).hexdigest()!=item['sha256']:
                raise ValueError(f'Manifest mismatch: {rel}')
        for name in [*(f'tests/{x}' for x in TEST_FILES),*(f'docs/{x}' for x in DOC_FILES)]:
            if f'{TOP}/{name}' not in expected:raise ValueError(f'Missing documented source: {name}')
        if archive.testzip():raise ValueError('ZIP integrity failure')
    report={'archive':str(path),'files':len(expected),'sha256':hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),
            'manifestHashes':'passed','whitelist':'passed','privateRuntimeIncluded':False}
    print(json.dumps(report),flush=True)
    return report

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=pathlib.Path,help='Audit a source ZIP, including manifest hashes and private-data exclusions')
    parser.add_argument('--archive-only',action='store_true',help='Skip fixture tests after archive audit')
    args=parser.parse_args()
    if args.archive:verify_archive(args.archive)
    if args.archive_only:
        if not args.archive:parser.error('--archive-only requires --archive')
        raise SystemExit(0)
    (ROOT / 'data/test-artifacts').mkdir(parents=True, exist_ok=True)
    names = ['test_arcaudio', 'test_regions_continuation', 'test_services_api',
             'test_mod_choreography13', 'test_recognition_import.ImportCache',
             'test_attention_dsp_quality', 'test_tempo_timeline', 'test_engine_selection_api',
             'test_lighting_shared', 'test_formats_v4', 'test_meter_stereo', 'test_composition',
             'test_source_release', 'test_dynamic_jump', 'test_attention_admission', 'test_local_access',
             'test_attention_accents', 'test_validator_portability',
             'test_attention_viability', 'test_inferno_runtime', 'test_recognition_components',
             'test_analysis_runtime', 'test_analysis_compute_api', 'test_mapping_source_regions',
             'test_mapping_region_safety', 'test_region_controls_api']
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
