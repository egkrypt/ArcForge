"""Release whitelist must exclude private runtime and include audited source."""
import pathlib,sys,unittest,contextlib,io,hashlib,json,tempfile,zipfile
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from package_source import admitted,selected_files,TEST_FILES,DOC_FILES
import verify_release


class SourceRelease(unittest.TestCase):
    def test_manifest_hashes_are_verified_without_extracting(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as temporary:
            path=pathlib.Path(temporary)/'source.zip';body=b'Example source'
            manifest={'version':'1.3.0','privateRuntimeIncluded':False,'hostedArcViewerBuildIncluded':False,'files':[{
                'path':'README.md','bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}]}
            def archive(payload):
                with zipfile.ZipFile(path,'w') as target:
                    target.writestr(verify_release.TOP+'/SOURCE_MANIFEST.json',json.dumps(manifest))
                    target.writestr(verify_release.TOP+'/README.md',payload)
            with patch.object(verify_release,'TEST_FILES',set()),patch.object(verify_release,'DOC_FILES',set()):
                archive(body)
                with contextlib.redirect_stdout(io.StringIO()):report=verify_release.verify_archive(path)
                self.assertEqual(report['archive'],str(path));self.assertEqual(report['files'],1)
                archive(b'Corrupted source')
                with self.assertRaisesRegex(ValueError,'Manifest mismatch'):verify_release.verify_archive(path)

    def test_unsafe_paths_and_opaque_runtime_declarations_fail(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as temporary:
            path=pathlib.Path(temporary)/'source.zip';body=b'Example source'
            for name in ['../README.md','frontend/src/stream:payload.ts','frontend/src/trailing.ts ',
                         'frontend/src/nested\\file.ts','frontend/src/control\x01.ts',
                         'frontend/src/CON.ts','frontend/src/lpt1.ts','frontend/src/aux',
                         'frontend/src//extra.ts','frontend/src/./extra.ts']:
                with self.subTest(path=name):
                    manifest={'version':'1.3.0','privateRuntimeIncluded':False,'hostedArcViewerBuildIncluded':False,
                        'files':[{'path':name,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}]}
                    with zipfile.ZipFile(path,'w') as target:
                        target.writestr(verify_release.TOP+'/SOURCE_MANIFEST.json',json.dumps(manifest))
                        target.writestr(verify_release.TOP+'/'+name,body)
                    with self.assertRaisesRegex(ValueError,'Unsafe archive path|Archive entries do not match'):
                        verify_release.verify_archive(path)
            manifest['hostedArcViewerBuildIncluded']=True
            with zipfile.ZipFile(path,'w') as target:
                target.writestr(verify_release.TOP+'/SOURCE_MANIFEST.json',json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Unexpected release identity'):
                verify_release.verify_archive(path)

    def test_case_collisions_fail_before_extraction(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as temporary:
            path=pathlib.Path(temporary)/'source.zip';body=b'Example source'
            files=[{'path':name,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
                   for name in ['frontend/src/Example.ts','frontend/src/example.ts']]
            manifest={'version':'1.3.0','privateRuntimeIncluded':False,'hostedArcViewerBuildIncluded':False,'files':files}
            with zipfile.ZipFile(path,'w') as target:
                target.writestr(verify_release.TOP+'/SOURCE_MANIFEST.json',json.dumps(manifest))
                for item in files:target.writestr(verify_release.TOP+'/'+item['path'],body)
            with self.assertRaisesRegex(ValueError,'Case-insensitive ZIP path collision'):
                verify_release.verify_archive(path)

    def test_private_and_opaque_runtime_are_excluded(self):
        for name in ['data/audio/example/song.ogg','models/style/model.npz','frontend/node_modules/example/a.js',
                     'frontend/public/private.mp3','frontend/public/private.wav','frontend/public/.env',
                     'frontend/public/secret.key','frontend/src/model.safetensors',
                     'frontend/public/private-project.json','frontend/public/ArcViewer.wasm',
                     'integrations/arcviewer-web/Build/ArcViewer.wasm','release/previous.zip',
                     'vendor/arcviewer/Build/ArcViewer.data',
                     'vendor/arcviewer/Assets/Plugins/MimeDetect/Winista.MimeDetect.dll',
                     'vendor/arcviewer/Assets/StandaloneFileBrowser/Plugins/Linux/x86_64/libStandaloneFileBrowser.so',
                     'tests/tmp/incoming.zip']:
            with self.subTest(path=name):self.assertFalse(admitted(pathlib.PurePosixPath(name)))

    def test_new_architecture_tests_and_docs_are_packaged(self):
        selected={path.relative_to(ROOT).as_posix() for path in selected_files()}
        for name in TEST_FILES:self.assertIn('tests/'+name,selected)
        for name in DOC_FILES:self.assertIn('docs/'+name,selected)
        self.assertIn('engine/formats/v4.py',selected)
        self.assertIn('engine/tempo.py',selected)
        self.assertIn('engine/lighting_plan.py',selected)


if __name__=='__main__':unittest.main()
