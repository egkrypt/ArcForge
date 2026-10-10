"""Native V4 fixtures, index safety, advanced lighting and sample-clock roundtrips."""
import copy
import asyncio
import gzip
import hashlib
import json
import pathlib
import sys
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bootstrap
from engine.formats import (FormatError, decode_v4, encode_v4, validate_v4,
    build_audio_data, audio_data_to_tempo, info_v2_to_v4, info_v4_to_v2, ensure_v3_compatible, read_native_json, measure_loudness)
from modding import BeatClock
import projects


def fixture():
    return {'version': '3.3.0', 'bpmEvents': [{'b': 8, 'm': 240}, {'b': 24, 'm': 90}],
        'colorNotes': [{'b': 1, 'x': 1, 'y': 0, 'c': 0, 'd': 1, 'a': 0},
                       {'b': 2, 'x': 1, 'y': 0, 'c': 0, 'd': 1, 'a': 0},
                       {'b': 3, 'x': 2, 'y': 1, 'c': 1, 'd': 4, 'a': 15}],
        'bombNotes': [{'b': 4, 'x': 0, 'y': 2}],
        'obstacles': [{'b': 4, 'x': 0, 'y': 0, 'd': 1, 'w': 1, 'h': 5}],
        'sliders': [{'b': 1, 'tb': 2, 'c': 0, 'x': 1, 'y': 0, 'd': 1,
                     'tx': 1, 'ty': 0, 'tc': 1, 'mu': 1, 'tmu': 1, 'm': 0}],
        'burstSliders': [{'b': 3, 'tb': 3.5, 'c': 1, 'x': 2, 'y': 1, 'd': 4,
                          'tx': 3, 'ty': 0, 'sc': 4, 's': 1}],
        'rotationEvents': [], 'waypoints': [{'b': 5, 'x': 1, 'y': 0, 'd': 1}],
        'basicBeatmapEvents': [{'b': 0, 'et': 0, 'i': 1, 'f': 1},
                              {'b': 2, 'et': 1, 'i': 7, 'f': 1.2, 'customData': {'lightID': [1, 2], 'color': [1, .2, .1, 1]}}],
        'colorBoostBeatmapEvents': [{'b': 2, 'o': True}],
        'lightColorEventBoxGroups': [{'b': 4, 'g': 1, 'e': [{'f': {'f': 1, 'p': 0, 't': 1}, 'w': .25, 'd': 1,
            'r': .5, 't': 1, 'b': 0, 'i': 0, 'e': [{'b': 0, 'c': 0, 'i': 0, 'f': 0, 's': 1}, {'b': .5, 'c': 1, 'i': 1, 'f': 4, 's': .5, 'sb': .2, 'sf': 1}]}]}],
        'lightRotationEventBoxGroups': [{'b': 4, 'g': 2, 'e': [{'f': {'f': 1}, 'w': 1, 'd': 1, 's': 90, 't': 1, 'a': 1, 'r': 0, 'b': 1, 'i': 0,
            'l': [{'b': 0, 'e': 4, 'l': 0, 'o': 1, 'p': 0, 'r': 180}]}]}],
        'lightTranslationEventBoxGroups': [{'b': 4, 'g': 3, 'e': [{'f': {'f': 1}, 'w': 1, 'd': 1, 's': 2, 't': 1, 'a': 0, 'r': 1, 'b': 0, 'i': 0,
            'l': [{'b': 0, 'e': 1, 'p': 0, 't': 2}]}]}],
        'vfxEventBoxGroups': [{'b': 4, 'g': 4, 't': 1, 'e': [{'f': {'f': 1}, 'w': .5, 'd': 1, 's': .2, 't': 1, 'b': 0, 'i': 0, 'l': [0]}]}],
        '_fxEventsCollection': {'_fl': [{'b': 0, 'i': 0, 'p': 0, 'v': .75}], '_il': []},
        'basicEventTypesWithKeywords': {'d': []}, 'useNormalEventsAsCompatibleEvents': False,
        'customData': {'environment': [{'id': 'TrackLaneRing', 'lookupMethod': 'Regex', 'active': True}]}}


class NativeV4Tests(unittest.TestCase):
    def test_arc_tail_direction_is_not_tail_color(self):
        chart=fixture();chart['sliders'][0]['tc']=7
        packed=encode_v4(chart);arc=packed['difficulty']['arcs'][0]
        tail=packed['difficulty']['colorNotesData'][arc['ti']]
        self.assertEqual((tail['c'],tail['d']),(0,7))
        restored=decode_v4(packed['difficulty'],packed['lightshow'])
        self.assertEqual(restored['sliders'],chart['sliders'])
        tail['c']=1
        restored=decode_v4(packed['difficulty'],packed['lightshow'])
        self.assertEqual(restored['sliders'][0]['_v4TailColor'],1)
        self.assertEqual(restored['sliders'][0]['tc'],7)
        self.assertEqual(encode_v4(restored),packed)
        with self.assertRaisesRegex(FormatError,'V4-only'):ensure_v3_compatible(restored)
        tail['c']=7
        with self.assertRaisesRegex(FormatError,'Color metadata'):validate_v4(packed['difficulty'])

    def test_spawn_rotation_native_t_and_shared_beat_execution(self):
        chart=fixture();chart['rotationEvents']=[{'b':2,'e':0,'r':15},{'b':2,'e':1,'r':30}]
        packed=encode_v4(chart)
        self.assertEqual(packed['difficulty']['spawnRotationsData'],[{'t':0,'r':15},{'t':1,'r':30}])
        self.assertEqual(packed['difficulty']['colorNotes'][1]['r'],15)
        self.assertEqual(packed['difficulty']['colorNotes'][2]['r'],45)
        restored=decode_v4(packed['difficulty'],packed['lightshow'])
        self.assertEqual(restored['rotationEvents'],chart['rotationEvents'])
        self.assertEqual(encode_v4(restored),packed)
        bad=copy.deepcopy(packed);bad['difficulty']['spawnRotationsData'][1]['e']=0
        with self.assertRaisesRegex(FormatError,'Conflicting'):decode_v4(bad['difficulty'])

    def test_red_tail_of_blue_arc_cannot_be_lost_in_v3_downgrade(self):
        chart=fixture();chart['sliders'][0].update(c=1,_v4TailColor=0)
        packed=encode_v4(chart);restored=decode_v4(packed['difficulty'],packed['lightshow'])
        self.assertEqual(restored['sliders'][0]['_v4TailColor'],0)
        self.assertEqual(encode_v4(restored),packed)
        with self.assertRaisesRegex(FormatError,'V4-only'):ensure_v3_compatible(restored)
        chart['sliders'][0]['_v4TailColor']=1
        self.assertIs(ensure_v3_compatible(chart),chart)

    def test_native_defaults_and_unknown_extension_downgrade_guard(self):
        chart=decode_v4({'version':'4.0.0','colorNotes':[{'b':2}]}, {'version':'4.0.0','eventBoxGroups':[{}]})
        self.assertEqual(chart['colorNotes'][0],{'b':2,'x':0,'y':0,'c':0,'d':0,'a':0})
        packed=encode_v4(chart);self.assertEqual(packed['lightshow']['eventBoxGroups'][0],{})
        unknown=decode_v4({'version':'4.1.0','unrecognizedExtension':[{'example':1}]})
        with self.assertRaisesRegex(FormatError,'unknown extension'):encode_v4(unknown)

    def test_loudness_is_measured_and_cached(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as temporary:
            path=pathlib.Path(temporary)/'song.ogg'
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','lavfi','-i',
                'sine=frequency=440:duration=2','-c:a','libvorbis',str(path)],check=True,timeout=15)
            result=measure_loudness(path)
            self.assertGreater(result,-40);self.assertLess(result,-5)
            with patch('engine.formats.v4.subprocess.run',side_effect=AssertionError('Cache should avoid repeat decoding')):
                self.assertEqual(measure_loudness(path),result)

    def test_compressed_native_metadata_and_expansion_limit(self):
        compressed = gzip.compress(json.dumps({'version': '4.1.0'}).encode())
        self.assertEqual(read_native_json(compressed), {'version': '4.1.0'})
        with self.assertRaisesRegex(FormatError, 'Expanded'):
            read_native_json(gzip.compress(json.dumps({'data': 'a' * 5000}).encode()), max_bytes=100)

    def test_indexed_dedup_and_native_object_roundtrip(self):
        chart = fixture(); untouched = copy.deepcopy(chart)
        result = encode_v4(chart); difficulty = result['difficulty']
        self.assertEqual(difficulty['version'], '4.1.0')
        self.assertEqual(difficulty['colorNotes'][0]['i'], difficulty['colorNotes'][1]['i'])
        self.assertNotIn('bpmEvents', difficulty)
        self.assertNotIn('basicBeatmapEvents', difficulty)
        restored = decode_v4(difficulty, result['lightshow'])
        for key in ('colorNotes', 'bombNotes', 'obstacles', 'burstSliders', 'basicBeatmapEvents', 'colorBoostBeatmapEvents', 'customData'):
            self.assertEqual(restored[key], chart[key])
        self.assertEqual(restored['sliders'][0]['mu'], 1)
        self.assertEqual(restored['sliders'][0]['tmu'], 1)
        self.assertEqual(chart, untouched)
        self.assertEqual(encode_v4(restored), result)

    def test_all_advanced_light_types_and_fx_roundtrip(self):
        chart = fixture(); packed = encode_v4(chart); light = packed['lightshow']
        self.assertEqual({g['t'] for g in light['eventBoxGroups']}, {1, 2, 3, 4})
        restored = decode_v4(packed['difficulty'], light)
        self.assertEqual(encode_v4(restored)['lightshow'], light)
        self.assertEqual(restored['lightRotationEventBoxGroups'][0]['e'][0]['l'][0]['e'], 4)
        self.assertEqual(restored['_fxEventsCollection']['_fl'][0]['v'], .75)

    def test_v4_only_njs_lanes_and_easing_are_preserved_or_refused(self):
        chart = fixture(); chart['njsEvents'] = [{'b': 4, 'd': 3, 'p': 0, 'e': 1}]
        chart['colorNotes'][0]['r'] = 45
        chart['sliders'][0].update(hr=45, tr=90)
        chart['lightColorEventBoxGroups'][0]['e'][0]['e'][0]['_v4Easing'] = 4
        packed = encode_v4(chart); restored = decode_v4(packed['difficulty'], packed['lightshow'])
        self.assertEqual(restored['njsEvents'], chart['njsEvents'])
        self.assertEqual(restored['colorNotes'][0]['r'], 45)
        self.assertEqual(restored['sliders'][0]['tr'], 90)
        self.assertEqual(encode_v4(restored), packed)
        with self.assertRaisesRegex(FormatError, 'NJS'): ensure_v3_compatible(restored)
        ordinary = fixture()
        self.assertIs(ensure_v3_compatible(ordinary), ordinary)

    def test_invalid_indices_fail_instead_of_becoming_zero_notes(self):
        packed = encode_v4(fixture())
        for bad in (-1, 999, .5, True):
            damaged = copy.deepcopy(packed); damaged['difficulty']['colorNotes'][0]['i'] = bad
            with self.assertRaises(FormatError): decode_v4(damaged['difficulty'], damaged['lightshow'])
        damaged = copy.deepcopy(packed); damaged['lightshow']['eventBoxGroups'][0]['e'][0]['l'][0]['i'] = 999
        with self.assertRaises(FormatError): decode_v4(damaged['difficulty'], damaged['lightshow'])
        with self.assertRaises(FormatError): encode_v4({**fixture(), 'burstSliders': [{'b': 0, 'tb': 1, 'sc': 4, 's': 0}]})

    def test_audio_clock_dynamic_tempo_and_offset(self):
        chart = fixture(); bpm = 120; duration = 20; offset = .125
        data = build_audio_data(bpm, duration, chart['bpmEvents'], offset, 48000)
        converted = audio_data_to_tempo(data)
        before = BeatClock(bpm, chart['bpmEvents']); after = BeatClock(converted['bpm'], converted['bpmEvents'])
        self.assertAlmostEqual(converted['songTimeOffset'], offset, places=6)
        for beat in [0, 1, 7.5, 8, 12, 24, 25, 40]:
            self.assertAlmostEqual(before.time(beat) + offset, after.time(beat) + converted['songTimeOffset'], delta=1 / 48000)
        self.assertEqual(data['bpmData'][-1]['ei'], data['songSampleCount'])
        broken = copy.deepcopy(data); broken['bpmData'][1]['si'] += 1
        with self.assertRaises(FormatError): audio_data_to_tempo(broken)

    def test_info_metadata_references_split_files(self):
        info2 = {'_version': '2.1.0', '_songName': 'Native fixture', '_songSubName': '', '_songAuthorName': 'Test',
            '_levelAuthorName': 'Mapper', '_beatsPerMinute': 120, '_songFilename': 'song.ogg',
            '_coverImageFilename': 'cover.png', '_environmentNames': ['DefaultEnvironment'],
            '_difficultyBeatmapSets': [{'_beatmapCharacteristicName': 'Standard', '_difficultyBeatmaps': [{
                '_difficulty': 'ExpertPlus', '_difficultyRank': 9, '_beatmapFilename': 'ExpertPlusStandard.dat',
                '_noteJumpMovementSpeed': 18, '_noteJumpStartBeatOffset': .25,
                '_customData': {'_difficultyLabel': 'Fallen', '_requirements': ['Chroma']}}]}]}
        info4 = info_v2_to_v4(info2, 20)
        self.assertEqual(info4['audio']['audioDataFilename'], 'AudioData.dat')
        diff = info4['difficultyBeatmaps'][0]
        self.assertEqual(diff['lightshowDataFilename'], 'ExpertPlusStandard.lightshow.dat')
        self.assertEqual(diff['customData']['difficultyLabel'], 'Fallen')
        restored = info_v4_to_v2(info4)
        self.assertEqual(restored['_difficultyBeatmapSets'][0]['_difficultyBeatmaps'][0]['_customData']['_difficultyLabel'], 'Fallen')

    def test_project_export_uses_config_and_never_mutates_saved_chart(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as temporary, patch.object(projects, 'DATA', pathlib.Path(temporary)):
            data = pathlib.Path(temporary); audio = data / 'audio' / 'audio-fixture'; audio.mkdir(parents=True)
            (audio / 'song.ogg').write_bytes(b'OggS isolated export fixture')
            project = {'id': 'project-fixture', 'name': 'Native fixture', 'revision': 1, 'bpm': 120,
                'audio': {'id': 'audio-fixture', 'bpm': 120, 'duration': 20, 'integratedLufs': -14.0},
                'config': {'beatmapVersion': '4'}, 'environment': 'DefaultEnvironment',
                'charts': [{'id': 'chart-fixture', 'name': 'Fallen', 'rank': 9, 'mode': 'Standard',
                            'njs': 18, 'data': fixture()}]}
            projects.write_json(data / 'projects' / project['id'] / 'project.json', project)
            before = (data / 'projects' / project['id'] / 'project.json').read_bytes()
            archive = projects.export_project(project['id'])
            with zipfile.ZipFile(archive) as z:
                info = json.loads(z.read('Info.dat')); self.assertEqual(info['version'], '4.0.1')
                native = json.loads(z.read('ExpertPlusStandard.dat'))
                lightshow = json.loads(z.read('ExpertPlusStandard.lightshow.dat'))
                tempo = json.loads(z.read('AudioData.dat'))
                validate_v4(native, lightshow, tempo)
                self.assertEqual(len(native['arcs']), 1)
                self.assertEqual(len(lightshow['eventBoxGroups']), 4)
                self.assertEqual(z.read('song.ogg'), b'OggS isolated export fixture')
            v3 = projects.export_project(project['id'], version='3')
            self.assertNotEqual(v3, archive)
            with zipfile.ZipFile(v3) as z:
                self.assertEqual(json.loads(z.read('Info.dat'))['_version'], '2.1.0')
                self.assertTrue(json.loads(z.read('ExpertPlusStandard.dat'))['version'].startswith('3'))
            self.assertEqual(before, (data / 'projects' / project['id'] / 'project.json').read_bytes())

    def test_v4_export_rejects_inconsistent_shared_audio_tempo(self):
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as temporary, patch.object(projects, 'DATA', pathlib.Path(temporary)):
            data = pathlib.Path(temporary); audio = data / 'audio' / 'audio-fixture'; audio.mkdir(parents=True)
            (audio / 'song.ogg').write_bytes(b'OggS isolated fixture')
            second = fixture(); second['bpmEvents'] = []
            project = {'id': 'fixture', 'name': 'Fixture', 'revision': 1, 'bpm': 120,
                'audio': {'id': 'audio-fixture', 'bpm': 120, 'duration': 20}, 'config': {'beatmapVersion': '4'},
                'charts': [{'id': 'a', 'name': 'Expert+', 'rank': 9, 'mode': 'Standard', 'data': fixture()},
                           {'id': 'b', 'name': 'Expert', 'rank': 7, 'mode': 'Standard', 'data': second}]}
            with self.assertRaisesRegex(FormatError, 'shared AudioData'):
                projects._export_project('fixture', project=project, folder=data/'draft')

    def test_native_v4_project_import_export_and_training_parser(self):
        import server
        from engine import dataset
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as temporary:
            data=pathlib.Path(temporary)
            with patch.object(server,'DATA',data), patch.object(projects,'DATA',data), patch.object(dataset,'CACHE_ROOT',data/'cache'):
                chart=fixture();chart['colorNotes']=[{'b': i*.25, 'x': i%4, 'y': i%3, 'c': i%2, 'd': 1 if i%4<2 else 0, 'a': 0} for i in range(40)]
                chart['njsEvents']=[{'b':4,'d':2,'p':0,'e':0}]
                packed=encode_v4(chart);tempo=build_audio_data(120,20,chart['bpmEvents'],.125)
                info2={'_songName':'Import fixture','_songAuthorName':'Test','_levelAuthorName':'Mapper','_beatsPerMinute':120,
                    '_songFilename':'song.ogg','_environmentNames':['DefaultEnvironment'],'_difficultyBeatmapSets':[{'_beatmapCharacteristicName':'Standard','_difficultyBeatmaps':[{
                    '_difficulty':'ExpertPlus','_difficultyRank':9,'_beatmapFilename':'ExpertPlusStandard.dat','_noteJumpMovementSpeed':18,'_noteJumpStartBeatOffset':0,'_customData':{'_difficultyLabel':'Fallen'}}]}]}
                info=info_v2_to_v4(info2,20);info['customData']['unknownPreservedMetadata']={'author':'test'}
                info['songPreviewFilename']='preview.ogg'
                input_path=data/'incoming.zip'
                with zipfile.ZipFile(input_path,'w') as z:
                    for name,value in {'Info.dat':info,'AudioData.dat':tempo,'ExpertPlusStandard.dat':packed['difficulty'],'ExpertPlusStandard.lightshow.dat':packed['lightshow']}.items():
                        z.writestr(name,gzip.compress(json.dumps(value).encode()) if name!='Info.dat' else json.dumps(value))
                    z.writestr('song.ogg',b'OggS fixture');z.writestr('preview.ogg',b'OggS preview fixture')
                corpus=dataset.read_archive(input_path)
                self.assertEqual(len(corpus['maps']),1);self.assertEqual(len(corpus['maps'][0]['notes']),40)
                self.assertTrue(corpus['maps'][0]['nativeV4']);self.assertAlmostEqual(corpus['maps'][0]['offset'],.125)
                async def upload(file,folder): folder.mkdir(parents=True,exist_ok=True);return input_path
                def analyze(source,folder,identity,name):
                    (folder/'song.ogg').write_bytes(b'OggS fixture')
                    return {'id':identity,'name':name,'duration':20,'bpm':120,'sampleRate':48000,'audioUrl':f'/api/audio/{identity}/song.ogg'}
                with patch.object(server,'store_upload',upload),patch.object(server,'analyze',analyze):
                    project=asyncio.run(server.import_project(None))
                self.assertEqual(project['config']['beatmapVersion'],'4')
                self.assertEqual(project['charts'][0]['data']['njsEvents'],chart['njsEvents'])
                self.assertAlmostEqual(project['songTimeOffset'],.125)
                export=projects.export_project(project['id'])
                with zipfile.ZipFile(export) as z:
                    self.assertEqual(json.loads(z.read('ExpertPlusStandard.dat')),packed['difficulty'])
                    self.assertEqual(json.loads(z.read('ExpertPlusStandard.lightshow.dat')),packed['lightshow'])
                    self.assertEqual(json.loads(z.read('AudioData.dat')),tempo)
                    exported_info=json.loads(z.read('Info.dat'))
                    self.assertEqual(exported_info['customData']['unknownPreservedMetadata'],{'author':'test'})
                    self.assertEqual(exported_info['songPreviewFilename'],'preview.ogg')
                    self.assertEqual(z.read('preview.ogg'),b'OggS preview fixture')
                with self.assertRaisesRegex(FormatError,'NJS'):projects.export_project(project['id'],version='3')


if __name__ == '__main__': unittest.main()
