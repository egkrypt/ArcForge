"""Export existing real-generation evaluation artifacts as genuine native V4.

Reads stored songs and after-maps only. Copies audio to an isolated fixture
workspace before loudness analysis, so original audio/project records are never
modified. Outputs remain private under data/test-artifacts.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, pathlib, shutil, subprocess, sys, tempfile, zipfile
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bootstrap, projects
from audio import public_analysis
from engine.formats import decode_v4, audio_data_to_tempo
from engine.music import ensure_music_analysis
from modding import apply_mods, BeatClock


def validation_tools():
    requested=os.environ.get('ARCFORGE_NODE')
    node=shutil.which(requested or 'node')
    if not node and requested and pathlib.Path(requested).is_file():node=str(pathlib.Path(requested).resolve())
    if not node:raise ValueError('Install Node.js or set ARCFORGE_NODE to its executable before running native schema validation.')
    configured=os.environ.get('ARCFORGE_BSMAP_ROOT')
    schema=pathlib.Path(configured).expanduser().resolve() if configured else None
    if not schema or not (schema/'esm/beatmap/schema/v4/declaration/difficulty.js').is_file():
        raise ValueError('Install test-only bsmap@3.0.3 and valibot, then set ARCFORGE_BSMAP_ROOT to the installed bsmap package directory.')
    return node,str(schema)


def require_analysis(analyses,name):
    if name not in analyses:
        raise ValueError(f'Audio for {name!r} from the evaluation summary is no longer imported. Import the song and rerun tests/quality_benchmark.py after --song SONG, then validate that new summary.')
    return analyses[name]


def run(summary_path=None):
    summary_path=pathlib.Path(summary_path or ROOT/'data/test-artifacts/master-quality/after-summary.json').resolve()
    if not summary_path.is_file():
        raise ValueError('No real-generation summary was found. Import audio, train the local model and run tests/quality_benchmark.py after --song SONG first, or pass --summary PATH.')
    source=summary_path.parent
    summary=json.loads(summary_path.read_text('utf-8'))
    node,schema_root=validation_tools()
    private_root=ROOT/'data/test-artifacts'
    output=source/'v4-native' if source.is_relative_to(private_root.resolve()) else private_root/'native-v4-validation'
    output.mkdir(parents=True,exist_ok=True)
    analyses={}
    for file in sorted((ROOT/'data/audio').glob('*/analysis.json')):
        analysis=json.loads(file.read_text('utf-8'))
        if analysis.get('musicAnalysis'):analyses[analysis['name']]=analysis
    for file in sorted((ROOT/'data/audio').glob('*/analysis.json')):
        analysis=json.loads(file.read_text('utf-8'))
        analyses.setdefault(analysis['name'],analysis)
    rows=[]
    with tempfile.TemporaryDirectory(prefix='native-validation-',dir=output) as temporary:
        fixture_data=pathlib.Path(temporary)
        with patch.object(projects,'DATA',fixture_data):
            for index,row in enumerate(summary['rows']):
                analysis=ensure_music_analysis(copy.deepcopy(require_analysis(analyses,row['song'])));audio=ROOT/'data/audio'/analysis['id']/'song.ogg'
                if not audio.is_file():raise ValueError(f'The imported OGG for {row["song"]!r} is missing. Reimport the song and rerun the generation benchmark.')
                before=hashlib.sha256(audio.read_bytes()).hexdigest()
                folder=fixture_data/'audio'/analysis['id'];folder.mkdir(parents=True)
                shutil.copyfile(audio,folder/'song.ogg')
                artifact=pathlib.PurePosixPath(str(row['artifact']).replace('\\','/'))
                if artifact.is_absolute() or '..' in artifact.parts:
                    raise ValueError('Evaluation artifact paths must stay beside their summary.')
                artifact_path=source/pathlib.Path(*artifact.parts)
                if not artifact_path.is_file():raise ValueError('A generated chart artifact is missing. Rerun tests/quality_benchmark.py to recreate the summary and charts together.')
                raw=json.loads(artifact_path.read_text('utf-8'))
                # Exercise the same musical-plan-driven native lightshow stage
                # used by server generation, without regenerating note patterns.
                chart,modinfo=apply_mods(raw['beatmap'],analysis,{'bpm':row['bpm'],'seed':32026,
                    'mods':[],'compatibility':'Vanilla','lightingStyle':'Modern','events':{'lighting':True},
                    '_arcForgeMapPlan':raw['report'].get('mapPlan',{})})
                project={'id':f'{index:032x}','name':row['song'],'revision':1,'bpm':row['bpm'],
                    'audio':public_analysis(analysis),'config':{'beatmapVersion':'4','author':'ArcForge evaluation'},
                    'charts':[{'id':f'{index+10:032x}','name':'Expert+','rank':9,'mode':'Standard','njs':20,'jumpDistance':24,
                               'data':chart,'modInfo':modinfo}]}
                target=output/pathlib.Path(*artifact.with_suffix('').parts)
                target.mkdir(parents=True,exist_ok=True)
                archive=projects._export_project(project['id'],project=project,folder=target,extras_folder=target)
                with zipfile.ZipFile(archive) as z:
                    for name in ('Info.dat','AudioData.dat','ExpertPlusStandard.dat','ExpertPlusStandard.lightshow.dat'):
                        (target/name).write_bytes(z.read(name))
                    packed=json.loads(z.read('ExpertPlusStandard.dat'));lights=json.loads(z.read('ExpertPlusStandard.lightshow.dat'))
                    data=json.loads(z.read('AudioData.dat'))
                    restored=decode_v4(packed,lights,data);tempo=audio_data_to_tempo(data)
                    original_clock=BeatClock(row['bpm'],chart.get('bpmEvents',[]))
                    native_clock=BeatClock(tempo['bpm'],tempo['bpmEvents'])
                    timing=max((abs(original_clock.time(n['b'])-(native_clock.time(n['b'])+tempo['songTimeOffset'])) for n in chart['colorNotes']),default=0)
                    assert timing<.0001, timing
                    assert restored['colorNotes']==chart['colorNotes']
                    assert len(restored['basicBeatmapEvents'])==len(chart['basicBeatmapEvents'])
                env=os.environ.copy();env['ARCFORGE_BSMAP_ROOT']=schema_root
                result=subprocess.run([node,str(ROOT/'tests/validate_exports.mjs'),str(target)],env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=25)
                if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
                schemas=json.loads((target/'schema-validation.json').read_text('utf-8'))
                assert all(file['success'] for file in schemas)
                after=hashlib.sha256(audio.read_bytes()).hexdigest();assert before==after
                item={'song':row['song'],'notes':len(chart['colorNotes']),'nativeLights':len(chart['basicBeatmapEvents']),
                    'tempoSegments':len(data['bpmData']),'maxTimingRoundtripErrorMs':round(timing*1000,6),
                    'sourceAudioUnchanged':before==after,'nativeSchemasPassed':len(schemas),
                    'integratedLufs':json.loads((target/'Info.dat').read_text('utf-8'))['audio']['lufs'],
                    'archive':str(archive.relative_to(ROOT))}
                rows.append(item);print(json.dumps(item),flush=True)
    (output/'validation-report.json').write_text(json.dumps({'generatedBeforeExport':True,'notesRegeneratedDuringValidation':False,
        'gameplaySource':'master-quality after-map generation artifacts','lighting':'Same native lighting stage as server, vanilla Modern',
        'rows':rows,'scope':'Native schemas and clock preservation; no headset or ranked-acceptance claim'},indent=2),encoding='utf-8')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=pathlib.Path,help='Generated quality_benchmark summary; artifacts must remain beside it')
    args=parser.parse_args()
    try:run(args.summary)
    except (ValueError,FileNotFoundError) as error:parser.exit(2,str(error)+'\n')
