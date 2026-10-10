"""Optional real-export validator must not assume the development machine."""
import os,pathlib,sys,tempfile,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from validate_real_v4 import validation_tools,require_analysis,run


class ValidatorPortability(unittest.TestCase):
    def test_missing_node_has_actionable_setup_error(self):
        with patch.dict(os.environ,{},clear=True),patch('validate_real_v4.shutil.which',return_value=None):
            with self.assertRaisesRegex(ValueError,'ARCFORGE_NODE'):validation_tools()

    def test_missing_schema_has_actionable_setup_error(self):
        with patch.dict(os.environ,{},clear=True),patch('validate_real_v4.shutil.which',return_value=sys.executable):
            with self.assertRaisesRegex(ValueError,'ARCFORGE_BSMAP_ROOT'):validation_tools()

    def test_explicit_executable_and_schema_are_respected(self):
        with tempfile.TemporaryDirectory() as temporary:
            schema=pathlib.Path(temporary);file=schema/'esm/beatmap/schema/v4/declaration/difficulty.js'
            file.parent.mkdir(parents=True);file.write_text('export {};',encoding='utf-8')
            with patch.dict(os.environ,{'ARCFORGE_NODE':sys.executable,'ARCFORGE_BSMAP_ROOT':str(schema)},clear=True),patch('validate_real_v4.shutil.which',return_value=None):
                self.assertEqual(validation_tools(),(str(pathlib.Path(sys.executable).resolve()),str(schema.resolve())))

    def test_missing_historic_audio_or_summary_explains_regeneration(self):
        with self.assertRaisesRegex(ValueError,'Import the song and rerun'):require_analysis({},'Removed fixture')
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'--summary PATH'):run(pathlib.Path(temporary)/'missing.json')


if __name__=='__main__':unittest.main()
