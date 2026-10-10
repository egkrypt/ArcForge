import sys,pathlib,tempfile,json,zipfile,hashlib,io,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
from training_runtime import runtime_status,select_runtime,recommend_training,TrainingCancelled
from engine import trainer
from playlists import parse_playlist,validate_url,beatsaver_hash,download_playlist

def map_zip():
    info={'_version':'2.1.0','_songName':'Test human chart','_songFilename':'song.ogg','_beatsPerMinute':190,
          '_difficultyBeatmapSets':[{'_beatmapCharacteristicName':'Standard','_difficultyBeatmaps':[{'_difficulty':'ExpertPlus','_beatmapFilename':'ExpertPlus.dat'}]}]}
    chart={'version':'3.3.0','colorNotes':[{'b':float(i)/2,'x':i%4,'y':i%3,'c':i%2,'d':i%2,'a':0} for i in range(48)]}
    info_bytes=json.dumps(info).encode();chart_bytes=json.dumps(chart).encode()
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w') as z:z.writestr('Info.dat',info_bytes);z.writestr('ExpertPlus.dat',chart_bytes)
    return payload.getvalue(),hashlib.sha1(info_bytes+chart_bytes).hexdigest()

class RuntimeTests(unittest.TestCase):
    def test_selection_and_recommendation(self):
        self.assertEqual(select_runtime('cpu',threads=1)['backend'],'cpu')
        with self.assertRaises(ValueError):select_runtime('pretend')
        with self.assertRaises(ValueError):select_runtime('cpu',threads=0)
        self.assertEqual(recommend_training({'archives':[{'mapCount':1,'noteCount':100}]*127})['epochs'],24)
        self.assertEqual(recommend_training({'archives':[{'mapCount':0,'noteCount':0}]*127})['usableArchives'],0)

    def test_playlist_validation_and_duplicate_skipping(self):
        blob,digest=map_zip()
        playlist={'playlistTitle':'Training','songs':[{'hash':digest.upper()},{'hash':digest},{'key':'12162'},{'hash':'../bad'}]}
        parsed=parse_playlist(playlist);self.assertEqual(parsed['songCount'],2);self.assertEqual(len(parsed['invalid']),1)
        for url in ['http://api.beatsaver.com/maps/id/1','https://evil.example/x','https://api.beatsaver.com@evil.example/x','https://cdn.beatsaver.com:80/x']:
            with self.assertRaises(ValueError):validate_url(url)
        def fetch(url,limit,cancel):
            if '/maps/' in url:return json.dumps({'id':'12162','automapper':False,'declaredAi':'None','versions':[{'hash':digest,'state':'Published','downloadURL':f'https://r2cdn.beatsaver.com/{digest}.zip'}]}).encode()
            return blob
        with tempfile.TemporaryDirectory() as temp:
            result=download_playlist({'songs':[{'hash':digest}]},temp,fetcher=fetch)
            self.assertEqual(len(result['downloaded']),1)
            saved=pathlib.Path(result['downloaded'][0]['path']);self.assertEqual(beatsaver_hash(saved),digest)
            second=download_playlist({'songs':[{'hash':digest}]},temp,fetcher=lambda *_:self.fail('Existing map should not download'))
            self.assertEqual(len(second['skipped']),1)
            with self.assertRaises(TrainingCancelled):download_playlist({'songs':[{'hash':digest}]},temp,cancel_callback=lambda:True)

    def test_mismatch_and_ai_rejection(self):
        blob,digest=map_zip();bad='f'*40
        def mismatch(url,limit,cancel):
            if '/maps/' in url:return json.dumps({'id':'1','versions':[{'hash':bad,'state':'Published','downloadURL':'https://r2cdn.beatsaver.com/'+bad+'.zip'}]}).encode()
            return blob
        with tempfile.TemporaryDirectory() as temp:
            r=download_playlist({'songs':[{'hash':bad}]},temp,fetcher=mismatch)
            self.assertEqual(len(r['downloaded']),0);self.assertIn('SHA-1',r['errors'][0]['error']);self.assertFalse(list(pathlib.Path(temp).glob('*.zip')))

    def test_cpu_training_warmstart_cancellation_and_cuda(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(1200,44)).astype(np.float32);y=(x[:,0]>0).astype(np.int32);ids=np.repeat(np.arange(3),400)
        provenance=[{'archive':f'{i}.zip','sourceIndex':i,'sha256':str(i+1)*64,'maps':1,'notes':400} for i in range(3)]
        original=trainer._collect
        trainer._collect=lambda *a,**k:(x,y,ids,provenance,[],{},3,np.zeros((3,16),np.float32),{})
        try:
            with tempfile.TemporaryDirectory() as temp:
                cpu=trainer.train_model(temp,temp,epochs=2,backend='cpu',threads=1)
                self.assertEqual(cpu['runtime']['backend'],'cpu');self.assertTrue(np.isfinite(cpu['bestValidationLoss']))
                checkpoint=(pathlib.Path(temp)/'model.npz').read_bytes()
                resumed=trainer.train_model(temp,temp,epochs=1,backend='cpu',threads=1)
                self.assertTrue(resumed['warmStarted']);self.assertLessEqual(resumed['bestValidationLoss'],cpu['bestValidationLoss']+.001)
                checkpoint=(pathlib.Path(temp)/'model.npz').read_bytes()
                with self.assertRaises(TrainingCancelled):trainer.train_model(temp,temp,epochs=2,backend='cpu',cancel_callback=lambda:True)
                self.assertEqual((pathlib.Path(temp)/'model.npz').read_bytes(),checkpoint)
                if runtime_status(refresh=True)['gpu']['available']:
                    gpu=trainer.train_model(temp,temp,epochs=2,backend='gpu')
                    self.assertEqual(gpu['runtime']['backend'],'gpu');self.assertIn('RTX',gpu['runtime']['device']);self.assertTrue(np.isfinite(gpu['bestValidationLoss']))
        finally:trainer._collect=original

if __name__=='__main__':unittest.main()
