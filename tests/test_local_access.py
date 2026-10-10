"""Loopback service access must resist hostname rebinding without network calls."""
import pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import bootstrap,server
from fastapi import FastAPI
from fastapi.testclient import TestClient


class LocalAccess(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        self.app=FastAPI()
        self.app.middleware('http')(server.local_access)
        @self.app.api_route('/probe',methods=['GET','POST','OPTIONS'])
        def probe():
            self.calls.append(True)
            return {'local':True}

    def test_exact_loopback_names_and_ports_remain_usable(self):
        for authority in ['localhost','localhost:8769','LOCALHOST:5173','127.0.0.1:8769','[::1]:8769']:
            with self.subTest(authority=authority):
                response=TestClient(self.app,base_url='http://'+authority).get('/probe')
                self.assertEqual(response.status_code,200,response.text)

    def test_remote_hostname_reads_and_writes_never_reach_routes(self):
        client=TestClient(self.app)
        for method in ['GET','POST','OPTIONS']:
            response=client.request(method,'/probe',headers={'host':'example.invalid','origin':'http://localhost:8769'})
            self.assertEqual(response.status_code,403)
        self.assertEqual(self.calls,[])

    def test_malformed_or_lookalike_hostnames_are_rejected(self):
        client=TestClient(self.app)
        for host in ['localhost.evil.invalid','127.0.0.1.evil.invalid','0x7f000001','evil@localhost','@localhost',
                     'localhost/path','localhost?query','localhost#fragment','localhost:bad','localhost:0',
                     'localhost:65536',' localhost','localhost ']:
            with self.subTest(host=host):self.assertEqual(client.get('/probe',headers={'host':host}).status_code,403)
        self.assertEqual(self.calls,[])

    def test_duplicate_host_headers_are_rejected(self):
        client=TestClient(self.app)
        response=client.get('/probe',headers=[('host','localhost'),('host','example.invalid')])
        self.assertEqual(response.status_code,403)
        self.assertEqual(self.calls,[])

    def test_testserver_exception_requires_in_process_transport_marker(self):
        self.assertEqual(TestClient(self.app).get('/probe').status_code,200)
        network_client=TestClient(self.app,client=('127.0.0.1',50100))
        self.assertEqual(network_client.get('/probe').status_code,403)
        self.assertEqual(network_client.get('/probe',headers={'host':'localhost:8769'}).status_code,200)
        self.assertEqual(TestClient(self.app,base_url='http://127.0.0.1:8769').get('/probe',
            headers={'host':'testserver'}).status_code,403)

    def test_testserver_without_transport_extension_is_rejected(self):
        app=FastAPI()
        app.middleware('http')(server.local_access)
        @app.middleware('http')
        async def strip_transport_extension(request,call_next):
            request.scope['extensions'].pop('http.response.debug',None)
            return await call_next(request)
        @app.get('/probe')
        def probe():return {'local':True}
        self.assertEqual(TestClient(app).get('/probe').status_code,403)

    def test_remote_write_origins_remain_blocked(self):
        client=TestClient(self.app,base_url='http://127.0.0.1:8769')
        self.assertEqual(client.post('/probe',headers={'origin':'https://example.invalid'}).status_code,403)
        self.assertEqual(client.post('/probe',headers={'origin':'http://localhost:5173'}).status_code,200)
        self.assertEqual(len(self.calls),1)


if __name__=='__main__':unittest.main()
