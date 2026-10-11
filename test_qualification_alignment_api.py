"""Real upload/job ASGI route; permission/worker gate are explicit CPU fixtures."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from fastapi import FastAPI
from httpx import AsyncClient,ASGITransport
from owned_alignment import AlignmentJobs,alignment_digest
from owned_alignment_api import build_owned_alignment_router

class QualificationAlignmentApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor=ThreadPoolExecutor(max_workers=1);self.jobs=AlignmentJobs(self.executor,lambda:None)
        self.events=[];self.uploads=[];self.revoked=False
        self.permission='Bearer zzops_'+'d'*64;self.token='fixture-control-secret-'*3;self.audio=b'CPU fixture only'
        self.request={'audio_sha256':hashlib.sha256(self.audio).hexdigest(),'language':'en','max_chunk_seconds':15}
        self.body={'job_id':str(uuid4()),'owner_id':'fixture-owner','control_token':self.token,'request':self.request}
        real_temp=tempfile.NamedTemporaryFile
        def temporary(*args,**kwargs):
            file=real_temp(*args,**kwargs);self.uploads.append(Path(file.name));return file
        self.patch=patch('owned_alignment_api.tempfile.NamedTemporaryFile',temporary);self.patch.start()
        def authorize(parsed,digest,authorization):
            if authorization!=self.permission or self.revoked:raise PermissionError('mock denied')
            self.assertEqual(digest,alignment_digest(self.request));self.assertEqual(parsed,self.body)
            self.events.append('permission')
            def wrap(worker):
                def physical(signal):self.events.append('physical-scope');return worker(signal)
                return physical
            return wrap
        def runner(path,request,signal):
            self.events.append('worker');self.assertEqual(path.read_bytes(),self.audio);return {'text':'CPU fixture','segments':[],'language':'en'}
        app=FastAPI();app.include_router(build_owned_alignment_router(self.jobs,runner,lambda:{'physical_cancellation':{'supported':True}},authorize=authorize))
        self.client=AsyncClient(transport=ASGITransport(app=app),base_url='http://fixture')
    async def asyncTearDown(self):
        await self.client.aclose();self.executor.shutdown(wait=True);self.patch.stop()
        self.assertTrue(all(not path.exists() for path in self.uploads))
    async def submit(self,permission):
        return await self.client.post('/v1/align/jobs',data={'submission':json.dumps(self.body)},files={'file':('fixture.wav',self.audio,'audio/wav')},headers={'X-Provider-Qualification-Authorization':permission})
    async def test_anonymous_forged_and_control_tokens_refuse_before_owned_upload_or_runner(self):
        for token in ('','Bearer forged','Bearer '+self.token):self.assertEqual((await self.submit(token)).status_code,403)
        self.assertEqual(self.events,[]);self.assertEqual(self.uploads,[])
    async def test_exact_request_audio_and_digest_precede_actual_worker_scope(self):
        response=await self.submit(self.permission);self.assertEqual(response.status_code,200)
        self.executor.submit(lambda:None).result(timeout=2)
        self.assertEqual(self.events,['permission','physical-scope','worker']);self.assertNotIn(self.permission,response.text)
    async def test_owned_result_remains_reachable_after_canonical_revocation(self):
        self.assertEqual((await self.submit(self.permission)).status_code,200)
        self.executor.submit(lambda:None).result(timeout=2);self.revoked=True
        self.assertEqual((await self.submit(self.permission)).status_code,403)
        response=await self.client.get('/v1/align/jobs/'+self.body['job_id']+'/result',headers={'Authorization':'Bearer '+self.token})
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()['text'],'CPU fixture')

if __name__=='__main__':unittest.main()
