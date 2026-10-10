import asyncio
import hashlib
import json
import threading
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import tempfile

import httpx
from fastapi import FastAPI
from owned_alignment import AlignmentJobs, alignment_digest
from owned_alignment_api import build_owned_alignment_router


class OwnedAlignmentApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.barriers = []
        self.jobs = AlignmentJobs(self.executor, lambda: self.barriers.append('synchronized'))
        self.calls, self.paths, self.uploads = [], [], []
        real_temporary = tempfile.NamedTemporaryFile
        def owned_temporary(*args, **kwargs):
            file = real_temporary(*args, **kwargs); self.uploads.append(Path(file.name)); return file
        self.temporary_patch = patch('owned_alignment_api.tempfile.NamedTemporaryFile', owned_temporary)
        self.temporary_patch.start()
        self.job_id, self.token = str(uuid.uuid4()), 'fixture-secret-' * 4
        self.content = b'fixture audio bytes'
        self.request = {'audio_sha256': hashlib.sha256(self.content).hexdigest(), 'language': 'en', 'max_chunk_seconds': 15}
        self.entered, self.release = threading.Event(), threading.Event()
        def runner(path, request, signal):
            self.calls.append(request); self.paths.append(path); self.entered.set()
            self.release.wait(timeout=2); signal.check()
            return {'text': 'Reviewed text.', 'segments': [], 'language': 'en', 'model': 'cpu-fixture'}
        app = FastAPI()
        app.include_router(build_owned_alignment_router(self.jobs, runner, lambda: {'physical_cancellation': {'supported': True, 'scope': ['alignment']}}))
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://fixture')

    async def asyncTearDown(self):
        self.release.set(); await self.client.aclose(); self.executor.shutdown(wait=True, cancel_futures=True)
        self.temporary_patch.stop()
        self.assertTrue(all(not path.exists() for path in self.uploads))

    async def submit(self, **changes):
        body = {'job_id': self.job_id, 'owner_id': 'owner', 'control_token': self.token, 'request': self.request, **changes}
        return await self.client.post('/v1/align/jobs', data={'submission': json.dumps(body)}, files={'file': ('audio.wav', self.content, 'audio/wav')})

    async def finish(self):
        await asyncio.to_thread(lambda: self.executor.submit(lambda: None).result(timeout=2))
        return (await self.client.get(f'/v1/align/jobs/{self.job_id}', headers={'Authorization': f'Bearer {self.token}'})).json()

    async def test_async_acceptance_is_owned_until_worker_and_barrier_settle(self):
        accepted = await self.submit(); self.assertEqual(accepted.status_code, 200)
        self.assertTrue(await asyncio.to_thread(self.entered.wait, 2))
        cancel = await self.client.post(f'/v1/align/jobs/{self.job_id}/cancel', json={'owner_id': 'owner'}, headers={'Authorization': f'Bearer {self.token}'})
        self.assertEqual(cancel.json()['state'], 'cancel_requested')
        self.assertFalse(cancel.json()['physical_settled'])
        self.release.set(); receipt = await self.finish()
        self.assertEqual(receipt['state'], 'cancelled'); self.assertTrue(receipt['physical_settled'])
        self.assertEqual(self.barriers, ['synchronized']); self.assertFalse(self.paths[0].exists())
        self.assertEqual((await self.client.get(f'/v1/align/jobs/{self.job_id}/result', headers={'Authorization': f'Bearer {self.token}'})).status_code, 409)

    async def test_replay_returns_identical_alignment_without_reexecution_or_leaked_input(self):
        self.release.set(); self.assertEqual((await self.submit()).status_code, 200)
        receipt = await self.finish()
        self.assertEqual((await self.submit()).status_code, 200)
        self.assertEqual(len(self.calls), 1)
        result = (await self.client.get(f'/v1/align/jobs/{self.job_id}/result', headers={'Authorization': f'Bearer {self.token}'})).json()
        self.assertEqual(receipt['result_sha256'], alignment_digest(result))
        self.assertNotIn(self.token, str(receipt))
        self.assertEqual((await self.client.get(f'/v1/align/jobs/{self.job_id}', headers={'Authorization': 'Bearer wrong-secret-' * 4})).status_code, 404)
        self.assertEqual((await self.submit(request={**self.request, 'language': 'zh'})).status_code, 409)

    async def test_cancel_before_submit_and_changed_audio_refuse_new_inference(self):
        response = await self.client.post(f'/v1/align/jobs/{self.job_id}/cancel', json={'owner_id': 'owner'}, headers={'Authorization': f'Bearer {self.token}'})
        self.assertTrue(response.json()['physical_settled'])
        self.assertEqual((await self.submit()).json()['state'], 'cancelled'); self.assertEqual(self.calls, [])
        self.assertEqual((await self.submit(request={**self.request, 'audio_sha256': '0' * 64})).status_code, 400)
