"""Upload-only owned ASR control API; old synchronous endpoints are unchanged."""
import asyncio
import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from fastapi import APIRouter, Body, File, Form, Header, HTTPException, UploadFile
from owned_alignment import JobCapacity, JobConflict, alignment_digest


def build_owned_alignment_router(jobs, runner, capability, *, authorize=None):
    router = APIRouter()

    def token(authorization):
        if not authorization or not authorization.startswith('Bearer '):
            raise HTTPException(401, 'owned alignment control required')
        return authorization[7:]

    def invoke(fn):
        try:
            return fn()
        except KeyError:
            raise HTTPException(404, 'unknown owned alignment job')
        except JobConflict:
            raise HTTPException(409, 'owned alignment identity conflict')
        except JobCapacity:
            raise HTTPException(503, 'owned alignment receipt capacity reached')
        except ValueError:
            raise HTTPException(400, 'invalid owned alignment control')

    @router.get('/v1/align/capabilities')
    def capabilities():
        return {**capability(), 'job_protocol_version': 1, 'server_instance_id': jobs.instance_id,
                'receipt_persistence': 'process_memory', 'bounded_physical_cancellation': False,
                'transports': ['upload']}

    @router.post('/v1/align/jobs')
    async def submit(submission: str = Form(...), file: UploadFile = File(...), x_provider_qualification_authorization: str = Header(None)):
        if capability().get('physical_cancellation', {}).get('supported') is not True:
            raise HTTPException(503, 'owned alignment physical settlement unqualified')
        try:
            parsed = json.loads(submission)
            if not isinstance(parsed, dict) or set(parsed) != {'job_id', 'owner_id', 'control_token', 'request'}:
                raise ValueError('invalid submission')
            if not all(isinstance(parsed[key], str) for key in ('job_id', 'owner_id', 'control_token')) or not parsed['owner_id'].strip() or len(parsed['control_token']) < 32:
                raise ValueError('invalid control identity')
            uuid.UUID(parsed['job_id'])
            request = parsed['request']
            if not isinstance(request, dict) or not set(request).issubset({'audio_sha256', 'language', 'max_chunk_seconds', 'model'}):
                raise ValueError('invalid request')
            seconds = request.get('max_chunk_seconds', 270)
            if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds <= 0:
                raise ValueError('invalid chunk duration')
            if request.get('language') is not None and not isinstance(request['language'], str):
                raise ValueError('invalid language')
            if request.get('model') is not None and not isinstance(request['model'], str):
                raise ValueError('invalid model')
            content = await file.read()
            if hashlib.sha256(content).hexdigest() != request.get('audio_sha256'):
                raise ValueError('audio snapshot mismatch')
        except (ValueError, TypeError, KeyError):
            raise HTTPException(400, 'invalid owned alignment submission')
        wrapper = None
        if authorize is not None:
            try:
                wrapper = await asyncio.to_thread(authorize,parsed,alignment_digest(request),x_provider_qualification_authorization)
            except PermissionError:
                raise HTTPException(403, 'qualification canonical alignment input refused')
            except ValueError:
                raise HTTPException(400, 'qualification alignment input binding refused')
        # Independent server-owned input survives the upload handler and client disconnect.
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as audio:
            audio.write(content)
            path = Path(audio.name)
        def work(signal):
            signal.check()
            return runner(path, request, signal)
        if wrapper is not None:
            work = wrapper(work)
        cleanup = lambda: path.unlink(missing_ok=True)
        try:
            return invoke(lambda: jobs.submit(parsed['job_id'], parsed['owner_id'], parsed['control_token'],
                alignment_digest(request), 'alignment', work, cleanup=cleanup))
        except BaseException:
            cleanup()
            raise

    @router.get('/v1/align/jobs/{job_id}')
    def status(job_id: str, authorization: str = Header(None)):
        return invoke(lambda: jobs.status(job_id, token(authorization)))

    @router.post('/v1/align/jobs/{job_id}/cancel')
    def cancel(job_id: str, payload: dict = Body(...), authorization: str = Header(None)):
        if set(payload) != {'owner_id'} or not isinstance(payload['owner_id'], str):
            raise HTTPException(400, 'owned alignment owner required')
        return invoke(lambda: jobs.cancel(job_id, token(authorization), payload['owner_id']))

    @router.get('/v1/align/jobs/{job_id}/result')
    def result(job_id: str, authorization: str = Header(None)):
        return invoke(lambda: jobs.result(job_id, token(authorization)))

    return router
