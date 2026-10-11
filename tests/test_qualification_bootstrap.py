"""Actual CUDA-server source wiring; no model imports/full-server qualification."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
import threading
from unittest.mock import patch
from fastapi import HTTPException

class QualificationBootstrapTests(unittest.TestCase):
    def namespace(self,enabled,gate):
        source=Path(__file__).resolve().parents[1]/'cuda/server.py';tree=ast.parse(source.read_text())
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_authorize_owned_qualification')
        namespace={'QUALIFICATION_ONLY':enabled,'app':SimpleNamespace(state=SimpleNamespace(qualification_gate=gate)),'HTTPException':HTTPException,'_synchronize_owned_alignment':lambda:None}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
        return namespace['_authorize_owned_qualification']
    def test_qualified_cold_barrier_does_not_initialize_cuda_and_legacy_device_is_retained(self):
        source=Path(__file__).resolve().parents[1]/'cuda/server.py';tree=ast.parse(source.read_text())
        node=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_synchronize_owned_alignment')
        calls=[];torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False,synchronize=lambda device:calls.append(device)))
        namespace={'QUALIFICATION_ONLY':True,'_transcribe_lock':threading.Lock(),'DEVICE':'cuda:0'}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
        with patch.dict('sys.modules',{'torch':torch}):
            namespace['_synchronize_owned_alignment']();self.assertEqual(calls,[])
            namespace['QUALIFICATION_ONLY']=False;namespace['_synchronize_owned_alignment']()
        self.assertEqual(calls,['cuda:0'])

    def test_default_is_preserved_and_unconfigured_qualification_refuses(self):
        self.assertIsNone(self.namespace(False,None)(None,None,None))
        with self.assertRaises(HTTPException) as error:self.namespace(True,None)(None,None,None)
        self.assertEqual(error.exception.status_code,503)
    def test_missing_auto_and_nonenglish_language_refuse_before_gate(self):
        class Gate:
            def admit(self,*args):raise AssertionError('invalid input reached gate')
        authorize=self.namespace(True,Gate())
        for request in ({},{'language':None},{'language':'auto'},{'language':'zh'}):
            with self.assertRaises(PermissionError):authorize({'request':request},None,None)
    def test_actual_worker_wrapper_preserves_job_hash_and_enters_gate_inside_worker(self):
        events=[]
        class Gate:
            def admit(self,job,authorization):events.append(('admit',job,authorization));return {'fence':7}
            @contextmanager
            def work(self,job,authorization,fence,*,synchronize):events.append(('work',job,authorization,fence));yield;events.append('settled')
        parsed={'request':{'language':'en'},'job_id':'fixture-id','owner_id':'fixture-owner'}
        wrap=self.namespace(True,Gate())(parsed,'exact-owner-fingerprint','fixture-permission')
        signal=SimpleNamespace(check=lambda:events.append('checked'))
        self.assertEqual(wrap(lambda signal:events.append('physical-worker') or {})(signal),{})
        job={'jobId':'fixture-id','ownerId':'fixture-owner','inputDigest':'exact-owner-fingerprint','modelKind':'asr'}
        self.assertEqual(events,[('admit',job,'fixture-permission'),'checked',('work',job,'fixture-permission',7),'physical-worker','settled'])

if __name__=='__main__':unittest.main()
