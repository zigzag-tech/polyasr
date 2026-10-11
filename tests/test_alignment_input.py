"""Actual isolated standard-library CLI; no alignment or model preparation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from alignment_input import alignment_input_digest

class AlignmentInputTests(unittest.TestCase):
    def cli(self,value):
        return subprocess.run([sys.executable,'-I','-B',str(Path(__file__).resolve().parents[1]/'alignment_input.py')],input=json.dumps(value,ensure_ascii=False).encode(),capture_output=True)
    def test_original_unicode_rules_insertion_order_and_explicit_defaults_remain_exact(self):
        request={'audio_sha256':'a'*64,'model':'fixture — β','language':'en','max_chunk_seconds':15}
        original=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        result=self.cli({'request':dict(reversed(list(request.items())))})
        self.assertEqual(result.returncode,0);self.assertEqual(json.loads(result.stdout),{'provider':'polyasr','inputDigest':original})
        self.assertNotEqual(alignment_input_digest({'audio_sha256':'a'*64}),alignment_input_digest({'audio_sha256':'a'*64,'max_chunk_seconds':270}))
        output={'segments':[],'text':'Exact — β'}
        self.assertEqual(alignment_input_digest(output),hashlib.sha256(json.dumps(output,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest())
    def test_control_secret_unknown_field_and_boolean_chunk_refuse_without_echo(self):
        for value in ({'request':{'audio_sha256':'a'*64},'control_token':'PRIVATE-FIXTURE'},{'request':{'audio_sha256':'a'*64,'unknown':1}},{'request':{'audio_sha256':'a'*64,'max_chunk_seconds':True}}):
            result=self.cli(value);self.assertEqual(result.returncode,2);self.assertEqual(result.stdout,b'');self.assertNotIn(b'PRIVATE-FIXTURE',result.stderr)

if __name__=='__main__':unittest.main()
