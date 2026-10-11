"""Canonical owned alignment digest; standard library only, no ASR preparation."""
import hashlib
import json
import re
import sys


def alignment_input_digest(value):
    # Also hashes owned result records; preserve the original public function.
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def main():
    try:
        raw=sys.stdin.buffer.read(65537)
        if len(raw)>65536:raise ValueError('input bound')
        value=json.loads(raw)
        if not isinstance(value,dict) or set(value)!={'request'} or not isinstance(value['request'],dict):
            raise ValueError('closed owned alignment input required')
        request=value['request']
        if not set(request).issubset({'audio_sha256','language','max_chunk_seconds','model'}) or not isinstance(request.get('audio_sha256'),str) or not re.fullmatch('[a-f0-9]{64}',request['audio_sha256']):
            raise ValueError('exact owned alignment audio snapshot required')
        if 'max_chunk_seconds' in request and (type(request['max_chunk_seconds']) is not int or request['max_chunk_seconds']<=0):
            raise ValueError('owned alignment chunk bound required')
        if any(request.get(key) is not None and not isinstance(request[key],str) for key in ('language','model')):
            raise ValueError('owned alignment string fields required')
        print(json.dumps({'provider':'polyasr','inputDigest':alignment_input_digest(request)},separators=(',',':')))
    except (ValueError,TypeError,UnicodeError):
        print('canonical alignment input digest refused',file=sys.stderr);return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
