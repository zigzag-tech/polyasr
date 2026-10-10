"""Owned alignment jobs over Livestack's executor settlement protocol."""
import hashlib
import json
from livestack_node.owned_executor_jobs import (
    CancellationSignal, JobCapacity, JobConflict, OwnedExecutorJobs, OwnedJobCancelled,
)


def alignment_digest(result):
    return hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


class AlignmentJobs(OwnedExecutorJobs):
    def __init__(self, executor, synchronize, max_jobs=256):
        super().__init__(executor, synchronize, max_jobs=max_jobs, result_sha256=alignment_digest)
