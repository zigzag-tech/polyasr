"""Owned alignment jobs over Livestack's executor settlement protocol."""
from alignment_input import alignment_input_digest
from livestack_node.owned_executor_jobs import (
    CancellationSignal, JobCapacity, JobConflict, OwnedExecutorJobs, OwnedJobCancelled,
)


def alignment_digest(result):
    return alignment_input_digest(result)


class AlignmentJobs(OwnedExecutorJobs):
    def __init__(self, executor, synchronize, max_jobs=256):
        super().__init__(executor, synchronize, max_jobs=max_jobs, result_sha256=alignment_digest)
