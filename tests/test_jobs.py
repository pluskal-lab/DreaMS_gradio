"""Tests for the job store and single-worker queue (model-free)."""

import asyncio
import time
from pathlib import Path

from dreams_web.jobs import Job, JobQueue, JobStatus, JobStore
from dreams_web.service.annotate import AnnotationResult
from dreams_web.service.errors import InvalidInputError


def _ok_result() -> AnnotationResult:
    """A canned successful result (no model needed)."""
    return AnnotationResult(
        display_rows=[{"x": 1}], tsv=b"x", n_query_spectra=1, n_matches=1
    )


def test_store_create_and_reap(tmp_path: Path) -> None:
    """Finished jobs past the retention window are reaped from memory and disk."""
    store = JobStore(tmp_path, retention_seconds=0)
    job = store.create(".mgf")
    assert store.get(job.job_id) is job
    assert job.work_dir.is_dir()
    assert job.upload_path.name == "input.mgf"

    job.status = JobStatus.DONE
    time.sleep(0.01)
    assert store.reap() == 1
    assert store.get(job.job_id) is None
    assert not job.work_dir.exists()


def test_queue_full_rejects(tmp_path: Path) -> None:
    """submit() returns False once the bounded queue is full (no worker draining)."""
    store = JobStore(tmp_path, retention_seconds=60)
    queue = JobQueue(store, process=lambda job: _ok_result(), max_queued=1)
    assert queue.submit("a") is True
    assert queue.submit("b") is False


def test_worker_serializes_and_completes(tmp_path: Path) -> None:
    """The single worker runs jobs one at a time; all finish DONE."""
    store = JobStore(tmp_path, retention_seconds=60)
    active = 0
    max_active = 0

    def process(job: Job) -> AnnotationResult:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        time.sleep(0.05)
        active -= 1
        return _ok_result()

    async def scenario() -> None:
        queue = JobQueue(store, process=process, max_queued=10)
        queue.start()
        jobs = [store.create(".mgf") for _ in range(5)]
        for job in jobs:
            assert queue.submit(job.job_id)
        for _ in range(200):
            if all(store.get(j.job_id).status is JobStatus.DONE for j in jobs):
                break
            await asyncio.sleep(0.02)
        await queue.stop()
        assert all(store.get(j.job_id).status is JobStatus.DONE for j in jobs)
        # The worker must never run two jobs at once (single GPU).
        assert max_active == 1

    asyncio.run(scenario())


def test_worker_marks_dreams_error_cleanly(tmp_path: Path) -> None:
    """A DreamsWebError in the processor yields FAILED with its message."""

    def process(job: Job) -> AnnotationResult:
        raise InvalidInputError("bad file")

    async def scenario() -> None:
        store = JobStore(tmp_path, retention_seconds=60)
        queue = JobQueue(store, process=process, max_queued=10)
        queue.start()
        job = store.create(".mgf")
        queue.submit(job.job_id)
        for _ in range(200):
            if store.get(job.job_id).status is JobStatus.FAILED:
                break
            await asyncio.sleep(0.02)
        await queue.stop()
        failed = store.get(job.job_id)
        assert failed is not None
        assert failed.status is JobStatus.FAILED
        assert failed.error is not None and "bad file" in failed.error

    asyncio.run(scenario())
