"""In-memory job model, store, and a single-worker async queue."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

from dreams_web.service.annotate import AnnotationResult
from dreams_web.service.errors import DreamsWebError

_logger = logging.getLogger("dreams_web.jobs")


class JobStatus(StrEnum):
    """Lifecycle states of an annotation job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    """One annotation job and its working directory."""

    job_id: str
    work_dir: Path
    upload_path: Path
    created_at: float
    status: JobStatus = JobStatus.QUEUED
    result: AnnotationResult | None = None
    error: str | None = None


class JobStore:
    """Holds jobs in memory and reaps expired ones from disk."""

    def __init__(self, jobs_dir: Path, retention_seconds: int) -> None:
        """
        Args:
            jobs_dir (Path): Root directory for per-job working directories.
            retention_seconds (int): Age after which a finished job is reaped.
        """
        self._jobs: dict[str, Job] = {}
        self._jobs_dir = jobs_dir
        self._retention_seconds = retention_seconds

    def create(self, suffix: str) -> Job:
        """
        Create a job with a fresh working directory.
        Args:
            suffix (str): Extension for the input file (e.g. '.mgf'); the on-disk
                name is always 'input<suffix>' so a user filename can't traverse.
        Returns:
            Job: The newly created, queued job.
        """
        job_id = uuid4().hex
        work_dir = self._jobs_dir / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        job = Job(
            job_id=job_id,
            work_dir=work_dir,
            upload_path=work_dir / f"input{suffix}",
            created_at=time.time(),
        )
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job with this id, or None if unknown."""
        return self._jobs.get(job_id)

    def disk_usage(self) -> int:
        """Total bytes used across all job working directories."""
        if not self._jobs_dir.exists():
            return 0
        return sum(f.stat().st_size for f in self._jobs_dir.rglob("*") if f.is_file())

    def reap(self) -> int:
        """
        Delete finished jobs older than the retention window.
        Returns:
            int: Number of jobs reaped.
        """
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in (JobStatus.DONE, JobStatus.FAILED)
            and now - job.created_at > self._retention_seconds
        ]
        for job_id in expired:
            job = self._jobs.pop(job_id)
            rmtree(job.work_dir, ignore_errors=True)
        return len(expired)


class JobQueue:
    """A single-worker async queue so GPU inference is serialized."""

    def __init__(
        self,
        store: JobStore,
        process: Callable[[Job], AnnotationResult],
        max_queued: int,
    ) -> None:
        """
        Args:
            store (JobStore): Where jobs are looked up.
            process (Callable[[Job], AnnotationResult]): Blocking job processor;
                run in a worker thread so the event loop stays responsive.
            max_queued (int): Maximum jobs waiting before submit() rejects.
        """
        self._store = store
        self._process = process
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queued)
        self._worker: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the background worker (idempotent)."""
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the background worker."""
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None

    def submit(self, job_id: str) -> bool:
        """
        Enqueue a job without blocking.
        Returns:
            bool: True if queued, False if the queue is full.
        """
        try:
            self._queue.put_nowait(job_id)
            return True
        except asyncio.QueueFull:
            return False

    async def _run(self) -> None:
        """Process queued jobs one at a time (serializes the single GPU)."""
        while True:
            job_id = await self._queue.get()
            try:
                await self._process_one(job_id)
            finally:
                self._queue.task_done()

    async def _process_one(self, job_id: str) -> None:
        """Run one job off the event loop and record its outcome."""
        job = self._store.get(job_id)
        if job is None:
            return
        job.status = JobStatus.RUNNING
        try:
            # Off the event loop so the server keeps answering other requests.
            job.result = await asyncio.to_thread(self._process, job)
            job.status = JobStatus.DONE
        except DreamsWebError as exc:
            job.error = str(exc)
            job.status = JobStatus.FAILED
        except Exception:
            # Unexpected failure: log internally, show the user a clean message.
            _logger.exception("job %s failed", job_id)
            job.error = "Could not process the spectra. Please check the file."
            job.status = JobStatus.FAILED
        finally:
            # The result is in memory; free the on-disk working files immediately.
            rmtree(job.work_dir, ignore_errors=True)
