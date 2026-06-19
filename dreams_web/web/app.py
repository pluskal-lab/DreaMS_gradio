"""FastAPI application serving DreaMS spectral library matching."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from shutil import copyfile, rmtree

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from dreams_web.config import DEFAULT_SETTINGS, SUPPORTED_EXTENSIONS, Settings
from dreams_web.jobs import Job, JobQueue, JobStatus, JobStore
from dreams_web.service.annotate import AnnotationResult, annotate_upload
from dreams_web.service.errors import InvalidInputError
from dreams_web.service.reference_library import massspecgym_library
from dreams_web.service.search_backend import SearchBackend
from dreams_web.web.examples import EXAMPLES, ensure_example, get_example
from dreams_web.web.rendering import smiles_to_svg

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_TEMPLATES.env.globals["mol_svg"] = smiles_to_svg
_REAP_INTERVAL_SECONDS = 60
_UPLOAD_CHUNK = 1 << 20  # 1 MiB


async def _reaper(store: JobStore) -> None:
    """Periodically delete expired jobs from memory."""
    while True:
        await asyncio.sleep(_REAP_INTERVAL_SECONDS)
        store.reap()


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the backend, job store, queue worker, and reaper at startup."""
    settings: Settings = app.state.settings
    # Tests inject app.state.backend beforehand to skip the heavy library load.
    if not hasattr(app.state, "backend"):
        app.state.backend = SearchBackend(massspecgym_library(settings))
    backend = app.state.backend

    def process(job: Job) -> AnnotationResult:
        return annotate_upload(job.upload_path, backend, settings, job.work_dir)

    store = JobStore(settings.jobs_dir, settings.job_retention_seconds)
    queue = JobQueue(store, process, settings.max_queued_jobs)
    queue.start()
    app.state.store = store
    app.state.queue = queue
    reaper = asyncio.create_task(_reaper(store))
    try:
        yield
    finally:
        reaper.cancel()
        await queue.stop()


def create_app(settings: Settings = DEFAULT_SETTINGS) -> FastAPI:
    """
    Build the FastAPI application.
    Args:
        settings (Settings): Service settings (caps, library + examples paths).
    Returns:
        FastAPI: The configured application.
    """
    app = FastAPI(title="DreaMS", lifespan=_lifespan)
    app.state.settings = settings

    def _error(request: Request, message: str, status: int = 400) -> Response:
        return _TEMPLATES.TemplateResponse(
            request, "error.html", {"message": message}, status_code=status
        )

    def _enqueue(request: Request, job: Job) -> Response:
        if not app.state.queue.submit(job.job_id):
            rmtree(job.work_dir, ignore_errors=True)
            return _error(
                request, "Server busy — too many jobs queued. Try again shortly.", 429
            )
        return _TEMPLATES.TemplateResponse(request, "pending.html", {"job": job})

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe (stays fast even while a job runs)."""
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        """Serve the upload page with the example shortcuts."""
        return _TEMPLATES.TemplateResponse(
            request, "index.html", {"examples": EXAMPLES}
        )

    @app.post("/jobs", response_class=HTMLResponse)
    async def submit_job(request: Request, file: UploadFile) -> Response:
        """Validate + stream an upload, enqueue a job, and return a polling view."""
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            return _error(request, f"Unsupported file type: {suffix or '(none)'}.")
        store: JobStore = app.state.store
        if store.disk_usage() > settings.job_store_max_bytes:
            store.reap()
            if store.disk_usage() > settings.job_store_max_bytes:
                return _error(
                    request, "Server busy — storage full. Try again later.", 503
                )
        job = store.create(suffix)
        # Stream to disk with a hard size cap so a huge upload can't exhaust RAM/disk.
        total = 0
        try:
            with job.upload_path.open("wb") as out:
                while chunk := await file.read(_UPLOAD_CHUNK):
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise InvalidInputError("File too large.")
                    out.write(chunk)
        except InvalidInputError as exc:
            rmtree(job.work_dir, ignore_errors=True)
            return _error(request, str(exc))
        return _enqueue(request, job)

    @app.post("/examples/{example_id}", response_class=HTMLResponse)
    def run_example(request: Request, example_id: str) -> Response:
        """Enqueue a prepared example file (downloaded from HF on first use)."""
        example = get_example(example_id)
        if example is None:
            raise HTTPException(status_code=404, detail="Unknown example")
        source = ensure_example(example, settings.examples_dir)
        job = app.state.store.create(source.suffix.lower())
        copyfile(source, job.upload_path)
        return _enqueue(request, job)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_status(request: Request, job_id: str) -> Response:
        """Return the job's current view: still-working, results, or error."""
        job = app.state.store.get(job_id)
        if job is None:
            return _error(request, "Job not found or expired.", 404)
        if job.status is JobStatus.DONE and job.result is not None:
            return _TEMPLATES.TemplateResponse(
                request, "results.html", {"result": job.result, "job": job}
            )
        if job.status is JobStatus.FAILED:
            return _error(request, job.error or "Job failed.")
        return _TEMPLATES.TemplateResponse(request, "pending.html", {"job": job})

    @app.get("/jobs/{job_id}/download")
    def download(job_id: str) -> Response:
        """Download a finished job's results as TSV."""
        job = app.state.store.get(job_id)
        if job is None or job.result is None:
            raise HTTPException(status_code=404, detail="Not found")
        return Response(
            content=job.result.tsv,
            media_type="text/tab-separated-values",
            headers={"Content-Disposition": "attachment; filename=dreams_hits.tsv"},
        )

    return app


app = create_app()
