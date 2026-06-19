"""FastAPI application serving DreaMS spectral library matching."""

import base64
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import mkdtemp

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from dreams_web.config import DEFAULT_SETTINGS, Settings
from dreams_web.service.annotate import annotate_upload
from dreams_web.service.errors import DreamsWebError
from dreams_web.service.reference_library import massspecgym_library
from dreams_web.service.search_backend import SearchBackend
from dreams_web.web.examples import EXAMPLES, ensure_example, get_example
from dreams_web.web.rendering import smiles_to_svg

_TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
_TEMPLATES.env.globals["mol_svg"] = smiles_to_svg


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the search backend (model + reference library) once, unless preset."""
    # Tests inject app.state.backend beforehand to skip the heavy library load.
    if not hasattr(app.state, "backend"):
        app.state.backend = SearchBackend(massspecgym_library(app.state.settings))
    yield


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

    def _render(request: Request, upload_path: Path, work_dir: Path) -> Response:
        """Annotate a file already staged in work_dir, then render and clean up."""
        try:
            result = annotate_upload(upload_path, app.state.backend, settings, work_dir)
        except DreamsWebError as exc:
            return _TEMPLATES.TemplateResponse(
                request, "error.html", {"message": str(exc)}, status_code=400
            )
        finally:
            # Results are already in memory; never keep the user's uploaded data.
            shutil.rmtree(work_dir, ignore_errors=True)

        tsv_b64 = base64.b64encode(result.tsv).decode("ascii")
        return _TEMPLATES.TemplateResponse(
            request, "results.html", {"result": result, "tsv_b64": tsv_b64}
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe."""
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Response:
        """Serve the upload page with the example shortcuts."""
        return _TEMPLATES.TemplateResponse(
            request, "index.html", {"examples": EXAMPLES}
        )

    @app.post("/annotate", response_class=HTMLResponse)
    async def annotate(request: Request, file: UploadFile) -> Response:
        """Annotate an uploaded MS/MS file and render the results table."""
        work_dir = Path(mkdtemp(prefix="dreams_job_"))
        upload_path = work_dir / (file.filename or "upload")
        upload_path.write_bytes(await file.read())
        return _render(request, upload_path, work_dir)

    @app.post("/examples/{example_id}", response_class=HTMLResponse)
    def run_example(request: Request, example_id: str) -> Response:
        """Annotate a prepared example file (downloaded from HF on first use)."""
        example = get_example(example_id)
        if example is None:
            raise HTTPException(status_code=404, detail="Unknown example")
        source = ensure_example(example, settings.examples_dir)
        work_dir = Path(mkdtemp(prefix="dreams_job_"))
        # Copy in so the loader's sibling .hdf5 lands in work_dir, not the examples dir.
        local = work_dir / example.filename
        shutil.copy2(source, local)
        return _render(request, local, work_dir)

    return app


app = create_app()
