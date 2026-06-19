# dreams_web — DreaMS inference web service

FastAPI + HTMX service for **DreaMS spectral library matching** against MassSpecGym,
deployed on **terka** (GPU 1). This is a branch of `DreaMS_gradio` replacing the Gradio demo
with a backend+frontend we control; `app.py` (the Gradio app) on `main` is unchanged.

- **Design & plan:** vault `CODE/DreaMS_inference/plans/deployment-architecture.md`
- **Code style:** see `CLAUDE.md`

## Status

Phase 0 — scaffold (package, tooling, CI, tests). The build proceeds phase-by-phase, each
tested green before the next (see the plan's phase table).
