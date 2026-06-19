# DreaMS web service — repo guide (branch: `fastapi-webapp`)

This branch replaces the Gradio demo with a **FastAPI + HTMX** web service for DreaMS spectral
library matching (deployed on terka, GPU 1). `app.py` on `main` (the Gradio app) is left
untouched. Design & plan: vault `CODE/DreaMS_inference/plans/deployment-architecture.md`.

Package: `dreams_web/`. Tooling: `ruff`, `black`, `mypy`, `pytest` (config in `pyproject.toml`).
Run before committing: `ruff check dreams_web tests && black --check dreams_web tests && mypy dreams_web && pytest -q`.

## Mission

When editing or generating Python here, write **minimal, correct code**:

- Prefer **early returns** to reduce nested `if`s.
- Remove **dead code**, unused variables/imports, and unreachable branches.
- Avoid placeholder code (e.g., `raise NotImplementedError` after doing the work).
- Keep public APIs small; don't add config flags unless clearly needed.
- Avoid hardcoding rules/types/values; prefer data-driven or composable design for future extension.

## Style

Use **PEP 8**:

1. **Naming** — variables/functions `snake_case`; constants `UPPER_CASE`; classes `CamelCase`; private `_leading_underscore`.
2. **Imports** — standard library first, third-party next, local last; one import per line.
3. **Docstrings & comments**
   - Triple-quoted docstrings on modules, classes, functions, methods. Template:
     ```python
     def factorial(n: int) -> int:
         """
         Calculate factorial of n recursively.
         Args:
             n (int): Non-negative integer.
         Returns:
             int: Factorial of n.
         """
     ```
   - Preserve rationale comments (assumptions, invariants, edge cases, trade-offs); compress, don't delete.
   - Add a short `#` comment immediately above any non-obvious conditional, loop, validation check,
     transformation, or external-API quirk — state the **reason/assumption/invariant/edge case**,
     not what the code does. Keep it ≤80 chars.
4. **Annotations** — built-in generics (`list`, `dict`, `tuple`); `|` unions and `T | None`
   (not `Optional[T]`); `from __future__ import annotations` when forward references are needed;
   `collections.abc` for abstract container types.
5. **Structure** — keep code modular; each file has one clear purpose.
