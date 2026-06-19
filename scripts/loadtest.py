"""Load + adversarial probe for the DreaMS web service.

Usage:
    python scripts/loadtest.py [BASE_URL] [N_CONCURRENT]

Floods the server with concurrent example jobs and polls them to completion,
then checks that /health stays responsive under load and that bad inputs are
handled cleanly (no 500s, no path traversal). Run against a disposable server.
"""

import asyncio
import os
import re
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
N_CONCURRENT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
_JOB_ID = re.compile(r"/jobs/([0-9a-f]+)")


async def _submit_example(client: httpx.AsyncClient, example_id: str) -> str | None:
    """Submit an example job and return its id (or None on failure)."""
    response = await client.post(f"{BASE}/examples/{example_id}")
    match = _JOB_ID.search(response.text)
    return match.group(1) if match else None


async def _poll(
    client: httpx.AsyncClient, job_id: str, timeout: float = 240
) -> tuple[str, float]:
    """Poll a job until it shows results or an error."""
    start = time.time()
    while time.time() - start < timeout:
        response = await client.get(f"{BASE}/jobs/{job_id}")
        if "Matches" in response.text:
            return "done", round(time.time() - start, 1)
        if "Error" in response.text:
            return "error", round(time.time() - start, 1)
        await asyncio.sleep(0.5)
    return "timeout", timeout


async def flood(n: int) -> None:
    """Submit n example jobs at once and wait for all to finish."""
    async with httpx.AsyncClient() as client:
        start = time.time()
        submitted = await asyncio.gather(
            *(_submit_example(client, "drugs") for _ in range(n))
        )
        ids = [job_id for job_id in submitted if job_id]
        outcomes = await asyncio.gather(*(_poll(client, job_id) for job_id in ids))
    print(f"[FLOOD] {len(ids)}/{n} accepted; all terminal in {round(time.time()-start,1)}s")
    print(f"   outcomes: {outcomes}")


async def responsiveness() -> None:
    """Measure /health latency while a slow (2k-spectra) job runs."""
    async with httpx.AsyncClient() as client:
        job_id = await _submit_example(client, "piper")
        await asyncio.sleep(1)
        samples: list[tuple[object, object]] = []
        for _ in range(8):
            start = time.time()
            try:
                response = await client.get(f"{BASE}/health", timeout=5)
                samples.append((response.status_code, round(time.time() - start, 3)))
            except Exception as exc:
                samples.append(("ERR", str(exc)[:20]))
            await asyncio.sleep(0.5)
        if job_id:
            await _poll(client, job_id)
    floats = [s for _, s in samples if isinstance(s, float)]
    print(f"[RESPONSIVENESS] worst /health latency during a 2k job: {max(floats, default=None)}s")
    print(f"   samples: {samples}")


async def hardening() -> None:
    """Submit bad inputs and confirm clean handling (no 500, no traversal)."""
    pwn = "/tmp/dreams_pwn2.txt"
    if os.path.exists(pwn):
        os.remove(pwn)
    cases = [
        ("unsupported.txt", b"hello"),
        ("garbage.mgf", os.urandom(4000)),
        ("empty.mgf", b""),
        ("../../../../tmp/dreams_pwn2.txt.mgf", b"BEGIN IONS\nEND IONS\n"),
    ]
    async with httpx.AsyncClient() as client:
        for filename, payload in cases:
            response = await client.post(
                f"{BASE}/jobs",
                files={"file": (filename, payload, "application/octet-stream")},
                timeout=120,
            )
            outcome = f"submit={response.status_code}"
            if response.status_code == 200:
                match = _JOB_ID.search(response.text)
                if match:
                    outcome += f" -> {(await _poll(client, match.group(1)))[0]}"
            print(f"[HARDEN {filename[:26]:26}] {outcome}")
    print(f"[HARDEN traversal-escape] file_written={os.path.exists(pwn)}")


asyncio.run(flood(N_CONCURRENT))
asyncio.run(responsiveness())
asyncio.run(hardening())
print("LOADTEST_DONE")
