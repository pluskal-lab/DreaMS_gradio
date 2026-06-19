"""Tests for the examples registry."""

from pathlib import Path

from dreams_web.web.examples import EXAMPLES, ensure_example, get_example


def test_get_example_known_and_unknown() -> None:
    """get_example returns a registered example, or None for unknown ids."""
    assert get_example("drugs") is not None
    assert get_example("nope") is None


def test_example_urls_point_to_huggingface() -> None:
    """Each example's URL is a Hugging Face link ending in its filename."""
    for example in EXAMPLES:
        assert "huggingface.co" in example.url
        assert example.url.endswith(example.filename)


def test_ensure_example_returns_present_file_without_download(tmp_path: Path) -> None:
    """ensure_example returns an already-present file (no network access)."""
    example = EXAMPLES[0]
    (tmp_path / example.filename).write_bytes(b"x")
    assert ensure_example(example, tmp_path) == tmp_path / example.filename
