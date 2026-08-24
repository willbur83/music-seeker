"""Tests for catalog metadata embedding (spec 17–20)."""

import asyncio
import hashlib
import subprocess
from unittest.mock import AsyncMock, patch

import pytest
from mutagen.easyid3 import EasyID3

from app.services.audio_tags import embed_catalog_metadata


def _run(coro):
    return asyncio.run(coro)


def _make_source_mp3(path: str) -> None:
  """Create a tiny MP3 with bogus Soulseek-style source tags."""
  subprocess.run(
      [
          "ffmpeg",
          "-y",
          "-f",
          "lavfi",
          "-i",
          "anullsrc=r=44100:cl=stereo",
          "-t",
          "0.1",
          "-c:a",
          "libmp3lame",
          "-metadata",
          "title=All The Small Things",
          "-metadata",
          "artist=Blink 182",
          "-metadata",
          "album=Billboard Top 100",
          "-metadata",
          "track=040",
          str(path),
      ],
      check=True,
      capture_output=True,
  )


def _read_mp3_tags(path: str) -> dict[str, str]:
    tags = EasyID3(path)
    return {key: tags[key][0] for key in tags.keys()}


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.fixture
def source_mp3(tmp_path):
    path = tmp_path / "source.mp3"
    _make_source_mp3(path)
    return path


# --- spec 17: catalog tags replace bogus source tags ---


def test_embed_catalog_metadata_rewrites_source_tags_to_catalog(source_mp3):
    before = _read_mp3_tags(source_mp3)
    assert before["title"] == "All The Small Things"
    assert before["artist"] == "Blink 182"
    assert before["album"] == "Billboard Top 100"
    assert before["tracknumber"] == "040"

    ok = _run(
        embed_catalog_metadata(
            str(source_mp3),
            artist="Blink-182",
            title="All The Small Things",
            album="Enema Of The State",
            track_number=8,
        )
    )

    assert ok is True
    after = _read_mp3_tags(source_mp3)
    assert after["title"] == "All The Small Things"
    assert after["artist"] == "Blink-182"
    assert after["album"] == "Enema Of The State"
    assert after["tracknumber"] == "8"


# --- spec 18: ffmpeg uses stream copy, no transcode ---


def test_embed_catalog_metadata_ffmpeg_uses_audio_copy(source_mp3):
    captured: list[list[str]] = []
    real_create = asyncio.create_subprocess_exec

    async def spy_create(*args, **kwargs):
        if args and args[0] == "ffmpeg":
            captured.append(list(args))
        return await real_create(*args, **kwargs)

    with patch("app.services.audio_tags.asyncio.create_subprocess_exec", side_effect=spy_create):
        ok = _run(
            embed_catalog_metadata(
                str(source_mp3),
                artist="Blink-182",
                title="All The Small Things",
                album="Enema Of The State",
                track_number=8,
            )
        )

    assert ok is True
    assert captured, "expected ffmpeg subprocess"
    ffmpeg_cmd = captured[0]
    copy_idx = ffmpeg_cmd.index("-c:a")
    assert ffmpeg_cmd[copy_idx + 1] == "copy"


# --- spec 19: missing optional artwork does not fail core tag write ---


def test_embed_catalog_metadata_succeeds_without_artwork(source_mp3):
    ok = _run(
        embed_catalog_metadata(
            str(source_mp3),
            artist="Blink-182",
            title="All The Small Things",
            album="Enema Of The State",
            track_number=8,
            image_url="",
        )
    )

    assert ok is True
    after = _read_mp3_tags(source_mp3)
    assert after["artist"] == "Blink-182"
    assert after["album"] == "Enema Of The State"


def test_embed_catalog_metadata_succeeds_when_cover_download_fails(source_mp3):
    with patch(
        "app.services.audio_tags.download_cover",
        new=AsyncMock(return_value=False),
    ):
        ok = _run(
            embed_catalog_metadata(
                str(source_mp3),
                artist="Blink-182",
                title="All The Small Things",
                album="Enema Of The State",
                track_number=8,
                image_url="https://example.invalid/cover.jpg",
            )
        )

    assert ok is True
    after = _read_mp3_tags(source_mp3)
    assert after["artist"] == "Blink-182"


# --- spec 20: metadata write failure leaves original file intact ---


def test_embed_catalog_metadata_failure_preserves_original_file(source_mp3):
    original_hash = _file_hash(source_mp3)
    original_tags = _read_mp3_tags(source_mp3)

    async def failing_subprocess(*args, **kwargs):
        if args and args[0] == "ffmpeg":
            proc = AsyncMock()
            proc.wait = AsyncMock(return_value=None)
            proc.returncode = 1
            return proc
        return await asyncio.create_subprocess_exec(*args, **kwargs)

    with patch("app.services.audio_tags.asyncio.create_subprocess_exec", side_effect=failing_subprocess):
        ok = _run(
            embed_catalog_metadata(
                str(source_mp3),
                artist="Blink-182",
                title="All The Small Things",
                album="Enema Of The State",
                track_number=8,
            )
        )

    assert ok is False
    assert _file_hash(source_mp3) == original_hash
    assert _read_mp3_tags(source_mp3) == original_tags
    assert not source_mp3.with_suffix(".mp3.tmp.mp3").exists()
