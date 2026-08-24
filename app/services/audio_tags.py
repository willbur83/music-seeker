"""Catalog metadata embedding for downloaded audio files."""

import asyncio
import logging
import os

import httpx

log = logging.getLogger("musicseeker")


async def download_cover(image_url: str, dest_path: str) -> bool:
    """Download album art from a URL to a temp file."""
    if not image_url:
        return False
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(resp.content)
            return True
    except Exception:
        return False


def _flac_tag_args(
    artist: str,
    title: str,
    album: str,
    track_number: int | None,
    album_artist: str | None,
    year: str | None,
) -> list[str]:
    tags = [
        f"--set-tag=ARTIST={artist}",
        f"--set-tag=TITLE={title}",
        f"--set-tag=ALBUM={album}",
    ]
    if track_number is not None:
        tags.append(f"--set-tag=TRACKNUMBER={track_number}")
    if album_artist:
        tags.append(f"--set-tag=ALBUMARTIST={album_artist}")
    if year:
        tags.append(f"--set-tag=DATE={year}")
    return tags


def _ffmpeg_metadata_args(
    artist: str,
    title: str,
    album: str,
    track_number: int | None,
    album_artist: str | None,
    year: str | None,
) -> list[str]:
    args = [
        "-metadata", f"artist={artist}",
        "-metadata", f"title={title}",
        "-metadata", f"album={album}",
    ]
    if track_number is not None:
        args.extend(["-metadata", f"track={track_number}"])
    if album_artist:
        args.extend(["-metadata", f"album_artist={album_artist}"])
    if year:
        args.extend(["-metadata", f"date={year}"])
    return args


async def _run_subprocess(cmd: list[str]) -> int:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    await proc.wait()
    return proc.returncode or 0


async def _embed_flac(
    file_path: str,
    *,
    artist: str,
    title: str,
    album: str,
    track_number: int | None,
    album_artist: str | None,
    year: str | None,
    image_url: str,
) -> bool:
    tag_cmd = [
        "metaflac",
        "--remove-all-tags",
        *_flac_tag_args(artist, title, album, track_number, album_artist, year),
        file_path,
    ]
    if await _run_subprocess(tag_cmd) != 0:
        log.warning("embed_catalog_metadata: metaflac tag write failed for %s", file_path)
        return False

    if not image_url:
        return True

    cover_path = f"{file_path}.cover.jpg"
    if not await download_cover(image_url, cover_path):
        return True

    try:
        if await _run_subprocess(
            ["metaflac", "--import-picture-from", cover_path, file_path]
        ) != 0:
            log.warning("embed_catalog_metadata: metaflac cover embed failed for %s", file_path)
    finally:
        if os.path.exists(cover_path):
            os.remove(cover_path)
    return True


async def _embed_id3_copy(
    file_path: str,
    *,
    artist: str,
    title: str,
    album: str,
    track_number: int | None,
    album_artist: str | None,
    year: str | None,
    image_url: str,
) -> bool:
    ext = os.path.splitext(file_path)[1].lstrip(".") or "mp3"
    cover_path = f"{file_path}.cover.jpg"
    has_cover = bool(image_url) and await download_cover(image_url, cover_path)
    tmp_out = f"{file_path}.tmp.{ext}"

    try:
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", file_path]
        if has_cover:
            ffmpeg_cmd.extend([
                "-i", cover_path,
                "-map", "0:a",
                "-map", "1:0",
                "-c:v", "mjpeg",
                "-id3v2_version", "3",
                "-metadata:s:v", "title=Album cover",
                "-metadata:s:v", "comment=Cover (front)",
            ])
        else:
            ffmpeg_cmd.extend(["-map", "0:a"])

        ffmpeg_cmd.extend([
            "-c:a", "copy",
            *_ffmpeg_metadata_args(artist, title, album, track_number, album_artist, year),
            tmp_out,
        ])

        if await _run_subprocess(ffmpeg_cmd) != 0 or not os.path.exists(tmp_out):
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            log.warning("embed_catalog_metadata: ffmpeg metadata write failed for %s", file_path)
            return False

        os.replace(tmp_out, file_path)
        return True
    finally:
        if os.path.exists(cover_path):
            os.remove(cover_path)
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


async def embed_catalog_metadata(
    file_path: str,
    *,
    artist: str,
    title: str,
    album: str,
    track_number: int | None = None,
    album_artist: str | None = None,
    year: str | None = None,
    image_url: str = "",
) -> bool:
    """Rewrite audio tags from catalog metadata without re-encoding audio."""
    if not os.path.exists(file_path):
        log.warning("embed_catalog_metadata: file not found: %s", file_path)
        return False

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".flac":
            return await _embed_flac(
                file_path,
                artist=artist,
                title=title,
                album=album,
                track_number=track_number,
                album_artist=album_artist,
                year=year,
                image_url=image_url,
            )
        return await _embed_id3_copy(
            file_path,
            artist=artist,
            title=title,
            album=album,
            track_number=track_number,
            album_artist=album_artist,
            year=year,
            image_url=image_url,
        )
    except Exception as exc:
        log.warning("embed_catalog_metadata failed for %s: %s", file_path, exc)
        return False
