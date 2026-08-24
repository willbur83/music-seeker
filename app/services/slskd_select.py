"""Soulseek search-result identity and quality selection helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.downloader import _slskd_file_extension

# --- Identity tuning ---
MIN_IDENTITY_CONFIDENCE = 70.0
TITLE_MATCH_WEIGHT = 50.0
ARTIST_PATH_WEIGHT = 40.0
ALBUM_PATH_WEIGHT = 10.0

# --- Quality tuning ---
MIN_FILE_SIZE = 500_000
BITRATE_SCORE_CAP = 50
SIZE_SCORE_CAP = 30
UPLOAD_SLOT_BONUS = 10
UPLOAD_SPEED_SCORE_CAP = 20
QUEUE_SCORE_CAP = 10

# --- Text / title heuristics ---
SHORT_TITLE_MAX_LEN = 4
VERSION_QUALIFIERS = (
    "live",
    "remix",
    "instrumental",
    "karaoke",
    "acoustic",
    "cover",
    "tribute",
    "mashup",
    "bootleg",
    "redrum",
)

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_TRACK_PREFIX_RE = re.compile(
    r"^(?:\d{1,2}[-.\s]+)?\d{1,2}[-.\s]+",
    re.IGNORECASE,
)


@dataclass
class IdentityResult:
    confidence: float
    score: float
    accepted: bool
    reject_reason: str | None = None
    title_matched: bool = False
    artist_matched: bool = False
    album_matched: bool = False
    details: dict = field(default_factory=dict)


@dataclass
class QualityResult:
    score: float
    bit_rate: int = 0
    size: int = 0
    details: dict = field(default_factory=dict)


def normalize_query_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation, treat hyphens as spaces."""
    text = text.lower()
    text = text.replace("-", " ")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize_path_text(path: str) -> str:
    """Normalize a remote path/filename for token and qualifier matching."""
    path = path.replace("\\", "/")
    return normalize_query_text(path)


def strip_track_prefix(filename: str) -> str:
    """Remove leading track-number prefixes such as '04 -', '04.', or '1-04'."""
    base = filename_from_remote(filename)
    stem = base.rsplit(".", 1)[0] if "." in base else base
    return _TRACK_PREFIX_RE.sub("", stem).strip()


def filename_from_remote(path: str) -> str:
    """Return the basename from a Soulseek remote path (handles \\ and /)."""
    normalized = path.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def requested_qualifiers(title: str) -> set[str]:
    """Return version qualifiers explicitly present in the requested catalog title."""
    norm = normalize_query_text(title)
    found: set[str] = set()
    for qualifier in VERSION_QUALIFIERS:
        if re.search(rf"\b{re.escape(qualifier)}\b", norm):
            found.add(qualifier)
    return found


def unexpected_qualifiers(path: str, requested_title: str) -> list[str]:
    """Version qualifiers on the remote path that were not requested."""
    allowed = requested_qualifiers(requested_title)
    norm_path = normalize_path_text(path)
    unexpected: list[str] = []
    for qualifier in VERSION_QUALIFIERS:
        if qualifier in allowed:
            continue
        if re.search(rf"\b{re.escape(qualifier)}\b", norm_path):
            unexpected.append(qualifier)
    return unexpected


def _is_short_title(title: str) -> bool:
    norm = normalize_query_text(title)
    if not norm:
        return False
    tokens = norm.split()
    if len(tokens) == 1:
        return len(tokens[0]) <= SHORT_TITLE_MAX_LEN
    return len(norm) <= SHORT_TITLE_MAX_LEN


def _title_tokens(title: str) -> list[str]:
    return normalize_query_text(title).split()


def _title_matches(requested_title: str, remote_path: str) -> bool:
    if not requested_title.strip():
        return False

    norm_path = normalize_path_text(remote_path)

    if _is_short_title(requested_title):
        filename = filename_from_remote(remote_path)
        stem = strip_track_prefix(filename)
        stem = stem.rsplit(".", 1)[0] if "." in stem else stem
        norm_title = normalize_query_text(requested_title)
        norm_stem = normalize_query_text(stem)
        return bool(re.search(rf"\b{re.escape(norm_title)}\b", norm_stem))

    filename = filename_from_remote(remote_path)
    stem = strip_track_prefix(filename)
    stem_tokens = _title_tokens(stem)
    title_tokens = _title_tokens(requested_title)
    if len(stem_tokens) < len(title_tokens):
        return False
    return stem_tokens[-len(title_tokens) :] == title_tokens


def _artist_matches(requested_artist: str, remote_path: str) -> bool:
    if not requested_artist.strip():
        return True

    norm_path = normalize_path_text(remote_path)
    artist_tokens = _title_tokens(requested_artist.split(",")[0])
    if not artist_tokens:
        return True

    path_tokens = norm_path.split()
    if len(artist_tokens) == 1:
        token = artist_tokens[0]
        return bool(re.search(rf"\b{re.escape(token)}\b", norm_path))

    for start in range(len(path_tokens) - len(artist_tokens) + 1):
        if path_tokens[start : start + len(artist_tokens)] == artist_tokens:
            return True
    return False


def _album_matches(requested_album: str, remote_path: str) -> bool:
    if not requested_album.strip():
        return False

    norm_path = normalize_path_text(remote_path)
    album_tokens = _title_tokens(requested_album)
    if not album_tokens:
        return False

    path_tokens = norm_path.split()
    for start in range(len(path_tokens) - len(album_tokens) + 1):
        if path_tokens[start : start + len(album_tokens)] == album_tokens:
            return True
    return False


def score_identity(
    requested_artist: str,
    requested_title: str,
    requested_album: str,
    remote_path: str,
) -> IdentityResult:
    """Score how well a remote file path matches the requested catalog metadata."""
    unexpected = unexpected_qualifiers(remote_path, requested_title)
    if unexpected:
        qualifier = unexpected[0]
        return IdentityResult(
            confidence=0.0,
            score=0.0,
            accepted=False,
            reject_reason=f"unexpected_{qualifier}_qualifier",
            details={"unexpected_qualifiers": unexpected},
        )

    title_matched = _title_matches(requested_title, remote_path)
    artist_matched = _artist_matches(requested_artist, remote_path)
    album_matched = _album_matches(requested_album, remote_path)

    score = 0.0
    if title_matched:
        score += TITLE_MATCH_WEIGHT
    if artist_matched:
        score += ARTIST_PATH_WEIGHT
    if album_matched:
        score += ALBUM_PATH_WEIGHT

    confidence = score
    accepted = title_matched and confidence >= MIN_IDENTITY_CONFIDENCE

    return IdentityResult(
        confidence=confidence,
        score=score,
        accepted=accepted,
        title_matched=title_matched,
        artist_matched=artist_matched,
        album_matched=album_matched,
        details={
            "title_matched": title_matched,
            "artist_matched": artist_matched,
            "album_matched": album_matched,
        },
    )


def score_quality(file_dict: dict, peer_dict: dict) -> QualityResult:
    """Rank acceptable identity matches by transfer/audio quality signals."""
    bit_rate = int(file_dict.get("bitRate", 0) or 0)
    size = int(file_dict.get("size", 0) or 0)
    upload_speed = int(peer_dict.get("uploadSpeed", 0) or 0)
    queue_length = int(peer_dict.get("queueLength", 0) or 0)
    has_slot = bool(peer_dict.get("hasFreeUploadSlot"))

    score = 0.0
    score += min(bit_rate // 10, BITRATE_SCORE_CAP)
    score += min(size // 1_000_000, SIZE_SCORE_CAP)
    if has_slot:
        score += UPLOAD_SLOT_BONUS
    score += min(upload_speed // 1000, UPLOAD_SPEED_SCORE_CAP)
    score += max(0, QUEUE_SCORE_CAP - queue_length)

    return QualityResult(
        score=score,
        bit_rate=bit_rate,
        size=size,
        details={
            "bit_rate": bit_rate,
            "size": size,
            "upload_speed": upload_speed,
            "queue_length": queue_length,
            "has_free_upload_slot": has_slot,
        },
    )


def select_best_candidate(
    responses: list,
    requested_format: str,
    artist: str,
    title: str,
    album: str,
) -> tuple[str, dict, dict] | None:
    """Pick the best Soulseek file after strict format, size, and identity filters."""
    requested = requested_format.lower().lstrip(".")
    if not requested:
        return None

    survivors: list[tuple[QualityResult, IdentityResult, str, dict]] = []
    debug: dict = {"candidates": [], "rejected": []}

    for resp in responses:
        peer = resp.get("username", "")
        for file_info in resp.get("files", []):
            filename = file_info.get("filename", "")
            ext = _slskd_file_extension(filename)
            entry = {
                "peer": peer,
                "filename": filename,
                "extension": ext,
            }

            if ext != requested:
                entry["reject_reason"] = "format_mismatch"
                debug["rejected"].append(entry)
                continue

            size = int(file_info.get("size", 0) or 0)
            if size < MIN_FILE_SIZE:
                entry["reject_reason"] = "too_small"
                debug["rejected"].append(entry)
                continue

            identity = score_identity(artist, title, album, filename)
            entry["identity"] = {
                "confidence": identity.confidence,
                "accepted": identity.accepted,
                "reject_reason": identity.reject_reason,
                "details": identity.details,
            }

            if not identity.accepted:
                entry["reject_reason"] = identity.reject_reason or "low_identity_confidence"
                debug["rejected"].append(entry)
                continue

            quality = score_quality(file_info, resp)
            entry["quality_score"] = quality.score
            debug["candidates"].append(entry)
            survivors.append((quality, identity, peer, file_info))

    if not survivors:
        return None

    survivors.sort(key=lambda item: item[0].score, reverse=True)
    quality, identity, peer, file_info = survivors[0]
    debug["selected"] = {
        "peer": peer,
        "filename": file_info.get("filename", ""),
        "identity_confidence": identity.confidence,
        "quality_score": quality.score,
    }
    return peer, file_info, debug
