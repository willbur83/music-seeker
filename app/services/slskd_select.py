"""Soulseek search-result identity and quality selection helpers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_LOG_TOP_N = 10


def _slskd_file_extension(filename: str) -> str:
    """Return the real file extension from a slskd path, case-insensitive."""
    basename = filename.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if "." not in basename:
        return ""
    return basename.rsplit(".", 1)[-1].lower()

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


def search_attempts(artist: str, title: str, album: str) -> list[dict]:
    """Return ordered Soulseek search queries, deduplicated by normalized text."""
    attempts: list[dict] = []
    seen_normalized: set[str] = set()

    def _add(strategy: str, query: str) -> None:
        normalized = normalize_query_text(query)
        if not normalized or normalized in seen_normalized:
            return
        seen_normalized.add(normalized)
        attempts.append({"strategy": strategy, "query": query})

    search_artist = artist.split(",")[0].strip() if artist else ""
    artist_title_query = f"{search_artist} {title}" if search_artist else title
    _add("artist_title", artist_title_query)
    _add("title_only", title)
    if album.strip():
        _add("title_album", f"{title} {album}")

    return attempts


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


def _bitrate_reason(bit_rate: int) -> str | None:
    if bit_rate >= 320:
        return "320kbps"
    if bit_rate >= 256:
        return "256kbps"
    if bit_rate >= 192:
        return "192kbps"
    if bit_rate > 0:
        return f"{bit_rate}kbps"
    return None


def _identity_reasons(identity: IdentityResult, quality: QualityResult | None = None) -> list[str]:
    reasons: list[str] = []
    if identity.title_matched:
        reasons.append("title_match")
    if identity.artist_matched:
        reasons.append("artist_path_match")
    if identity.album_matched:
        reasons.append("album_path_match")
    if quality is not None:
        bitrate_label = _bitrate_reason(quality.bit_rate)
        if bitrate_label:
            reasons.append(bitrate_label)
    return reasons


def _is_interesting_reject(entry: dict) -> bool:
    reason = entry.get("reject_reason")
    if not reason:
        return False
    if reason in ("format_mismatch", "too_small"):
        return False
    if reason == "low_identity_confidence":
        return True
    return reason.startswith("unexpected_") and reason.endswith("_qualifier")


def _log_selection_debug(
    strategy: str | None,
    query: str | None,
    peer_response_count: int,
    eligible_format: int,
    above_threshold: int,
    candidates: list[dict],
    rejected: list[dict],
    selected: dict | None,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return

    summary_parts = [
        f"responses={peer_response_count}",
        f"eligible_format={eligible_format}",
        f"above_confidence_threshold={above_threshold}",
    ]
    if strategy:
        summary_parts.insert(0, f"strategy={strategy}")
    if query:
        summary_parts.insert(1 if strategy else 0, f"query={query!r}")
    logger.debug("slskd_select attempt %s", " ".join(summary_parts))

    sorted_candidates = sorted(
        candidates,
        key=lambda entry: entry.get("quality_score", 0),
        reverse=True,
    )
    for entry in sorted_candidates[:_LOG_TOP_N]:
        identity = entry.get("identity", {})
        logger.debug(
            "slskd_select candidate peer=%s path=%r identity_score=%s quality_score=%s reasons=%s",
            entry.get("peer"),
            entry.get("filename"),
            identity.get("confidence"),
            entry.get("quality_score"),
            entry.get("reasons"),
        )

    interesting_rejects = [entry for entry in rejected if _is_interesting_reject(entry)]
    for entry in interesting_rejects[:_LOG_TOP_N]:
        logger.debug(
            "slskd_select rejected peer=%s path=%r reject_reason=%s",
            entry.get("peer"),
            entry.get("filename"),
            entry.get("reject_reason"),
        )

    if selected:
        logger.debug(
            "slskd_select selected_candidate peer=%s path=%r strategy=%s reason=best_quality_score",
            selected.get("peer"),
            selected.get("filename"),
            strategy,
        )
    else:
        logger.debug(
            "slskd_select no_candidate_selected strategy=%s fallback_continues",
            strategy,
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
    strategy: str | None = None,
    query: str | None = None,
) -> tuple[str | None, dict | None, dict]:
    """Pick the best Soulseek file after strict format, size, and identity filters."""
    debug: dict = {"candidates": [], "rejected": []}
    requested = requested_format.lower().lstrip(".")
    if not requested:
        _log_selection_debug(
            strategy, query, len(responses), 0, 0, debug["candidates"], debug["rejected"], None
        )
        return None, None, debug

    survivors: list[tuple[QualityResult, IdentityResult, str, dict]] = []
    eligible_format = 0

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

            eligible_format += 1
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
            entry["reasons"] = _identity_reasons(identity, quality)
            debug["candidates"].append(entry)
            survivors.append((quality, identity, peer, file_info))

    above_threshold = len(survivors)
    selected: dict | None = None

    if survivors:
        survivors.sort(key=lambda item: item[0].score, reverse=True)
        quality, identity, peer, file_info = survivors[0]
        selected = {
            "peer": peer,
            "filename": file_info.get("filename", ""),
            "identity_confidence": identity.confidence,
            "quality_score": quality.score,
            "reasons": _identity_reasons(identity, quality),
        }
        debug["selected"] = selected
        _log_selection_debug(
            strategy,
            query,
            len(responses),
            eligible_format,
            above_threshold,
            debug["candidates"],
            debug["rejected"],
            selected,
        )
        return peer, file_info, debug

    _log_selection_debug(
        strategy,
        query,
        len(responses),
        eligible_format,
        above_threshold,
        debug["candidates"],
        debug["rejected"],
        None,
    )
    return None, None, debug
