"""Unit tests for Soulseek identity/quality selection helpers."""

from app.services.slskd_select import (
    MIN_IDENTITY_CONFIDENCE,
    select_best_candidate,
    score_identity,
    strip_track_prefix,
    unexpected_qualifiers,
)


def _file(
    filename: str,
    size: int = 5_000_000,
    bit_rate: int = 320,
) -> dict:
    return {"filename": filename, "size": size, "bitRate": bit_rate}


def _responses(username: str, *files: dict, **peer_kwargs) -> list:
    peer = {"username": username, "files": list(files)}
    peer.update(peer_kwargs)
    return [peer]


# --- normalization helpers ---


def test_strip_track_prefix_variants():
    assert strip_track_prefix("04 - Can't Stop.mp3") == "Can't Stop"
    assert strip_track_prefix("04. Can't Stop.mp3") == "Can't Stop"
    assert strip_track_prefix("1-04 Can't Stop.mp3") == "Can't Stop"


# --- Blink-182 / compilation album eligibility ---


def test_blink182_greatest_hits_request_matches_studio_album_path():
    path = r"\Blink-182\Enema Of The State\All The Small Things.mp3"
    identity = score_identity(
        "Blink-182",
        "All The Small Things",
        "Greatest Hits",
        path,
    )
    assert identity.accepted
    assert identity.title_matched
    assert identity.artist_matched
    assert not identity.album_matched


def test_blink182_selects_studio_path_despite_album_mismatch():
    responses = _responses(
        "peer1",
        _file(r"\Blink-182\Enema Of The State\All The Small Things.mp3", bit_rate=192),
    )
    result = select_best_candidate(
        responses,
        "mp3",
        "Blink-182",
        "All The Small Things",
        "Greatest Hits",
    )
    assert result is not None
    peer, chosen, _debug = result
    assert peer == "peer1"
    assert "All The Small Things" in chosen["filename"]


def test_matching_album_is_bonus_not_required():
    without_album = score_identity(
        "Blink-182",
        "All The Small Things",
        "Greatest Hits",
        r"\Blink-182\Enema Of The State\All The Small Things.mp3",
    )
    with_album = score_identity(
        "Blink-182",
        "All The Small Things",
        "Enema Of The State",
        r"\Blink-182\Enema Of The State\All The Small Things.mp3",
    )
    assert without_album.accepted
    assert with_album.accepted
    assert with_album.confidence > without_album.confidence
    assert with_album.album_matched
    assert not without_album.album_matched


# --- RHCP Can't Stop ---


def test_rhcp_cant_stop_prefers_correct_artist_path():
    responses = [
        {
            "username": "good_peer",
            "files": [
                _file(
                    r"\Red Hot Chili Peppers\By the Way\04 Can't Stop.mp3",
                    bit_rate=192,
                ),
            ],
            "uploadSpeed": 1000,
            "queueLength": 5,
        },
        {
            "username": "fast_peer",
            "files": [
                _file(
                    r"\Christchurch Fireworks Music\1.03 Can't Stop.mp3",
                    bit_rate=320,
                ),
            ],
            "uploadSpeed": 50000,
            "queueLength": 0,
            "hasFreeUploadSlot": True,
        },
    ]
    result = select_best_candidate(
        responses,
        "mp3",
        "Red Hot Chili Peppers",
        "Can't Stop",
        "By the Way",
    )
    assert result is not None
    peer, chosen, _debug = result
    assert peer == "good_peer"
    assert "Red Hot Chili Peppers" in chosen["filename"]


def test_rhcp_wrong_artist_path_below_identity_threshold():
    identity = score_identity(
        "Red Hot Chili Peppers",
        "Can't Stop",
        "By the Way",
        r"\Christchurch Fireworks Music\1.03 Can't Stop.mp3",
    )
    assert not identity.accepted
    assert identity.confidence < MIN_IDENTITY_CONFIDENCE


# --- Matchbox Twenty 3AM / remix ---


def test_matchbox_twenty_3am_studio_beats_ultimix_remix():
    responses = [
        {
            "username": "remix_peer",
            "files": [
                _file(
                    r"\DJ Ultimix\Matchbox 20 - 3AM (Ultimix Remix).mp3",
                    bit_rate=320,
                ),
            ],
            "uploadSpeed": 80000,
            "queueLength": 0,
            "hasFreeUploadSlot": True,
        },
        {
            "username": "studio_peer",
            "files": [
                _file(
                    r"\Matchbox Twenty\Yourself or Someone Like You\3AM.mp3",
                    bit_rate=192,
                ),
            ],
            "uploadSpeed": 2000,
            "queueLength": 8,
        },
    ]
    result = select_best_candidate(
        responses,
        "mp3",
        "Matchbox Twenty",
        "3AM",
        "Yourself or Someone Like You",
    )
    assert result is not None
    peer, chosen, _debug = result
    assert peer == "studio_peer"
    assert "Ultimix" not in chosen["filename"]


def test_matchbox_twenty_remix_only_returns_none():
    responses = _responses(
        "remix_peer",
        _file(r"\DJ Ultimix\Matchbox 20 - 3AM (Ultimix Remix).mp3", bit_rate=320),
        uploadSpeed=80000,
        hasFreeUploadSlot=True,
    )
    assert (
        select_best_candidate(
            responses,
            "mp3",
            "Matchbox Twenty",
            "3AM",
            "Yourself or Someone Like You",
        )
        is None
    )


# --- Live / version qualifiers ---


def test_requested_live_title_allows_live_candidate():
    path = r"\Artist\Album\Song (Live).mp3"
    assert unexpected_qualifiers(path, "Song (Live)") == []
    identity = score_identity("Artist", "Song (Live)", "Album", path)
    assert identity.accepted


def test_studio_request_rejects_live_remix_instrumental_karaoke():
    cases = [
        (r"\Artist\Album\Song (Live).mp3", "unexpected_live_qualifier"),
        (r"\Artist\Album\Song (Remix).mp3", "unexpected_remix_qualifier"),
        (r"\Artist\Album\Song (Instrumental).mp3", "unexpected_instrumental_qualifier"),
        (r"\Artist\Album\Song (Karaoke).mp3", "unexpected_karaoke_qualifier"),
    ]
    for path, expected_reason in cases:
        identity = score_identity("Artist", "Song", "Album", path)
        assert not identity.accepted
        assert identity.reject_reason == expected_reason


# --- Short titles ---


def test_short_title_3am_does_not_match_substring_in_unrelated_filename():
    identity = score_identity(
        "Matchbox Twenty",
        "3AM",
        "Yourself or Someone Like You",
        r"\Various\Dreaming At 3AM Sessions\Intro.mp3",
    )
    assert not identity.title_matched
    assert not identity.accepted


def test_short_title_one_does_not_match_someone():
    identity = score_identity(
        "U2",
        "One",
        "Achtung Baby",
        r"\Compilations\Someone Like You.mp3",
    )
    assert not identity.title_matched
    assert not identity.accepted


def test_short_title_one_matches_exact_filename():
    identity = score_identity(
        "U2",
        "One",
        "Achtung Baby",
        r"\U2\Achtung Baby\One.mp3",
    )
    assert identity.title_matched
    assert identity.accepted


# --- Identity beats quality ---


def test_correct_studio_192k_beats_wrong_artist_320k_remix():
    responses = [
        {
            "username": "remix_peer",
            "files": [
                _file(
                    r"\DJ Mix\Artist - Song (Remix).mp3",
                    bit_rate=320,
                ),
            ],
            "uploadSpeed": 100000,
            "queueLength": 0,
            "hasFreeUploadSlot": True,
        },
        {
            "username": "studio_peer",
            "files": [
                _file(
                    r"\Artist\Album\Song.mp3",
                    bit_rate=192,
                ),
            ],
            "uploadSpeed": 1000,
            "queueLength": 10,
        },
    ]
    result = select_best_candidate(
        responses,
        "mp3",
        "Artist",
        "Song",
        "Album",
    )
    assert result is not None
    peer, chosen, _debug = result
    assert peer == "studio_peer"
    assert chosen["bitRate"] == 192


def test_correct_artist_title_beats_compilation_with_better_peer_metrics():
    responses = [
        {
            "username": "compilation_peer",
            "files": [
                _file(
                    r"\90s Hits\04 Can't Stop.mp3",
                    bit_rate=320,
                ),
            ],
            "uploadSpeed": 90000,
            "queueLength": 0,
            "hasFreeUploadSlot": True,
        },
        {
            "username": "studio_peer",
            "files": [
                _file(
                    r"\Red Hot Chili Peppers\By the Way\04 Can't Stop.mp3",
                    bit_rate=192,
                ),
            ],
            "uploadSpeed": 1500,
            "queueLength": 6,
        },
    ]
    result = select_best_candidate(
        responses,
        "mp3",
        "Red Hot Chili Peppers",
        "Can't Stop",
        "By the Way",
    )
    assert result is not None
    peer, _chosen, _debug = result
    assert peer == "studio_peer"
