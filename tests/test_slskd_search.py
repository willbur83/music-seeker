"""Unit tests for progressive Soulseek search fallback."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.services import downloader
from app.services.downloader import _download_track_slskd
from app.services.slskd_select import search_attempts


def _file(filename: str, size: int = 5_000_000, bit_rate: int = 320) -> dict:
    return {"filename": filename, "size": size, "bitRate": bit_rate}


def _responses(username: str, *files: dict, **peer_kwargs) -> list:
    peer = {"username": username, "files": list(files)}
    peer.update(peer_kwargs)
    return [peer]


def _valid_mp3_response(
    username: str = "peer1",
    artist: str = "Artist",
    title: str = "Song",
    album: str = "Album",
) -> list:
    return _responses(
        username,
        _file(rf"\{artist}\{album}\{title}.mp3", bit_rate=320),
    )


def _flac_only_response(
    username: str = "peer1",
    artist: str = "Artist",
    title: str = "Song",
    album: str = "Album",
) -> list:
    return _responses(
        username,
        _file(rf"\{artist}\{album}\{title}.flac", size=30_000_000, bit_rate=1411),
    )


def _low_confidence_response(username: str = "peer1", title: str = "Song") -> list:
    return _responses(
        username,
        _file(rf"\Compilation\Hits\{title}.mp3", bit_rate=320),
    )


# --- search_attempts ---


def test_search_attempts_artist_title_title_only_title_album_order():
    attempts = search_attempts("Red Hot Chili Peppers", "Can't Stop", "By the Way")
    assert [a["strategy"] for a in attempts] == [
        "artist_title",
        "title_only",
        "title_album",
    ]
    assert attempts[0]["query"] == "Red Hot Chili Peppers Can't Stop"
    assert attempts[1]["query"] == "Can't Stop"
    assert attempts[2]["query"] == "Can't Stop By the Way"


def test_search_attempts_skips_title_album_when_album_empty():
    attempts = search_attempts("Artist", "Title", "")
    assert [a["strategy"] for a in attempts] == ["artist_title", "title_only"]


def test_search_attempts_dedupes_when_missing_artist_makes_artist_title_equal_title_only():
    attempts = search_attempts("", "Can't Stop", "By the Way")
    assert len(attempts) == 2
    assert attempts[0] == {"strategy": "artist_title", "query": "Can't Stop"}
    assert attempts[1] == {"strategy": "title_album", "query": "Can't Stop By the Way"}


def test_search_attempts_uses_primary_artist_only():
    attempts = search_attempts("Artist One, Artist Two", "Song", "Album")
    assert attempts[0]["query"] == "Artist One Song"


# --- _download_track_slskd progressive search ---


def _run_download(*args, **kwargs):
    return asyncio.run(_download_track_slskd(*args, **kwargs))


def _patch_download_env(monkeypatch, tmp_path):
    monkeypatch.setattr(downloader, "MUSIC_DIR", str(tmp_path))
    monkeypatch.setattr(downloader, "SLSKD_API_KEY", "test-key")


def _make_slskd_api_mock(responses_by_query: dict):
    """Build an AsyncMock for _slskd_api keyed by search query text."""
    search_ids: dict[str, str] = {}
    id_to_query: dict[str, str] = {}
    counter = {"n": 0}
    post_searches_calls: list[str] = []
    transfer_posts: list[tuple] = []

    async def mock_api(method: str, path: str, json_data: dict | None = None):
        if method == "POST" and path == "searches":
            query = (json_data or {}).get("searchText", "")
            post_searches_calls.append(query)
            counter["n"] += 1
            search_id = f"search-{counter['n']}"
            search_ids[query] = search_id
            id_to_query[search_id] = query
            return {"id": search_id}

        if method == "GET" and path.startswith("searches/") and path.endswith("/responses"):
            search_id = path.split("/")[1]
            query = id_to_query.get(search_id, "")
            return responses_by_query.get(query, [])

        if method == "GET" and path.startswith("searches/") and "/responses" not in path:
            return {"state": "Completed"}

        if method == "DELETE" and path.startswith("searches/"):
            return None

        if method == "POST" and path.startswith("transfers/downloads/"):
            peer = path.split("/")[-1]
            transfer_posts.append((peer, json_data))
            return None

        if method == "GET" and path.startswith("transfers/downloads/"):
            if not transfer_posts:
                return None
            _, queued = transfer_posts[-1]
            filename = queued[0].get("filename", "") if queued else ""
            return {
                "directories": [
                    {
                        "files": [
                            {"filename": filename, "state": "Completed, Succeeded"},
                        ],
                    },
                ],
            }

        raise AssertionError(f"Unexpected API call: {method} {path}")

    mock_api.post_searches_calls = post_searches_calls
    mock_api.transfer_posts = transfer_posts
    return mock_api


def test_first_strategy_success_posts_one_search_only(monkeypatch, tmp_path):
    _patch_download_env(monkeypatch, tmp_path)
    responses = {
        "Red Hot Chili Peppers Can't Stop": _valid_mp3_response(
            artist="Red Hot Chili Peppers",
            title="Can't Stop",
            album="By the Way",
        ),
    }
    mock = _make_slskd_api_mock(responses)

    with patch.object(downloader, "_slskd_api", mock), patch.object(
        asyncio, "sleep", new=AsyncMock()
    ), patch.object(
        downloader, "_find_completed_slskd_file", return_value=str(tmp_path / "Song.mp3")
    ), patch("shutil.move"):
        ok = _run_download(
            "Red Hot Chili Peppers",
            "Can't Stop",
            "By the Way",
            "mp3",
        )

    assert ok is True
    assert mock.post_searches_calls == ["Red Hot Chili Peppers Can't Stop"]
    assert len(mock.transfer_posts) == 1


def test_empty_first_search_falls_back_to_title_only(monkeypatch, tmp_path):
    _patch_download_env(monkeypatch, tmp_path)
    responses = {
        "Red Hot Chili Peppers Can't Stop": [],
        "Can't Stop": _valid_mp3_response(
            "title_peer",
            artist="Red Hot Chili Peppers",
            title="Can't Stop",
            album="By the Way",
        ),
    }
    mock = _make_slskd_api_mock(responses)

    with patch.object(downloader, "_slskd_api", mock), patch.object(
        asyncio, "sleep", new=AsyncMock()
    ), patch.object(
        downloader, "_find_completed_slskd_file", return_value=str(tmp_path / "Song.mp3")
    ), patch("shutil.move"):
        ok = _run_download(
            "Red Hot Chili Peppers",
            "Can't Stop",
            "By the Way",
            "mp3",
        )

    assert ok is True
    assert mock.post_searches_calls == [
        "Red Hot Chili Peppers Can't Stop",
        "Can't Stop",
    ]
    assert mock.transfer_posts[0][0] == "title_peer"


def test_flac_only_first_search_falls_back_to_broader_query(monkeypatch, tmp_path):
    _patch_download_env(monkeypatch, tmp_path)
    responses = {
        "Artist Song": _flac_only_response(),
        "Song": _valid_mp3_response("mp3_peer"),
    }
    mock = _make_slskd_api_mock(responses)

    with patch.object(downloader, "_slskd_api", mock), patch.object(
        asyncio, "sleep", new=AsyncMock()
    ), patch.object(
        downloader, "_find_completed_slskd_file", return_value=str(tmp_path / "Song.mp3")
    ), patch("shutil.move"):
        ok = _run_download("Artist", "Song", "Album", "mp3")

    assert ok is True
    assert mock.post_searches_calls == ["Artist Song", "Song"]
    assert mock.transfer_posts[0][0] == "mp3_peer"


def test_all_strategies_fail_without_queueing_download(monkeypatch, tmp_path):
    _patch_download_env(monkeypatch, tmp_path)
    responses = {
        "Artist Song": _flac_only_response(),
        "Song": _low_confidence_response(),
        "Song Album": [],
    }
    mock = _make_slskd_api_mock(responses)

    with patch.object(downloader, "_slskd_api", mock), patch.object(
        asyncio, "sleep", new=AsyncMock()
    ):
        ok = _run_download("Artist", "Song", "Album", "mp3")

    assert ok is False
    assert mock.post_searches_calls == ["Artist Song", "Song", "Song Album"]
    assert mock.transfer_posts == []


def test_duplicate_normalized_queries_search_executed_once(monkeypatch, tmp_path):
    _patch_download_env(monkeypatch, tmp_path)
    responses = {
        "Can't Stop": [],
        "Can't Stop By the Way": _valid_mp3_response(
            "album_peer",
            artist="Various Artists",
            title="Can't Stop",
            album="By the Way",
        ),
    }
    mock = _make_slskd_api_mock(responses)

    with patch.object(downloader, "_slskd_api", mock), patch.object(
        asyncio, "sleep", new=AsyncMock()
    ), patch.object(
        downloader, "_find_completed_slskd_file", return_value=str(tmp_path / "Song.mp3")
    ), patch("shutil.move"):
        ok = _run_download("", "Can't Stop", "By the Way", "mp3")

    assert ok is True
    assert mock.post_searches_calls == ["Can't Stop", "Can't Stop By the Way"]
    assert mock.transfer_posts[0][0] == "album_peer"
