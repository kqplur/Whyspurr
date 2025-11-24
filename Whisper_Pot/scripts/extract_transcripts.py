"""Utilities for retrieving YouTube transcripts from Takeout watch history.

This script parses the exported watch history JSON file produced by
Google Takeout, extracts each unique YouTube video identifier and downloads
its transcript using the `youtube-transcript-api` package.  Transcripts are
stored as JSON documents with light metadata to help downstream retrieval
augmented generation (RAG) pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

try:  # pragma: no cover - dependency availability is environment specific
    from youtube_transcript_api import (
        CouldNotRetrieveTranscript,
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
        YouTubeTranscriptApi,
    )
    _YTA_IMPORT_ERROR: Optional[Exception] = None
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    # Defer the hard failure until a transcript download is attempted so that
    # ``--help`` and other lightweight invocations still succeed.
    CouldNotRetrieveTranscript = NoTranscriptFound = TranscriptsDisabled = VideoUnavailable = Exception  # type: ignore[assignment]
    YouTubeTranscriptApi = None  # type: ignore[assignment]
    _YTA_IMPORT_ERROR = exc

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoEntry:
    """Representation of a single watch history item."""

    video_id: str
    url: str
    title: Optional[str]
    watched_at: Optional[str]
    channel_name: Optional[str]


def _extract_video_id(url: str) -> Optional[str]:
    """Return the YouTube video identifier embedded in *url*.

    Supports regular watch URLs (``https://www.youtube.com/watch?v=...``)
    and short links (``https://youtu.be/...``).  Any additional query
    parameters are ignored.
    """

    parsed = urlparse(url)
    if parsed.netloc in {"www.youtube.com", "youtube.com", "m.youtube.com"}:
        query = parse_qs(parsed.query)
        video_id = query.get("v", [None])[0]
    elif parsed.netloc in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/") or None
    else:
        video_id = None

    if video_id:
        return video_id
    LOGGER.debug("Failed to extract video id from url: %s", url)
    return None


def _iter_video_entries(history: Iterable[dict]) -> Iterable[VideoEntry]:
    """Yield :class:`VideoEntry` objects from a watch history payload."""

    for entry in history:
        url = entry.get("titleUrl")
        if not url:
            continue
        video_id = _extract_video_id(url)
        if not video_id:
            continue

        subtitles = entry.get("subtitles") or []
        channel_name = None
        if subtitles and isinstance(subtitles, list):
            channel_info = subtitles[0]
            if isinstance(channel_info, dict):
                channel_name = channel_info.get("name")

        yield VideoEntry(
            video_id=video_id,
            url=url,
            title=entry.get("title"),
            watched_at=entry.get("time"),
            channel_name=channel_name,
        )


def fetch_transcript(video_id: str, languages: Optional[List[str]] = None) -> List[dict]:
    """Download the transcript for ``video_id``.

    Parameters
    ----------
    video_id:
        The YouTube video identifier.
    languages:
        Optional language codes.  If omitted, YouTube's default ordering is
        used.  When provided, the API will attempt the languages in order
        and fall back to automatic captions if necessary.
    """

    if YouTubeTranscriptApi is None:  # pragma: no cover - runtime safeguard
        raise RuntimeError(
            "youtube-transcript-api is required to download transcripts. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from _YTA_IMPORT_ERROR

    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except (NoTranscriptFound, TranscriptsDisabled):
        LOGGER.warning("Transcript unavailable for %s", video_id)
    except (VideoUnavailable, CouldNotRetrieveTranscript) as exc:
        LOGGER.warning("Failed to retrieve transcript for %s: %s", video_id, exc)
    except Exception:  # pragma: no cover - defensive logging
        LOGGER.exception("Unexpected error retrieving transcript for %s", video_id)

    return []


def save_transcript(
    entry: VideoEntry,
    transcript: List[dict],
    output_path: Path,
) -> None:
    """Persist *transcript* alongside metadata for *entry* to ``output_path``."""

    payload = {
        "video_id": entry.video_id,
        "url": entry.url,
        "title": entry.title,
        "channel_name": entry.channel_name,
        "watched_at": entry.watched_at,
        "transcript": transcript,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "history_file",
        type=Path,
        help="Path to the Google Takeout watch-history.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transcripts"),
        help="Directory to store transcript JSON files (default: ./transcripts)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Optional list of language codes to prioritise (e.g. en, en-US).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download transcripts even if the output file already exists.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of transcripts to download.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Configure logging verbosity (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.history_file.exists():
        LOGGER.error("History file does not exist: %%s", args.history_file)
        return 1

    try:
        history_data = json.loads(args.history_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.error("Failed to parse history file %%s: %%s", args.history_file, exc)
        return 1

    entries = list(_iter_video_entries(history_data))
    LOGGER.info("Loaded %d watch history entries", len(entries))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    downloaded = 0

    for entry in entries:
        if entry.video_id in seen:
            continue
        seen.add(entry.video_id)

        output_path = output_dir / f"{entry.video_id}.json"
        if output_path.exists() and not args.overwrite:
            LOGGER.info("Skipping existing transcript for %s", entry.video_id)
            continue

        transcript = fetch_transcript(entry.video_id, languages=args.languages)
        if not transcript:
            continue

        save_transcript(entry, transcript, output_path)
        downloaded += 1
        LOGGER.info("Saved transcript for %s -> %s", entry.video_id, output_path)

        if args.limit and downloaded >= args.limit:
            break

    LOGGER.info(
        "Finished. %d unique videos processed, %d transcripts saved.",
        len(seen),
        downloaded,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
