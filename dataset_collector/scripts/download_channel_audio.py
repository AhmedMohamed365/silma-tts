from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from yt_dlp import YoutubeDL

from common import append_jsonl, ensure_dir, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download channel videos as audio-only WAV files.")
    parser.add_argument("--channel-url", required=True, help="YouTube channel/videos URL")
    parser.add_argument("--channel-slug", required=True, help="Stable folder name for this channel")
    parser.add_argument("--output-root", default="data", help="Root output folder")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit videos (0 = all)")
    parser.add_argument("--audio-format", default="wav", choices=["wav", "mp3", "m4a"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent fragments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channel_slug = slugify(args.channel_slug)
    root = Path(args.output_root)
    channel_root = root / "channels" / channel_slug
    raw_audio_dir = ensure_dir(channel_root / "raw_audio")
    manifests_dir = ensure_dir(channel_root / "manifests")
    runs_dir = ensure_dir(channel_root / "runs")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(runs_dir / run_id)

    outtmpl = str(raw_audio_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": False,
        "ignoreerrors": True,
        "quiet": False,
        "concurrent_fragment_downloads": args.workers,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": args.audio_format,
                "preferredquality": "0",
            }
        ],
        "postprocessor_args": ["-ar", str(args.sample_rate), "-ac", "1"],
    }

    if args.max_videos > 0:
        ydl_opts["playlistend"] = args.max_videos

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(args.channel_url, download=True)

    entries = info.get("entries", []) if isinstance(info, dict) else []
    rows = []
    for item in entries:
        if not item:
            continue
        video_id = item.get("id")
        if not video_id:
            continue
        ext = args.audio_format
        audio_path = raw_audio_dir / f"{video_id}.{ext}"
        rows.append(
            {
                "id": f"{channel_slug}_{video_id}",
                "channel_slug": channel_slug,
                "video_id": video_id,
                "title": item.get("title", ""),
                "uploader": item.get("uploader", ""),
                "duration_sec": item.get("duration"),
                "source_url": f"https://www.youtube.com/watch?v={video_id}",
                "audio_path": str(audio_path),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    manifest_raw = manifests_dir / "manifest_raw.jsonl"
    append_jsonl(manifest_raw, rows)

    run_cfg = {
        "run_id": run_id,
        "stage": "download",
        "channel_url": args.channel_url,
        "channel_slug": channel_slug,
        "output_root": str(root),
        "max_videos": args.max_videos,
        "audio_format": args.audio_format,
        "sample_rate": args.sample_rate,
        "workers": args.workers,
        "downloaded_entries": len(rows),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Downloaded/registered {len(rows)} items for channel: {channel_slug}")
    print(f"Raw manifest: {manifest_raw}")


if __name__ == "__main__":
    main()
