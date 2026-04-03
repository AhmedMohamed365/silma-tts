from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import pipeline

from common import append_jsonl, ensure_dir, read_jsonl, slugify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe chunked audio into text.")
    parser.add_argument("--channel-slug", required=True)
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--model-id", default="facebook/omniASR-LLM-7B")
    parser.add_argument("--language", default="ar")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    channel_slug = slugify(args.channel_slug)
    root = Path(args.output_root)
    channel_root = root / "channels" / channel_slug
    chunk_manifest = channel_root / "manifests" / "manifest_chunks.jsonl"
    out_manifest = channel_root / "manifests" / "manifest_transcribed.jsonl"
    transcript_dir = ensure_dir(channel_root / "processed" / "transcripts")
    runs_dir = ensure_dir(channel_root / "runs")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ensure_dir(runs_dir / run_id)

    rows = read_jsonl(chunk_manifest)
    if not rows:
        raise SystemExit(f"No chunk manifest found at: {chunk_manifest}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model_id,
        torch_dtype=dtype,
        device=device,
    )

    out_rows = []
    for row in tqdm(rows, desc="transcribing"):
        audio_path = Path(row["audio_path"])
        if not audio_path.exists():
            continue

        pred = asr(str(audio_path), generate_kwargs={"language": args.language})
        text = pred["text"].strip()
        if not text:
            continue

        (transcript_dir / f"{row['id']}.txt").write_text(text, encoding="utf-8")
        out_rows.append(
            {
                "id": row["id"],
                "audio_path": row["audio_path"],
                "text": text,
                "duration_sec": row["duration_sec"],
                "sample_rate": row["sample_rate"],
                "channel_slug": row["channel_slug"],
                "video_id": row["video_id"],
                "source_url": row.get("source_url", ""),
                "language": args.language,
                "split": args.split,
            }
        )

    append_jsonl(out_manifest, out_rows)

    run_cfg = {
        "run_id": run_id,
        "stage": "transcribe",
        "channel_slug": channel_slug,
        "output_root": str(root),
        "model_id": args.model_id,
        "language": args.language,
        "split": args.split,
        "written": len(out_rows),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(out_rows)} transcribed rows")
    print(f"Manifest: {out_manifest}")


if __name__ == "__main__":
    main()
